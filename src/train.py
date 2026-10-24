import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import torch
import tyro
from torch import nn, optim
from tqdm import tqdm

from dataset.dataset import build_dataloaders
from eval import validate
from jepa import JEPA, SIGReg
from logger import Logger
from utils import (
    _configure_device,
    _configure_dtype,
    _set_seed,
    load_checkpoint,
    save_checkpoint,
)


# TODO: add comments
# TODO: move to separate config file for train and eval config
@dataclass
class TrainConfig:
    # experiment params
    run_id: str = "baseline"
    wandb_enabled: bool = False
    wandb_project: str = "wm"
    exp_dir: str = "exp"
    save_dir: str = "checkpoints"
    resume_from: str | None = None
    """Path to checkpoint to resume training from."""
    eval_interval: int = 5
    """Evaluate every `eval_interval` epochs."""
    log_interval: int = 10
    """Log every `log_interval` steps."""
    save_interval: int = 5
    """Save every `save_interval` epochs."""
    device: str | None = None
    """Device to use for training ('cuda', 'mps', 'cpu', or None to auto-detect)."""
    dtype: str | None = None
    """Precision for training ('bf16', 'fp32', or None to auto-select bf16 on cuda, fp32 otherwise)."""

    # model params
    image_size: int = 224
    in_channels: int = 3
    action_dim: int = 2
    patch_size: int = 14
    hidden_dim: int = 192
    enc_depth: int = 12
    pred_depth: int = 6
    n_heads: int = 3
    mlp_ratio: int = 4
    enc_proj_hidden_dim: int = 2048
    pred_dropout: float = 0.0
    pred_mlp_dropout: float = 0.1
    pred_mlp_dim: int = 2048
    pred_proj_hidden_dim: int = 2048

    # training params
    batch_size: int = 128
    num_epochs: int = 10
    lr: float = 5e-5
    weight_decay: float = 1e-3
    seed: int = 42
    frameskip: int = 5
    num_frames: int = 4
    sigreg_weight: float = 0.1

    # dataset params
    data_path: str = "data/tworooms"
    val_ratio: float = 0.1
    num_workers: int = 0


def main(config: TrainConfig):
    device = _configure_device(config.device)
    dtype = _configure_dtype(device, config.dtype)
    _set_seed(config.seed)

    run_dir = Path(config.exp_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / config.save_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # update action dim to account for frameskip
    # TODO: there might be a better way to do this
    config.action_dim = config.action_dim * config.frameskip

    logger = Logger(
        project=config.wandb_project,
        run_name=config.run_id,
        config=asdict(config) if is_dataclass(config) else vars(config),
        enabled=config.wandb_enabled,
        local_dir=run_dir,
    )

    # train data, val data
    trainloader, valloader = build_dataloaders(
        data_path=config.data_path,
        batch_size=config.batch_size,
        num_frames=config.num_frames,
        frameskip=config.frameskip,
        val_ratio=config.val_ratio,
        image_size=config.image_size,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=torch.cuda.is_available(),
    )
    if len(trainloader) <= 0:
        raise RuntimeError("Training dataloader has zero batches.")

    # model, sigreg
    model = JEPA(
        image_size=config.image_size,
        in_channels=config.in_channels,
        action_dim=config.action_dim,
        patch_size=config.patch_size,
        num_frames=config.num_frames,
        enc_depth=config.enc_depth,
        hidden_dim=config.hidden_dim,
        n_heads=config.n_heads,
        enc_proj_hidden_dim=config.enc_proj_hidden_dim,
        mlp_ratio=config.mlp_ratio,
        pred_depth=config.pred_depth,
        pred_proj_hidden_dim=config.pred_proj_hidden_dim,
        pred_mlp_dim=config.pred_mlp_dim,
    )
    model.to(device)
    sigreg = SIGReg(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    logger.log({
        "device": str(device),
        "dtype": str(dtype),
        "train_size": len(trainloader),
        "val_size": len(valloader),
        "parameters": sum(p.numel() for p in model.parameters()),
    })

    start_epoch = 0
    optimizer_step = 0
    if config.resume_from is not None:
        ckpt_path = Path(config.resume_from)
        if not ckpt_path.is_file():
            candidate = checkpoint_dir / config.resume_from
            if candidate.is_file():
                ckpt_path = candidate
            else:
                raise FileNotFoundError(
                    f"Checkpoint file not found at '{ckpt_path}' or '{candidate}'"
                )
        ckpt = load_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        optimizer_step = ckpt.get("step", 0)
        print(
            f"[checkpoint] Resumed from {ckpt_path} at epoch {start_epoch}, step {optimizer_step}"
        )

    model.train()
    train_start = time.perf_counter()

    for epoch in range(start_epoch, config.num_epochs):
        progress = tqdm(
            trainloader, desc=f"train[{config.run_id}|{epoch}]", unit="batch", dynamic_ncols=True
        )

        running_total_loss = 0.0
        running_pred_loss = 0.0
        running_sigreg_loss = 0.0
        running_steps = 0

        for batch in progress:
            frames = batch["frames"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=(device.type == "cuda" and dtype == torch.bfloat16),
            ):
                # preds (B, T-1, D) | embs (B, T, D)
                preds, embs = model(frames, actions)

                prediction_loss = nn.functional.mse_loss(preds, embs[:, 1:])

                # permute to (T, B, D) for sigreg
                temporal_embs = embs.transpose(0, 1)
                sigreg_loss = sigreg(temporal_embs)

                total_loss = prediction_loss + config.sigreg_weight * sigreg_loss

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm).item():
                raise FloatingPointError("Training produced a non-finite gradient norm")

            optimizer.step()
            optimizer.zero_grad()

            optimizer_step += 1

            total_item = total_loss.item()
            pred_item = prediction_loss.item()
            sigreg_item = sigreg_loss.item()

            running_total_loss += total_item
            running_pred_loss += pred_item
            running_sigreg_loss += sigreg_item
            running_steps += 1

            avg_total_loss = running_total_loss / running_steps
            progress.set_postfix(
                loss=f"{total_item:.4f}",
                avg_loss=f"{avg_total_loss:.4f}",
            )

            if optimizer_step % config.log_interval == 0 and running_steps > 0:
                step_metrics = {
                    "train_loss/total": running_total_loss / running_steps,
                    "train_loss/prediction_loss": running_pred_loss / running_steps,
                    "train_loss/sigreg_loss": running_sigreg_loss / running_steps,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                    "train/grad_norm": grad_norm,
                }
                logger.log(step_metrics, step=optimizer_step)
                running_total_loss = 0.0
                running_pred_loss = 0.0
                running_sigreg_loss = 0.0
                running_steps = 0

        progress.close()

        if (epoch + 1) % config.eval_interval == 0 or epoch == config.num_epochs - 1:
            val_metrics = validate(
                model=model,
                dataloader=valloader,
                sigreg=sigreg,
                config=config,
                device=device,
                dtype=dtype,
            )
            logger.log(val_metrics, step=optimizer_step)
            print(
                f"[val][epoch {epoch}] total_loss={val_metrics['val_loss/total']:.4f} "
                f"pred_loss={val_metrics['val_loss/prediction_loss']:.4f} "
                f"cos_sim={val_metrics['val/cosine_similarity']:.4f}"
            )
            model.train()

        if (epoch + 1) % config.save_interval == 0:
            saved_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=optimizer_step,
                save_dir=checkpoint_dir,
                config=asdict(config),
            )
            print(f"[checkpoint] Saved epoch {epoch} to {saved_path}")

    # Save final checkpoint and summary metrics
    saved_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config.num_epochs - 1 if config.num_epochs > 0 else 0,
        step=optimizer_step,
        save_dir=checkpoint_dir,
        filename="checkpoint_final.pt",
        config=asdict(config),
    )
    print(f"[checkpoint] Saved final model to {saved_path}")
    elapsed = max(1e-6, time.perf_counter() - train_start)
    logger.log(
        {
            "train/elapsed_seconds": elapsed,
            "train/optimizer_steps_completed": float(optimizer_step),
            "train/optimizer_steps_ps": float(optimizer_step / elapsed),
        },
        step=optimizer_step,
    )

    # Save run summary in experiment directory
    summary = {
        "run_id": config.run_id,
        "device": str(device),
        "dtype": str(dtype),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_size": len(trainloader),
        "val_size": len(valloader),
        "optimizer_steps_completed": optimizer_step,
        "elapsed_seconds": elapsed,
        "optimizer_steps_ps": float(optimizer_step / elapsed),
        "final_val_loss": val_metrics.get("val_loss/total", None),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    logger.finish()

    print(f"[done] optimizer_steps={optimizer_step} elapsed_seconds={elapsed:.1f}")


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)
    main(config)
