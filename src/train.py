import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
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
    save_run_summary,
)


def prepare_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    image_size: int | tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transfer batch to device and perform GPU-accelerated float/resize transforms."""
    frames = batch["frames"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)

    if frames.dtype == torch.uint8:
        target_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W).float().div_(255.0)
        if target_size is not None and (H, W) != target_size:
            frames = F.interpolate(frames, size=target_size, mode="bilinear", align_corners=False)
        target_h, target_w = target_size if target_size else (H, W)
        frames = frames.view(B, T, C, target_h, target_w)

    return frames, actions


@dataclass
class TrainConfig:
    # experiment params
    run_id: str = "baseline"
    """Identifier for the run."""
    wandb_enabled: bool = True
    """Enable logging to Weights & Biases."""
    wandb_project: str = "wm"
    """Weights & Biases project name."""
    exp_dir: str = "exp"
    """Directory to save experiment results."""
    save_dir: str = "checkpoints"
    """Directory to save checkpoints."""
    resume_from: str | None = None
    """Path or filename to resume training from (searches locally and in Hugging Face)."""
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
    hf_sync: bool = True
    """Sync checkpoints and summary to Hugging Face Hub if configured."""

    # model params
    image_size: int = 224
    """Input image size."""
    in_channels: int = 3
    """Number of input channels."""
    action_dim: int = 2
    """Dimension of action space."""
    patch_size: int = 14
    """Patch size for ViT encoder."""
    hidden_dim: int = 192
    """Embedding dimension."""
    enc_depth: int = 12
    """Depth of ViT encoder."""
    pred_depth: int = 6
    """Depth of predictor."""
    n_heads: int = 3
    """Number of attention heads."""
    mlp_ratio: int = 4
    """MLP ratio in transformer."""
    enc_proj_hidden_dim: int = 2048
    """Hidden dimension of projection."""
    pred_dropout: float = 0.0
    """Dropout in the attention unit of predictor."""
    pred_emb_dropout: float = 0.1
    """Input embedding dropout in the predictor."""
    pred_mlp_dim: int = 2048
    """MLP dimension in the predictor."""
    pred_proj_hidden_dim: int = 2048
    """Hidden dimension of projection."""

    # training params
    batch_size: int = 128
    """Batch size."""
    num_epochs: int = 10
    """Number of epochs."""
    lr: float = 5e-5
    """Learning rate."""
    weight_decay: float = 1e-3
    """Weight decay."""
    seed: int = 42
    """Random seed."""
    frameskip: int = 5
    """Frame skip."""
    num_frames: int = 4
    """Number of frames."""
    sigreg_weight: float = 0.1
    """Weight for self-information-gain regularization."""

    # dataset params
    data_path: str = "data/tworooms"
    """Path to the dataset."""
    val_ratio: float = 0.1
    """Fraction of data to use for validation."""
    num_workers: int = 4
    """Number of workers for dataloader."""


def main(config: TrainConfig):
    device = _configure_device(config.device)
    dtype = _configure_dtype(device, config.dtype)
    _set_seed(config.seed)

    run_dir = Path(config.exp_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / config.save_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # update action dim to account for frameskip
    config.action_dim = config.action_dim * config.frameskip

    logger = Logger(
        project=config.wandb_project,
        run_name=config.run_id,
        config=asdict(config) if is_dataclass(config) else vars(config),
        enabled=config.wandb_enabled,
        local_dir=run_dir,
    )

    # train data, val data
    use_gpu_transforms = device.type == "cuda"
    trainloader, valloader = build_dataloaders(
        data_path=config.data_path,
        batch_size=config.batch_size,
        num_frames=config.num_frames,
        frameskip=config.frameskip,
        val_ratio=config.val_ratio,
        image_size=config.image_size if not use_gpu_transforms else None,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=torch.cuda.is_available(),
        return_uint8=use_gpu_transforms,
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
        pred_dropout=config.pred_dropout,
        pred_emb_dropout=config.pred_emb_dropout,
    )
    model.to(device)
    sigreg = SIGReg(device)

    if device.type == "cuda":
        model = torch.compile(model)
        sigreg = torch.compile(sigreg)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        fused=(device.type == "cuda"),
    )
    logger.log({
        "device": str(device),
        "dtype": str(dtype),
        "train_size": len(trainloader),
        "val_size": len(valloader),
        "parameters": sum(p.numel() for p in model.parameters()),
    })

    start_epoch = 0
    optimizer_step = 0
    val_metrics = {}

    if config.resume_from is not None:
        ckpt = load_checkpoint(
            checkpoint_path=config.resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
            save_dir=checkpoint_dir,
            run_id=config.run_id,
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        optimizer_step = ckpt.get("step", 0)
        print(
            f"[checkpoint] Resumed from {config.resume_from} at epoch {start_epoch}, step {optimizer_step}"
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
            frames, actions = prepare_batch(batch, device=device, image_size=config.image_size)

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
            optimizer.zero_grad(set_to_none=True)

            optimizer_step += 1

            running_total_loss += total_loss.detach()
            running_pred_loss += prediction_loss.detach()
            running_sigreg_loss += sigreg_loss.detach()
            running_steps += 1

            if optimizer_step % config.log_interval == 0 and running_steps > 0:
                avg_total_loss = (running_total_loss / running_steps).item()
                avg_pred_loss = (running_pred_loss / running_steps).item()
                avg_sigreg_loss = (running_sigreg_loss / running_steps).item()

                progress.set_postfix(
                    total_loss=f"{avg_total_loss:.4f}",
                    pred_loss=f"{avg_pred_loss:.4f}",
                    sigreg_loss=f"{avg_sigreg_loss:.4f}",
                )

                step_metrics = {
                    "train_loss/total": avg_total_loss,
                    "train_loss/prediction_loss": avg_pred_loss,
                    "train_loss/sigreg_loss": avg_sigreg_loss,
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
                run_id=config.run_id,
                sync=config.hf_sync,
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
        run_id=config.run_id,
        sync=config.hf_sync,
        async_upload=False,
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

    # Save run summary in experiment directory and upload artifacts
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
    save_run_summary(run_dir, summary, run_id=config.run_id, sync=config.hf_sync)
    logger.finish()

    print(f"[done] optimizer_steps={optimizer_step} elapsed_seconds={elapsed:.1f}")


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)
    main(config)
