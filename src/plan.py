from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader

from dataset.dataset import build_dataloaders
from jepa import JEPA
from utils import _configure_device, _configure_dtype, _set_seed, prepare_batch


@dataclass
class PlanConfig:
    # Checkpoint and dataset setup
    checkpoint_path: Optional[str] = None
    """Path to trained model checkpoint containing weights and architecture config."""
    data_path: Optional[str] = None
    """Path to dataset directory / h5 files. Defaults to checkpoint's dataset if None."""
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 42
    device: Optional[str] = None
    dtype: Optional[str] = None
    num_eval_batches: Optional[int] = None
    """Number of batches to evaluate planning on (None for all)."""

    # CEM planning hyperparameters
    iterations: int = 10
    """Number of CEM optimization iterations per planning step."""
    candidates: int = 300
    """Number of candidate action sequences sampled per iteration."""
    elite_ratio: float = 0.1
    """Top fraction of candidate sequences selected as elites."""
    horizon: int = 5
    """Planning horizon (number of future action steps)."""
    smoothing: float = 0.1
    """Momentum / smoothing factor for distribution parameter updates."""
    action_min: float = -1.0
    """Minimum valid action value."""
    action_max: float = 1.0
    """Maximum valid action value."""


def planning_cost(predicted_latents: torch.Tensor, goal_latent: torch.Tensor) -> torch.Tensor:
    """Compute planning cost based on terminal latent distance to goal.

    Args:
        predicted_latents: (B, S, T, D) predicted latent trajectories.
        goal_latent: (B, D) target goal latent embedding.

    Returns:
        (B, S) trajectory costs
    """
    assert predicted_latents.ndim == 4, f"predicted_latents.ndim: {predicted_latents.ndim}"
    assert goal_latent.ndim == 2, f"goal_latent.ndim: {goal_latent.ndim}"

    terminal_pred = predicted_latents[:, :, -1]
    goal_latent = goal_latent.unsqueeze(1)  # (B, 1, D)

    # Cost is MSE / squared Euclidean distance: ||z_H - z_g||^2
    cost = torch.sum((terminal_pred - goal_latent) ** 2, dim=-1)  # (B, S)
    return cost


def init_action_dist(batch_size: int, horizon: int, action_dim: int, device: torch.device):
    mean = torch.zeros((batch_size, horizon, action_dim), device=device)
    std = torch.ones((batch_size, horizon, action_dim), device=device)
    return mean, std


@torch.inference_mode()
def cem_optimize(
    model: JEPA,
    init_obs: torch.Tensor,
    goal_obs: torch.Tensor,
    horizon: int = 5,
    candidates: int = 300,
    elite_ratio: float = 0.1,
    iterations: int = 10,
    smoothing: float = 0.1,
    action_dim: int = 2,
    action_min: float = -1.0,
    action_max: float = 1.0,
    context_actions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute Cross-Entropy Method (CEM) optimization for a batch of environments.

    Args:
        model: Trained world model (JEPA).
        init_obs: (B, H_ctx, D) or (B, D) initial context frames.
        goal_obs: (B, D) target goal embedding.
        horizon: Planning horizon steps T.
        candidates: Number of action candidates S sampled per iteration.
        elite_ratio: Ratio of elite samples retained.
        iterations: Number of CEM iterations.
        smoothing: Smoothing factor for distribution updates.
        action_dim: Dimension of each action.
        action_min: Lower bound for action clipping.
        action_max: Upper bound for action clipping.
        context_actions: Optional (B, H_ctx - 1, A) past actions during history context.

    Returns:
        best_actions: (B, T, A) optimal action sequence for each batch item.
        best_cost: (B,) final cost achieved by the best candidate.
    """
    assert init_obs.ndim in (2, 3), f"init_obs.ndim must be 2 or 3, got: {init_obs.ndim}"
    if init_obs.ndim == 2:
        init_obs = init_obs.unsqueeze(1)  # (B, 1, D)
    assert goal_obs.ndim == 2, f"goal_obs.ndim: {goal_obs.ndim}"

    device = init_obs.device
    B = init_obs.shape[0]
    # min is 2 because `.std()` computes bessel's correction (N-1) in the denominator
    num_elites = max(2, int(candidates * elite_ratio))

    mean, std = init_action_dist(B, horizon, action_dim, device)

    best_actions = mean.clone()
    best_cost = torch.full((B,), float("inf"), device=device)

    # Expand context observation and actions across candidate dimension S: (B, S, H_ctx, D)
    init_obs_expanded = init_obs.unsqueeze(1).expand(-1, candidates, -1, -1)
    if context_actions is not None:
        ctx_actions_expanded = context_actions.unsqueeze(1).expand(-1, candidates, -1, -1)
    else:
        ctx_actions_expanded = None

    for _ in range(iterations):
        # sample candidate action sequences: (B, S, T, A)
        eps = torch.randn((B, candidates, horizon, action_dim), device=device)
        actions = mean.unsqueeze(1) + std.unsqueeze(1) * eps
        actions = torch.clamp(actions, action_min, action_max)

        # evaluate candidates via model rollout
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            pred_latents = model.predict_rollout(
                init_obs=init_obs_expanded,
                actions=actions,
                context_actions=ctx_actions_expanded,
            )

        costs = planning_cost(pred_latents, goal_obs)

        # select top elite action sequences (smallest cost)
        top_costs, elite_indices = torch.topk(costs, k=num_elites, dim=1, largest=False)

        # update overall best
        batch_best_idx = elite_indices[:, 0]
        batch_indices = torch.arange(B, device=device)
        iter_best_actions = actions[batch_indices, batch_best_idx]
        iter_best_cost = top_costs[:, 0]

        improved_mask = iter_best_cost < best_cost
        best_cost[improved_mask] = iter_best_cost[improved_mask]
        best_actions[improved_mask] = iter_best_actions[improved_mask]

        # gather elite actions: (B, num_elites, T, A)
        elite_indices_expanded = (
            elite_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, horizon, action_dim)
        )
        elite_actions = torch.gather(actions, 1, elite_indices_expanded)

        # fit new distribution parameters
        new_mean = elite_actions.mean(dim=1)
        new_std = elite_actions.std(dim=1).clamp(min=1e-3)

        # apply momentum smoothing
        mean = smoothing * new_mean + (1.0 - smoothing) * mean
        std = smoothing * new_std + (1.0 - smoothing) * std

    return best_actions, best_cost


@torch.inference_mode()
def plan(
    model: JEPA,
    valloader: DataLoader,
    iterations: int = 10,
    candidates: int = 300,
    elite_ratio: float = 0.1,
    horizon: int = 5,
    history_len: Optional[int] = None,
    smoothing: float = 0.1,
    action_min: float = -1.0,
    action_max: float = 1.0,
    num_eval_batches: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    """Plan optimized action sequences and evaluate against validation trajectories.

    Args:
        model: Trained JEPA model.
        valloader: Evaluation dataloader.
        iterations: Number of CEM iterations.
        candidates: Number of candidates per step.
        elite_ratio: Ratio of candidates selected as elites.
        horizon: Action planning horizon.
        smoothing: Smoothing parameter.
        action_min: Minimum valid action bound.
        action_max: Maximum valid action bound.
        num_eval_batches: Maximum number of batches to evaluate (None for all).
        device: Device to run evaluation on.
        dtype: Data type for computation.

    Returns:
        Dictionary of planning metrics.
    """
    model.eval()

    total_cem_cost = 0.0
    total_gt_cost = 0.0
    total_action_l1 = 0.0
    total_action_cos = 0.0
    num_samples = 0

    image_size = getattr(model.encoder, "image_size", None)

    for batch_idx, batch in enumerate(valloader):
        if num_eval_batches is not None and batch_idx >= num_eval_batches:
            break

        frames, actions = prepare_batch(batch, device=device, image_size=image_size)
        B, T = frames.shape[:2]
        action_dim = actions.shape[-1]

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and dtype == torch.bfloat16),
        ):
            latents, _ = model.encode(frames, actions)  # (B, T, D)

            # Context observation: exactly history_len frames
            init_obs_latent = latents[:, :history_len]  # (B, history_len, D)

            # Goal latent is at index history_len - 1 + horizon
            goal_idx = history_len - 1 + horizon
            goal_latent = latents[:, goal_idx]  # (B, D)

            # Ground truth actions for rollout
            gt_ctx_actions = actions[:, : history_len - 1].unsqueeze(1) if history_len > 1 else None
            gt_actions_seq = actions[:, history_len - 1 : goal_idx].unsqueeze(
                1
            )  # (B, 1, horizon, A)

            gt_init_expanded = init_obs_latent.unsqueeze(1)  # (B, 1, history_len, D)
            gt_pred_latents = model.predict_rollout(
                init_obs=gt_init_expanded,
                actions=gt_actions_seq,
                context_actions=gt_ctx_actions,
            )
            gt_cost = planning_cost(gt_pred_latents, goal_latent).squeeze(1)  # (B,)

        best_actions, best_costs = cem_optimize(
            model=model,
            init_obs=init_obs_latent,
            goal_obs=goal_latent,
            horizon=horizon,
            candidates=candidates,
            elite_ratio=elite_ratio,
            iterations=iterations,
            smoothing=smoothing,
            action_dim=action_dim,
            action_min=action_min,
            action_max=action_max,
            context_actions=actions[:, : history_len - 1] if history_len > 1 else None,
        )

        gt_actions_target = actions[:, history_len - 1 : goal_idx]
        gt_actions_flat = gt_actions_target.flatten(1)
        best_actions_flat = best_actions.flatten(1)
        l1_diff = torch.abs(best_actions - gt_actions_target).mean(dim=(1, 2))
        cos_sim = F.cosine_similarity(best_actions_flat, gt_actions_flat, dim=-1)

        total_cem_cost += best_costs.sum().item()
        total_gt_cost += gt_cost.sum().item()
        total_action_l1 += l1_diff.sum().item()
        total_action_cos += cos_sim.sum().item()
        num_samples += B

        print(
            f"[plan batch {batch_idx:3d}] CEM cost: {best_costs.mean().item():.4f} | "
            f"GT cost: {gt_cost.mean().item():.4f} | "
            f"Action cos-sim to GT: {cos_sim.mean().item():.4f}"
        )

    avg_cem_cost = total_cem_cost / max(1, num_samples)
    avg_gt_cost = total_gt_cost / max(1, num_samples)

    metrics = {
        "plan/cem_terminal_cost": avg_cem_cost,
        "plan/gt_terminal_cost": avg_gt_cost,
        "plan/action_l1_to_gt": total_action_l1 / max(1, num_samples),
        "plan/action_cosine_to_gt": total_action_cos / max(1, num_samples),
        "plan/num_samples": num_samples,
    }
    return metrics


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[JEPA, Dict[str, Any]]:
    """Instantiate and load JEPA model using architecture config saved in the checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint.get("config", {}) or {}

    import inspect

    sig = inspect.signature(JEPA.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    model_kwargs = {k: v for k, v in saved_cfg.items() if k in valid_params}

    model = JEPA(**model_kwargs)
    state_dict = {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in checkpoint["model"].items()
    }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, saved_cfg


def main(config: PlanConfig):
    device = _configure_device(config.device)
    dtype = _configure_dtype(device, config.dtype)
    _set_seed(config.seed)

    if config.checkpoint_path and Path(config.checkpoint_path).exists():
        print(f"[plan] Loading model and config from checkpoint: {config.checkpoint_path}")
        model, saved_cfg = load_model_from_checkpoint(config.checkpoint_path, device=device)
    else:
        print("[plan] No valid checkpoint provided; initializing default JEPA model.")
        model = JEPA().to(device)
        model.eval()

    # load history_len + horizon frames for each episode in the validation set
    frameskip = saved_cfg.get("frameskip", 1)
    data_path = saved_cfg.get("data_path", config.data_path)
    history_len = model.history_len
    num_frames = history_len + config.horizon

    print(
        f"[plan] Setup: device={device}, data_path={data_path}, frameskip={frameskip}, "
        f"history_len={history_len}, horizon={config.horizon}, num_frames={num_frames}, "
        f"candidates={config.candidates}, iterations={config.iterations}"
    )

    _, valloader = build_dataloaders(
        data_path=data_path,
        batch_size=config.batch_size,
        num_frames=num_frames,
        frameskip=frameskip,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    metrics = plan(
        model=model,
        valloader=valloader,
        iterations=config.iterations,
        candidates=config.candidates,
        elite_ratio=config.elite_ratio,
        horizon=config.horizon,
        history_len=history_len,
        smoothing=config.smoothing,
        action_min=config.action_min,
        action_max=config.action_max,
        num_eval_batches=config.num_eval_batches,
        device=device,
        dtype=dtype,
    )
    print("\n" + "=" * 50)
    print("[plan] Evaluation completed successfully! Final Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.6f}")
        else:
            print(f"  {k:35s}: {v}")
    print("=" * 50)
    return metrics


if __name__ == "__main__":
    config = tyro.cli(PlanConfig)
    main(config)
