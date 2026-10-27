from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
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
    data_path: str = "data/tworooms"
    """Path to dataset directory / h5 files."""
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 42
    device: Optional[str] = None
    dtype: Optional[str] = None

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
    init_mean: Optional[torch.Tensor] = None,
    init_std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute Cross-Entropy Method (CEM) optimization for a batch of environments.

    Args:
        model: Trained world model (JEPA).
        init_obs: (B, H_ctx, D) initial context frames.
        goal_latent: (B, D) target goal embedding.
        horizon: Planning horizon steps T.
        candidates: Number of action candidates S sampled per iteration.
        elite_ratio: Ratio of elite samples retained.
        iterations: Number of CEM iterations.
        smoothing: Smoothing factor for distribution updates.
        action_dim: Dimension of each action.
        action_min: Lower bound for action clipping.
        action_max: Upper bound for action clipping.
        init_mean: Optional warm-start mean distribution (B, H_ctx, A).
        init_std: Optional warm-start standard deviation (B, H_ctx, A).

    Returns:
        best_actions: (B, T, A) optimal action sequence for each batch item.
        best_cost: (B,) final cost achieved by the best candidate.
    """
    assert init_obs.ndim == 4, f"init_obs.ndim: {init_obs.ndim}"
    assert goal_obs.ndim == 2, f"goal_obs.ndim: {goal_obs.ndim}"

    device = init_obs.device
    B = init_obs.shape[0]
    # min is 2 because `.std()` computes bessel's correction (N-1) in the denominator
    num_elites = max(2, int(candidates * elite_ratio))

    mean, std = init_action_dist(B, horizon, action_dim, device)

    best_actions = mean.clone()
    best_cost = torch.full((B,), float("inf"), device=device)

    # Expand context observation across candidate dimension S: (B, S, H_ctx, D)
    init_obs_expanded = init_obs.unsqueeze(1).expand(-1, candidates, *init_obs.shape[1:])

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
            pred_latents = model.predict_rollout(init_obs_expanded, actions)

        # compute cost for each candidate: (B, S)
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
    model: torch.nn.Module,
    valloader: DataLoader,
    iterations: int = 10,
    candidates: int = 300,
    elite_ratio: float = 0.1,
    horizon: int = 5,
    smoothing: float = 0.1,
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
        device: Device to run evaluation on.

    Returns:
        Dictionary of planning metrics.
    """
    model.eval()

    total_cost = 0.0
    num_samples = 0

    # TODO: run on specifc obs indices

    for batch in valloader:
        image_size = getattr(model.encoder, "image_size", None)
        frames, actions = prepare_batch(batch, device=device, image_size=image_size)

        B, T = frames.shape[:2]
        print(frames.shape, actions.shape)

        # Target goal embedding
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and dtype == torch.bfloat16),
        ):
            goal_latent, _ = model.encode(frames, actions)  # (B, T, D)
            # Context observation (e.g. frame 0) and goal observation (e.g. terminal frame)
            # TODO: observation can be a sequence of frames upto the receeding horizon
            # TODO: look at history length for specific environment
            init_obs_latent = goal_latent[:, 0]  # (B, D)
            goal_latent = goal_latent[:, -1]  # (B, D)

        best_actions, best_costs = cem_optimize(
            model=model,
            init_obs=init_obs_latent,
            goal_obs=goal_latent,
            horizon=horizon,
            candidates=candidates,
            elite_ratio=elite_ratio,
            iterations=iterations,
            smoothing=smoothing,
            action_dim=actions.shape[-1],
        )

        total_cost += best_costs.sum().item()
        num_samples += B

    avg_cost = total_cost / max(1, num_samples)
    return {"plan/terminal_cost": avg_cost}


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
    model.load_state_dict(checkpoint["model"])
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
        saved_cfg = {}

    data_path = config.data_path or saved_cfg.get("data_path", "data/tworooms")

    _, valloader = build_dataloaders(
        data_path=data_path,
        batch_size=config.batch_size,
        num_frames=config.horizon + 1,
        frameskip=config.frameskip,
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
        smoothing=config.smoothing,
        device=device,
        dtype=dtype,
    )
    print(f"[plan] Evaluation completed. Metrics: {metrics}")


if __name__ == "__main__":
    config = tyro.cli(PlanConfig)
    main(config)
