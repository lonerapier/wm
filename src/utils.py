from pathlib import Path
from typing import Any, Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    save_dir: str | Path = "checkpoints",
    filename: Optional[str] = None,
    config: Any = None,
) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    path = save_dir / filename

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "step": step,
    }
    torch.save(state, path)
    return path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device | str = "cpu",
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def _set_seed(seed):
    torch.manual_seed(seed)


def _configure_device() -> torch.device:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device
