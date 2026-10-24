from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    save_dir: str | Path = "checkpoints",
    filename: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
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
        "config": config,
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


def _configure_device(device: Optional[str | torch.device] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device


def _configure_dtype(
    device: torch.device | str,
    dtype: Optional[str | torch.dtype] = None,
) -> torch.dtype:
    """Select torch dtype based on device or explicit override.

    Defaults to torch.bfloat16 if using CUDA, and torch.float32 otherwise.
    """
    if dtype is not None:
        if isinstance(dtype, torch.dtype):
            return dtype
        dtype_str = str(dtype).lower().strip()
        if dtype_str in ("bf16", "bfloat16", "torch.bfloat16"):
            return torch.bfloat16
        if dtype_str in ("fp16", "float16", "half", "torch.float16"):
            return torch.float16
        if dtype_str in ("fp32", "float32", "float", "torch.float32"):
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    device_type = device.type if isinstance(device, torch.device) else torch.device(device).type
    if device_type == "cuda":
        return torch.bfloat16
    return torch.float32
