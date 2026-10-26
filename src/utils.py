from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_SYNC_MANAGER = None


def get_sync_manager():
    """Lazily initialize and return the Hugging Face sync manager if configured."""
    global _SYNC_MANAGER
    if _SYNC_MANAGER is None:
        try:
            from hf_storage import HFSyncManager, is_hf_configured

            if is_hf_configured():
                _SYNC_MANAGER = HFSyncManager()
        except Exception:
            _SYNC_MANAGER = None
    return _SYNC_MANAGER


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    save_dir: str | Path = "checkpoints",
    filename: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    sync: bool = False,
    async_upload: bool = True,
) -> Path:
    """Save model checkpoint to disk, and optionally sync to cloud/Hugging Face."""
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

    # Remote cloud sync if enabled
    if sync and run_id:
        manager = get_sync_manager()
        if manager and manager.enabled:
            manager.upload_checkpoint(
                path,
                run_id=run_id,
                filename=filename,
                async_upload=async_upload,
            )

    return path



def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device | str = "cpu",
    save_dir: Optional[str | Path] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Load checkpoint from local path, save_dir candidate, or Hugging Face repository."""
    ckpt_path = Path(checkpoint_path)

    if not ckpt_path.is_file():
        # Check in local checkpoint directory
        if save_dir is not None:
            candidate = Path(save_dir) / checkpoint_path
            if candidate.is_file():
                ckpt_path = candidate

        # If still not found, attempt download from Hugging Face
        if not ckpt_path.is_file():
            manager = get_sync_manager()
            if manager and manager.enabled:
                clean_target = str(checkpoint_path)
                if clean_target.startswith("hf://"):
                    clean_target = clean_target.split("/")[-1]
                target_dir = Path(save_dir) if save_dir is not None else Path("checkpoints")
                target_run_id = run_id or "baseline"
                print(f"[checkpoint] Fetching '{clean_target}' from Hugging Face...")
                try:
                    if clean_target in ("latest", "last"):
                        downloaded = manager.get_latest_checkpoint(target_run_id, target_dir)
                    else:
                        downloaded = manager.download_checkpoint(target_run_id, clean_target, target_dir)
                    if downloaded and downloaded.is_file():
                        ckpt_path = downloaded
                except Exception as e:
                    raise FileNotFoundError(f"Checkpoint not found locally or on Hugging Face: {e}") from e

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def save_run_summary(
    run_dir: str | Path,
    summary: Dict[str, Any],
    run_id: Optional[str] = None,
    sync: bool = True,
) -> Path:
    """Save experiment summary.json and optionally sync all artifacts to Hugging Face."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if sync:
        manager = get_sync_manager()
        if manager and manager.enabled:
            target_run_id = run_id or summary.get("run_id", "default")
            print("[summary] Uploading experiment artifacts to Hugging Face...")
            manager.upload_run_dir(run_dir, run_id=target_run_id, async_upload=False)
            manager.wait_for_uploads(timeout=60)
            manager.shutdown(wait=True)
            print("[summary] Hugging Face synchronization complete.")

    return summary_path


def _set_seed(seed: int) -> None:
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
