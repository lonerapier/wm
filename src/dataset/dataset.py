import glob
import os
from dataclasses import dataclass

import h5py
import hdf5plugin  # noqa: F401 - required for blosc filter decompression
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Normalizer:
    """Feature-wise normalizer for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.maximum(std, eps)

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> "Normalizer":
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean


class LeWMDataset(Dataset):
    """Generic dataset for LeWM trajectories stored in HDF5 format.

    Splits data at the episode level to avoid temporal leakage between train and val splits.
    Each sample returns a sequence of `num_frames` consecutive frames and actions.
    """

    def __init__(
        self,
        h5_file_path: str,
        split: str = "train",
        num_frames: int = 4,
        frameskip: int = 5,
        val_ratio: float = 0.1,
        seed: int = 42,
        image_size: int | tuple[int, int] | None = None,
        normalizer: Normalizer | None = None,
        return_uint8: bool = False,
    ):
        self.data: h5py.File | None = None
        self.split = split
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.span = self.num_frames * frameskip
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.normalizer = normalizer
        self.return_uint8 = return_uint8

        self.h5_file_path = self._resolve_path(h5_file_path)

        with h5py.File(self.h5_file_path, "r") as f:
            for required_key in ("pixels", "action", "ep_offset", "ep_len"):
                if required_key not in f:
                    raise KeyError(f"Missing required key '{required_key}' in {self.h5_file_path}")

            self.has_obs = "observation" in f
            self._action_dim = int(f["action"].shape[-1])
            self._obs_dim = int(f["observation"].shape[-1]) if self.has_obs else None

            ep_offsets = f["ep_offset"][:]
            ep_lens = f["ep_len"][:]

        num_episodes = len(ep_offsets)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(num_episodes)
        num_val = int(num_episodes * val_ratio)

        if split == "train":
            selected_eps = perm[num_val:]
        elif split == "val":
            selected_eps = perm[:num_val]
        elif split == "all":
            selected_eps = perm
        else:
            raise ValueError(f"Unknown split '{split}'. Expected 'train', 'val', or 'all'.")

        starts = []
        for ep in selected_eps:
            offset = int(ep_offsets[ep])
            length = int(ep_lens[ep])
            if length >= self.span:
                starts.append(np.arange(offset, offset + length - self.span + 1, dtype=np.int64))
            else:
                raise ValueError(
                    f"Episode {ep} is too short for span {self.span}, length: {length}"
                )

        self.valid_starts = np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)

    @staticmethod
    def _resolve_path(path: str) -> str:
        if os.path.isdir(path):
            candidates = sorted(glob.glob(os.path.join(path, "*.h5")))
            if not candidates:
                raise FileNotFoundError(f"No .h5 files found in directory: {path}")
            return candidates[0]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        return path

    @property
    def action_dim(self) -> int:
        return self._action_dim * self.frameskip

    @property
    def obs_dim(self) -> int | None:
        return self._obs_dim

    def _get_data(self) -> h5py.File:
        if self.data is None:
            self.data = h5py.File(
                self.h5_file_path,
                "r",
                swmr=True,
                rdcc_nbytes=256 * 1024 * 1024,
                rdcc_nslots=100003,
            )
        return self.data

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = self._get_data()

        start = self.valid_starts[idx]
        end = start + self.span

        # Contiguous HDF5 read: reading [start:end] in one hyperslab is much faster than strided slicing
        raw_slice = data["pixels"][start:end]
        raw_pixels = np.ascontiguousarray(raw_slice[:: self.frameskip])

        # TODO: remove `return_uint6` after testing
        if getattr(self, "return_uint8", False):
            # Return uint8 (T, C, H, W) for fast IPC transfer; normalization and resize done on GPU
            pixels = torch.from_numpy(raw_pixels).permute(0, 3, 1, 2)
        else:
            # Legacy/default: float in [0, 1] resized on CPU
            pixels = torch.from_numpy(raw_pixels).float().permute(0, 3, 1, 2) / 255.0
            if self.image_size is not None and pixels.shape[-2:] != self.image_size:
                pixels = F.interpolate(
                    pixels, size=self.image_size, mode="bilinear", align_corners=False
                )

        # Actions: (num_frames, action_dim)
        raw_actions = data["action"][start:end]
        if self.normalizer is not None:
            raw_actions = self.normalizer.normalize_action(raw_actions)
        raw_actions = raw_actions.reshape(self.num_frames, -1)
        actions = torch.from_numpy(raw_actions).float().nan_to_num(0.0)

        item = {
            "frames": pixels,
            "actions": actions,
        }

        return item

    def __del__(self):
        if getattr(self, "data", None) is not None:
            try:
                self.data.close()
            except Exception:
                pass


def build_dataloaders(
    data_path: str,
    batch_size: int,
    num_frames: int = 4,
    frameskip: int = 5,
    val_ratio: float = 0.1,
    image_size: int | tuple[int, int] | None = None,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = True,
    normalizer: Normalizer | None = None,
    return_uint8: bool = False,
    prefetch_factor: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders with episode-level splits."""
    train_dataset = LeWMDataset(
        h5_file_path=data_path,
        split="train",
        num_frames=num_frames,
        frameskip=frameskip,
        val_ratio=val_ratio,
        seed=seed,
        image_size=image_size,
        normalizer=normalizer,
        return_uint8=return_uint8,
    )
    val_dataset = LeWMDataset(
        h5_file_path=data_path,
        split="val",
        num_frames=num_frames,
        frameskip=frameskip,
        val_ratio=val_ratio,
        seed=seed,
        image_size=image_size,
        normalizer=normalizer,
        return_uint8=return_uint8,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return train_loader, val_loader
