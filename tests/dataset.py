from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dataset.dataset import LeWMDataset, build_dataloaders


@pytest.fixture
def synthetic_h5_file(tmp_path: Path):
    h5_path = tmp_path / "test_data.h5"

    num_episodes = 5
    ep_length = 30
    total_steps = num_episodes * ep_length
    H, W, C = 16, 16, 3
    action_dim = 2

    ep_offset = np.arange(0, total_steps, ep_length, dtype=np.int64)
    ep_len = np.full(num_episodes, ep_length, dtype=np.int64)

    # Random images and actions
    pixels = np.random.randint(0, 256, size=(total_steps, H, W, C), dtype=np.uint8)
    action = np.random.uniform(-1.0, 1.0, size=(total_steps, action_dim)).astype(np.float32)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("pixels", data=pixels)
        f.create_dataset("action", data=action)
        f.create_dataset("ep_offset", data=ep_offset)
        f.create_dataset("ep_len", data=ep_len)

    return str(h5_path)


def test_dataset_item_shape(synthetic_h5_file):
    num_frames = 3
    dataset = LeWMDataset(
        h5_file_path=synthetic_h5_file,
        split="train",
        num_frames=num_frames,
        val_ratio=0.2,
    )

    assert len(dataset) > 0
    item = dataset[0]

    assert "frames" in item
    assert "actions" in item

    # frames: (num_frames, C, H, W) in uint8 [0, 255]
    assert item["frames"].shape == (num_frames, 3, 16, 16)
    assert item["frames"].dtype == torch.uint8

    # actions: (num_frames, action_dim * frameskip)
    assert item["actions"].shape == (num_frames, 10)
    assert item["actions"].dtype == torch.float32


def test_dataset_splits_no_overlap(synthetic_h5_file):
    train_ds = LeWMDataset(synthetic_h5_file, split="train", val_ratio=0.4, seed=42)
    val_ds = LeWMDataset(synthetic_h5_file, split="val", val_ratio=0.4, seed=42)

    assert len(train_ds) > 0
    assert len(val_ds) > 0

    # Ensure train and val starts have no overlap
    train_starts = set(train_ds.valid_starts.tolist())
    val_starts = set(val_ds.valid_starts.tolist())
    assert train_starts.isdisjoint(val_starts)


def test_build_dataloaders(synthetic_h5_file):
    train_loader, val_loader = build_dataloaders(
        data_path=synthetic_h5_file,
        batch_size=2,
        num_frames=4,
        val_ratio=0.2,
        num_workers=0,
    )

    batch = next(iter(train_loader))
    assert batch["frames"].dtype == torch.uint8
    assert batch["frames"].shape == (2, 4, 3, 16, 16)
    assert batch["actions"].shape == (2, 4, 10)


def test_dataset_uint8_default(synthetic_h5_file):
    train_loader, _ = build_dataloaders(
        data_path=synthetic_h5_file,
        batch_size=2,
        num_frames=3,
        val_ratio=0.2,
    )

    batch = next(iter(train_loader))
    assert batch["frames"].dtype == torch.uint8
    assert batch["frames"].shape == (2, 3, 3, 16, 16)
