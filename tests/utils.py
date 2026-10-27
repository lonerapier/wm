from pathlib import Path

import torch
import torch.nn as nn

from utils import (
    load_checkpoint,
    prepare_batch,
    save_checkpoint,
)


def test_checkpoint_save_and_load(tmp_path: Path):
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Perform a dummy step
    loss = model(torch.randn(3, 10)).sum()
    loss.backward()
    optimizer.step()

    config = {"batch_size": 32, "lr": 0.01}
    ckpt_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=2,
        step=50,
        save_dir=tmp_path,
        filename="test_ckpt.pt",
        config=config,
    )

    assert ckpt_path.is_file()

    # Create new fresh model and optimizer
    model_new = nn.Linear(10, 2)
    optimizer_new = torch.optim.Adam(model_new.parameters(), lr=0.01)

    loaded_ckpt = load_checkpoint(
        checkpoint_path=ckpt_path,
        model=model_new,
        optimizer=optimizer_new,
        device=torch.device("cpu"),
    )

    assert loaded_ckpt["epoch"] == 2
    assert loaded_ckpt["step"] == 50
    assert loaded_ckpt["config"] == config

    # Verify weights are identical
    for p1, p2 in zip(model.parameters(), model_new.parameters()):
        torch.testing.assert_close(p1, p2)


def test_prepare_batch_uint8():
    device = torch.device("cpu")
    B, T, C, H, W = 2, 3, 3, 32, 32
    raw_frames = torch.randint(0, 256, (B, T, C, H, W), dtype=torch.uint8)
    raw_actions = torch.randn(B, T, 2)
    batch = {"frames": raw_frames, "actions": raw_actions}

    frames, actions = prepare_batch(batch, device=device, image_size=(64, 64))

    assert frames.dtype == torch.float32
    assert frames.shape == (B, T, C, 64, 64)
    assert frames.min() >= 0.0 and frames.max() <= 1.0
    assert actions.shape == (B, T, 2)


def test_validate_with_uint8_batch():
    from types import SimpleNamespace

    from eval import validate
    from jepa import JEPA, SIGReg

    device = torch.device("cpu")
    config = SimpleNamespace(
        image_size=16,
        patch_size=8,
        in_channels=3,
        action_dim=2,
        hidden_dim=32,
        enc_depth=1,
        pred_depth=1,
        n_heads=2,
        mlp_ratio=2,
        num_frames=3,
        sigreg_weight=0.1,
    )

    model = JEPA(
        image_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        enc_depth=config.enc_depth,
        pred_depth=config.pred_depth,
        n_heads=config.n_heads,
        mlp_ratio=config.mlp_ratio,
        num_frames=config.num_frames,
    )
    sigreg = SIGReg(device=device, num_proj=16, num_knots=5)

    # Batch with uint8 frames as returned by dataset by default
    B, T, C, H, W = 2, config.num_frames, 3, config.image_size, config.image_size
    dataloader = [
        {
            "frames": torch.randint(0, 256, (B, T, C, H, W), dtype=torch.uint8),
            "actions": torch.randn(B, T, config.action_dim),
        }
    ]

    metrics = validate(
        model=model,
        dataloader=dataloader,
        sigreg=sigreg,
        config=config,
        device=device,
        dtype=torch.float32,
    )

    assert "val_loss/total" in metrics
    assert "val_loss/prediction_loss" in metrics
    assert not torch.isnan(torch.tensor(metrics["val_loss/total"]))
