from pathlib import Path

import torch
import torch.nn as nn

from utils import (
    load_checkpoint,
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
