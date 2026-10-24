from pathlib import Path

import torch
import torch.nn as nn

from utils import (
    _configure_device,
    _configure_dtype,
    _set_seed,
    load_checkpoint,
    save_checkpoint,
)


def test_configure_device():
    device = _configure_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "mps", "cpu")

    # Explicit override
    cpu_device = _configure_device("cpu")
    assert cpu_device.type == "cpu"


def test_configure_dtype():
    # CUDA defaults to bf16
    cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert _configure_dtype("cuda") == torch.bfloat16
    assert _configure_dtype(
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu").type
    ) in (
        torch.bfloat16,
        torch.float32,
    )

    # CPU and MPS default to float32
    assert _configure_dtype("cpu") == torch.float32
    assert _configure_dtype(torch.device("cpu")) == torch.float32
    assert _configure_dtype("mps") == torch.float32

    # Explicit overrides
    assert _configure_dtype("cuda", "fp32") == torch.float32
    assert _configure_dtype("cpu", "bf16") == torch.bfloat16
    assert _configure_dtype("cpu", "float16") == torch.float16
    assert _configure_dtype("cpu", torch.bfloat16) == torch.bfloat16


def test_set_seed():
    _set_seed(123)
    a = torch.randn(5)
    _set_seed(123)
    b = torch.randn(5)
    torch.testing.assert_close(a, b)


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
