import torch

from jepa import SIGReg


def test_sigreg_buffers_and_shapes():
    device = torch.device("cpu")
    num_knots = 17
    num_proj = 512
    sigreg = SIGReg(device=device, num_proj=num_proj, num_knots=num_knots, lam=1.0)

    assert sigreg.t.shape == (num_knots,)
    assert sigreg.weights.shape == (num_knots,)
    assert sigreg.phi.shape == (num_knots,)
    assert sigreg.t[0] == 0.0
    assert sigreg.t[-1] == 3.0


def test_sigreg_forward_3d():
    device = torch.device("cpu")
    sigreg = SIGReg(device=device, num_proj=256, num_knots=17)

    T, B, D = 4, 32, 64
    emb = torch.randn(T, B, D, device=device)
    loss = sigreg(emb)

    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_sigreg_forward_2d():
    device = torch.device("cpu")
    sigreg = SIGReg(device=device, num_proj=256, num_knots=17)

    B, D = 64, 128
    emb = torch.randn(B, D, device=device)
    loss = sigreg(emb)

    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_sigreg_anti_collapse_property():
    """Verify that collapsed representations yield higher loss than isotropic normal representations."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    sigreg = SIGReg(device=device, num_proj=1024, num_knots=17)

    T, B, D = 4, 128, 64

    # Isotropic standard Gaussian embeddings
    isotropic_emb = torch.randn(T, B, D, device=device)

    # Completely collapsed embeddings (all samples identical)
    collapsed_emb = torch.ones(T, B, D, device=device) * 2.5

    iso_loss = sigreg(isotropic_emb).item()
    col_loss = sigreg(collapsed_emb).item()

    assert col_loss > iso_loss * 2.0, (
        f"Collapsed loss ({col_loss}) should be much higher than isotropic loss ({iso_loss})"
    )


def test_sigreg_gradient_flow():
    device = torch.device("cpu")
    sigreg = SIGReg(device=device, num_proj=128, num_knots=17)

    emb = torch.randn(4, 16, 32, device=device, requires_grad=True)
    loss = sigreg(emb)
    loss.backward()

    assert emb.grad is not None
    assert emb.grad.shape == emb.shape
    assert not torch.isnan(emb.grad).any()
    assert emb.grad.abs().sum() > 0
