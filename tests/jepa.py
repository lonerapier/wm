import pytest
import torch

from jepa import JEPA, SIGReg


@pytest.fixture
def jepa_model():
    return JEPA(
        image_size=(32, 32),
        patch_size=16,
        in_channels=3,
        action_dim=2,
        hidden_dim=64,
        enc_depth=2,
        pred_depth=2,
        n_heads=2,
        mlp_ratio=2,
        num_frames=4,
        pred_mlp_dim=128,
        pred_proj_hidden_dim=128,
        enc_proj_hidden_dim=128,
    )


def test_jepa_encode(jepa_model):
    B, T = 2, 4
    frames = torch.randn(B, T, 3, 32, 32)
    actions = torch.randn(B, T, 2)

    embs, action_embs = jepa_model.encode(frames, actions)

    assert embs.shape == (B, T, 64)
    assert action_embs.shape == (B, T, 64)
    assert not torch.isnan(embs).any()
    assert not torch.isnan(action_embs).any()


def test_jepa_predict(jepa_model):
    B, T, D = 2, 3, 64
    embs = torch.randn(B, T, D)
    action_embs = torch.randn(B, T, D)

    preds = jepa_model.predict(embs, action_embs)

    assert preds.shape == (B, T, D)
    assert not torch.isnan(preds).any()


def test_jepa_forward(jepa_model):
    B, T = 2, 4
    frames = torch.randn(B, T, 3, 32, 32)
    actions = torch.randn(B, T, 2)

    preds, embs = jepa_model(frames, actions)

    assert preds.shape == (B, T - 1, 64)
    assert embs.shape == (B, T, 64)


def test_jepa_predict_rollout(jepa_model):
    B, S, horizon = 2, 5, 4
    D = 64
    init_obs = torch.randn(B, S, 1, D)
    actions = torch.randn(B, S, horizon, 2)

    predicted_latents = jepa_model.predict_rollout(init_obs, actions)

    assert predicted_latents.shape == (B, S, horizon, D)
    assert not torch.isnan(predicted_latents).any()


def test_jepa_training_step(jepa_model):
    optimizer = torch.optim.AdamW(jepa_model.parameters(), lr=1e-4)
    sigreg = SIGReg(device=torch.device("cpu"), num_proj=128, num_knots=17)

    B, T = 2, 4
    frames = torch.randn(B, T, 3, 32, 32)
    actions = torch.randn(B, T, 2)

    preds, embs = jepa_model(frames, actions)
    pred_loss = torch.nn.functional.mse_loss(preds, embs[:, 1:])

    # Temporal embs: (T, B, D)
    temporal_embs = embs.transpose(0, 1)
    sigreg_loss = sigreg(temporal_embs)

    total_loss = pred_loss + 0.1 * sigreg_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    assert not torch.isnan(total_loss)
    assert total_loss.item() > 0.0


def test_jepa_training_step_bf16_autocast(jepa_model):
    optimizer = torch.optim.AdamW(jepa_model.parameters(), lr=1e-4)
    sigreg = SIGReg(device=torch.device("cpu"), num_proj=128, num_knots=17)

    B, T = 2, 4
    frames = torch.randn(B, T, 3, 32, 32)
    actions = torch.randn(B, T, 2)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        preds, embs = jepa_model(frames, actions)
        pred_loss = torch.nn.functional.mse_loss(preds, embs[:, 1:])
        temporal_embs = embs.transpose(0, 1)
        sigreg_loss = sigreg(temporal_embs)
        total_loss = pred_loss + 0.1 * sigreg_loss

    optimizer.zero_grad()
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(jepa_model.parameters(), 1.0)
    optimizer.step()

    assert torch.isfinite(grad_norm)
    assert not torch.isnan(total_loss)
    assert total_loss.item() > 0.0
