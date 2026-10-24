import torch

from network import (
    ActionEmbedder,
    ConditionalTransformerBlock,
    Encoder,
    Predictor,
    TransformerBlock,
)


def test_transformer_block():
    B, T, D = 2, 8, 64
    block = TransformerBlock(hidden_dim=D, mlp_ratio=4, n_heads=4)
    x = torch.randn(B, T, D)
    out = block(x)

    assert out.shape == (B, T, D)
    assert not torch.isnan(out).any()


def test_conditional_transformer_block_identity_at_init():
    """At zero-initialization of modulation weights, the block must act as an identity."""
    B, T, D = 2, 5, 64
    block = ConditionalTransformerBlock(hidden_dim=D, mlp_dim=128, n_heads=4)
    x = torch.randn(B, T, D)
    action_emb = torch.randn(B, T, D)
    out = block(x, action_emb)

    assert out.shape == (B, T, D)
    torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-5)


def test_action_embedder():
    B, T, A_dim, D = 4, 6, 2, 64
    embedder = ActionEmbedder(input_dim=A_dim, smoothed_dim=D, emb_dim=D, mlp_ratio=4)
    actions = torch.randn(B, T, A_dim)
    emb = embedder(actions)

    assert emb.shape == (B, T, D)
    assert not torch.isnan(emb).any()


def test_encoder():
    B, C, H, W = 2, 3, 32, 32
    patch_size = 16
    hidden_dim = 64
    encoder = Encoder(
        image_size=(H, W),
        patch_size=patch_size,
        in_channels=C,
        hidden_dim=hidden_dim,
        depth=2,
        n_heads=2,
        mlp_ratio=2,
    )
    images = torch.randn(B, C, H, W)
    out = encoder(images)

    assert out.shape == (B, hidden_dim)
    assert not torch.isnan(out).any()


def test_predictor():
    B, T, D = 2, 4, 64
    predictor = Predictor(
        num_frames=T,
        hidden_dim=D,
        depth=2,
        n_heads=2,
        mlp_dim=128,
        dropout=0.0,
        emb_dropout=0.0,
    )
    x_emb = torch.randn(B, T, D)
    action_emb = torch.randn(B, T, D)
    preds = predictor(x_emb, action_emb)

    assert preds.shape == (B, T, D)
    assert not torch.isnan(preds).any()
