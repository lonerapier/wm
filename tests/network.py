import torch

from network import (
    ActionEmbedder,
    ConditionalTransformerBlock,
    CrossAttention,
    CrossAttentionBlock,
    Decoder,
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


def test_cross_attention():
    B, N_q, N_kv, D = 2, 16, 1, 64
    attn = CrossAttention(dim=D, n_heads=4)
    q = torch.randn(B, N_q, D)
    kv = torch.randn(B, N_kv, D)
    out = attn(q, kv)

    assert out.shape == (B, N_q, D)
    assert not torch.isnan(out).any()


def test_cross_attention_block():
    B, N_q, N_kv, D = 2, 16, 1, 64
    block = CrossAttentionBlock(hidden_dim=D, mlp_ratio=4, n_heads=4)
    x = torch.randn(B, N_q, D)
    kv = torch.randn(B, N_kv, D)
    out = block(x, kv)

    assert out.shape == (B, N_q, D)
    assert not torch.isnan(out).any()


def test_decoder_default_config():
    B = 2
    decoder = Decoder(
        image_size=(224, 224),
        patch_size=16,
        in_channels=3,
        cls_dim=192,
        hidden_dim=192,
        depth=2,
        n_heads=3,
    )
    # Check learnable query tokens count: (224/16)^2 = 196
    assert decoder.num_patches == 196
    assert decoder.query_tokens.shape == (1, 196, 192)

    cls_token = torch.randn(B, 192)
    out = decoder(cls_token)

    assert out.shape == (B, 3, 224, 224)
    assert not torch.isnan(out).any()


def test_decoder_3d_input():
    B = 2
    decoder = Decoder(
        image_size=(224, 224),
        patch_size=16,
        in_channels=3,
        cls_dim=192,
        hidden_dim=192,
        depth=2,
        n_heads=3,
    )
    cls_token_3d = torch.randn(B, 1, 192)
    out = decoder(cls_token_3d)

    assert out.shape == (B, 3, 224, 224)
    assert not torch.isnan(out).any()


def test_decoder_backward():
    B = 2
    decoder = Decoder(
        image_size=(64, 64),
        patch_size=16,
        in_channels=3,
        cls_dim=64,
        hidden_dim=64,
        depth=2,
        n_heads=2,
    )
    cls_token = torch.randn(B, 64, requires_grad=True)
    out = decoder(cls_token)
    loss = out.sum()
    loss.backward()

    assert cls_token.grad is not None
    assert decoder.query_tokens.grad is not None
    assert decoder.pred_patch.weight.grad is not None


def test_encoder_to_decoder_end_to_end():
    B, C, H, W = 2, 3, 224, 224
    encoder = Encoder(
        image_size=(H, W),
        patch_size=16,
        in_channels=C,
        hidden_dim=192,
        depth=2,
        n_heads=3,
    )
    decoder = Decoder(
        image_size=(H, W),
        patch_size=16,
        in_channels=C,
        cls_dim=192,
        hidden_dim=192,
        depth=2,
        n_heads=3,
    )
    image = torch.randn(B, C, H, W)
    cls_rep = encoder(image)
    assert cls_rep.shape == (B, 192)

    recon = decoder(cls_rep)
    assert recon.shape == (B, C, H, W)
