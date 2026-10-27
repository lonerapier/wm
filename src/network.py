import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# TODO: try modern transformer stack, rope, qknorm, swiglu


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, is_causal: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_ratio: int,
        n_heads: int,
        dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = Attention(hidden_dim, n_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output = self.attn(x_norm, is_causal=False)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class ConditionalTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = Attention(hidden_dim, n_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(mlp_dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6, bias=True),
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0.0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0.0)

    def forward(self, x, action_emb):
        # x shape (B, T, D)
        # action_emb shape (B, T, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            action_emb
        ).chunk(6, dim=-1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output = self.attn(x_norm, is_causal=True)
        x = x + gate_msa * attn_output
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # x (queries): (B, N_q, dim)
        # kv (keys/values): (B, N_kv, dim)
        B, N_q, C = x.shape
        N_kv = kv.shape[1]

        q = self.q(x).reshape(B, N_q, self.n_heads, self.head_dim).transpose(1, 2)
        kv_proj = (
            self.kv(kv).reshape(B, N_kv, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )
        k, v = kv_proj.unbind(0)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N_q, C)
        return self.proj(out)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_ratio: int = 4,
        n_heads: int = 3,
        dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.attn = CrossAttention(hidden_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm_kv(kv))
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        patch_size: int = 14,
        in_channels: int = 3,
        hidden_dim: int = 192,
        depth: int = 12,
        n_heads: int = 3,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim,
                mlp_ratio,
                n_heads,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        # x shape (batch_size, 3, H, W)
        x = self.patch_embed(x)
        # x shape (batch_size, hidden_dim, H//p, W//p)
        x = rearrange(x, "b c h w -> b (h w) c")
        # x shape (batch_size, num_patches + 1, hidden_dim)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.pos_embed
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0, :]  # return projected CLS token


class Predictor(nn.Module):
    def __init__(
        self,
        num_frames: int = 4,
        hidden_dim: int = 192,
        depth: int = 6,
        n_heads: int = 3,
        mlp_dim: int = 2048,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_frames, hidden_dim))
        self.transformer_blocks = nn.ModuleList([
            ConditionalTransformerBlock(
                hidden_dim,
                mlp_dim,
                n_heads,
                dropout,
            )
            for _ in range(depth)
        ])
        self.dropout = nn.Dropout(emb_dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_emb: torch.Tensor, action_emb: torch.Tensor):
        # x shape (batch_size, num_frames, hidden_dim)
        # action shape (batch_size, num_frames, action_dim)
        B, T = x_emb.shape[:2]
        x = self.dropout(x_emb + self.pos_embed[:, :T])
        for block in self.transformer_blocks:
            x = block(x, action_emb)
        return self.norm(x)


class ActionEmbedder(nn.Module):
    def __init__(
        self, input_dim: int = 10, smoothed_dim: int = 10, emb_dim: int = 10, mlp_ratio: int = 4
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.action_emb = nn.Sequential(
            nn.Linear(smoothed_dim, emb_dim * mlp_ratio),
            nn.SiLU(),
            nn.Linear(emb_dim * mlp_ratio, emb_dim),
        )

    def forward(self, action: torch.Tensor):
        # action shape (batch_size, num_frames, action_dim)
        action = action.permute(0, 2, 1)
        action = self.patch_embed(action)
        action = action.permute(0, 2, 1)
        action = self.action_emb(action)
        return action


class Decoder(nn.Module):
    """Lightweight transformer decoder for visualising encoder state ([CLS] representation).

    Decodes the [CLS] token embedding (192 dim by default) from the last encoder layer
    into an RGB image (224x224 by default).
    The [CLS] representation is first projected to a hidden dimension and used as key
    and value in cross-attention. A fixed set of learnable query tokens (one per image patch,
    P = (224/16)^2 = 196) interacts with this global representation through cross-attention
    layers with residual MLP blocks. Resulting patch embeddings are linearly projected to
    pixel patches and rearranged to produce the RGB image.
    """

    def __init__(
        self,
        image_size: tuple[int, int] | int = (224, 224),
        patch_size: int = 16,
        in_channels: int = 3,
        cls_dim: int = 192,
        hidden_dim: int = 192,
        depth: int = 4,
        n_heads: int = 3,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.cls_dim = cls_dim
        self.hidden_dim = hidden_dim

        assert image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0, (
            f"image_size {image_size} must be divisible by patch_size {patch_size}"
        )

        grid_h = image_size[0] // patch_size
        grid_w = image_size[1] // patch_size
        self.grid_size = (grid_h, grid_w)
        self.num_patches = grid_h * grid_w

        # Project [CLS] token embedding to hidden dimension for key and value in cross-attention
        self.cls_proj = nn.Linear(cls_dim, hidden_dim)

        # Fixed set of learnable query tokens, one for each patch of the target image
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim) * 0.02)

        # Cross-attention layers with residual MLP blocks
        self.transformer_blocks = nn.ModuleList([
            CrossAttentionBlock(
                hidden_dim=hidden_dim,
                mlp_ratio=mlp_ratio,
                n_heads=n_heads,
                dropout=dropout,
                mlp_dropout=mlp_dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Linearly project patch embeddings to patch_size * patch_size * in_channels
        self.pred_patch = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        # cls_token shape: (B, cls_dim) or (B, 1, cls_dim)
        assert cls_token.ndim in (2, 3)
        assert cls_token.shape[-1] == self.cls_dim
        if cls_token.ndim == 2:
            cls_token = cls_token.unsqueeze(1)

        B = cls_token.shape[0]

        # [CLS] representation projected to hidden dimension, used as shared key and value
        kv = self.cls_proj(cls_token)  # (B, 1, hidden_dim)

        # Expand learnable query tokens for batch
        x = self.query_tokens.expand(B, -1, -1)  # (B, num_patches, hidden_dim)

        for block in self.transformer_blocks:
            x = block(x, kv)
        x = self.norm(x)

        # Linearly project to patch pixels
        patches = self.pred_patch(x)  # (B, num_patches, patch_size * patch_size * in_channels)

        # Rearrange patches to produce (B, C, H, W) RGB image
        grid_h, grid_w = self.grid_size
        out = rearrange(
            patches,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=grid_h,
            w=grid_w,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.in_channels,
        )
        return out
