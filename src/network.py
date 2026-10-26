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
    pass
