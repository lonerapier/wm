import torch
from einops import rearrange
from torch import nn

from network import ActionEmbedder, Encoder, Predictor


class SIGReg(nn.Module):
    def __init__(
        self,
        device: torch.device,
        num_proj: int = 1024,
        num_knots: int = 17,
        lam: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 3.0,
    ):
        super().__init__()
        self.knots = num_knots
        self.num_proj = num_proj
        t = torch.linspace(t_min, t_max, num_knots, device=device)
        dt = t[1] - t[0] / (num_knots - 1)
        trapezoid_weights = torch.full((num_knots,), 2 * dt, device=device)
        trapezoid_weights[[0, -1]] = dt
        phi = torch.exp(-(t**2) / (2 * lam**2))
        self.register_buffer("t", t)
        self.register_buffer("weights", phi * trapezoid_weights)
        self.register_buffer("phi", phi)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb (T, B, D)
        T, B, D = emb.shape

        random_proj = torch.nn.functional.normalize(
            torch.randn(self.num_proj, D, dtype=emb.dtype, device=emb.device), dim=-1
        )  # (num_proj, D)
        H = emb @ random_proj.T  # (T, B, num_proj)

        angles = H.unsqueeze(-1) * self.t  # (T, B, num_proj, num_knots)
        C = torch.cos(angles).mean(dim=1)  # (T, num_proj, num_knots)
        S = torch.sin(angles).mean(dim=1)  # (T, num_proj, num_knots)
        err = (C - self.phi) ** 2 + S**2
        statistic = (err @ self.weights) * B  # (T, num_proj)
        return statistic.mean()  # mean over time and projections


class JEPA(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        in_channels: int = 3,
        action_dim: int = 2,
        hidden_dim: int = 192,
        enc_depth: int = 12,
        pred_depth: int = 6,
        n_heads: int = 3,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        enc_proj_hidden_dim: int = 2048,
        num_frames: int = 4,
        pred_mlp_dim: int = 2048,
        pred_proj_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.encoder = Encoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            depth=enc_depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            mlp_dropout=mlp_dropout,
        )
        # TODO: check the dropout settings for predictor
        self.predictor = Predictor(
            num_frames=num_frames - 1,
            hidden_dim=hidden_dim,
            depth=pred_depth,
            n_heads=n_heads,
            dropout=dropout,
            mlp_dropout=mlp_dropout,
            mlp_dim=pred_mlp_dim,
        )
        self.action_embedder = ActionEmbedder(
            input_dim=action_dim,
            smoothed_dim=hidden_dim,
            emb_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
        )
        self.enc_proj = (
            nn.Sequential(
                nn.Linear(hidden_dim, enc_proj_hidden_dim),
                nn.BatchNorm1d(enc_proj_hidden_dim),
                nn.GELU(),
                nn.Linear(enc_proj_hidden_dim, hidden_dim),
            )
            if enc_proj_hidden_dim != 0
            else nn.Identity()
        )
        self.pred_proj = (
            nn.Sequential(
                nn.Linear(hidden_dim, pred_proj_hidden_dim),
                nn.BatchNorm1d(pred_proj_hidden_dim),
                nn.GELU(),
                nn.Linear(pred_proj_hidden_dim, hidden_dim),
            )
            if pred_proj_hidden_dim != 0
            else nn.Identity()
        )

    def forward(self, frames: torch.Tensor, actions: torch.Tensor):
        # frames (B, T, C, H, W)
        # actions (B, T, num_actions)

        embs = self.encode(frames)  # (B, T, D)
        actions_emb = self.action_embedder(actions)  # (B, T, D)

        pred = self.predict(embs[:, :-1], actions_emb[:, :-1])  # (B, T-1, D)

        return pred, embs

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        # frames (B, T, C, H, W)
        # actions (B, T, num_actions)
        B, T = frames.shape[:2]
        frames = rearrange(frames, "b t ... -> (b t) ...")
        emb = self.encoder(frames)
        emb = self.enc_proj(emb)
        emb = rearrange(emb, "(b t) d -> b t d", b=B, t=T)

        return emb

    def predict(self, emb: torch.Tensor, actions_emb: torch.Tensor) -> torch.Tensor:
        # emb (B, T, D)
        # actions_emb (B, T, D)
        B, T = emb.shape[:2]
        preds = self.predictor(emb, actions_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=B, t=T)

    def rollout(self, init_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # init_obs (B, S, H, C, h, w)
        # actions (B, S, T, action_dim)
        B, S, H = init_obs.shape[:3]
        T = actions.shape[2]

        raise NotImplementedError
