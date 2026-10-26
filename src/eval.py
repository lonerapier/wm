from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    sigreg: nn.Module,
    config: Any,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    """Run validation over a dataloader and compute loss and representation metrics.

    Metrics computed:
    - val_loss/total: Total weighted loss (prediction_loss + sigreg_weight * sigreg_loss)
    - val_loss/prediction_loss: Mean squared error on next latent states
    - val_loss/sigreg_loss: SIGReg regularizer loss
    - val_loss/l1_error: Mean absolute error on next latent states
    - val/cosine_similarity: Directional alignment between predicted and target embeddings
    - val/emb_norm: Average L2 norm of target representations
    - val/pred_norm: Average L2 norm of predicted representations
    - val/emb_std: Feature variance across batch and time (collapse monitor)
    """
    model.eval()

    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    total_pred_loss = 0.0
    total_sigreg_loss = 0.0
    total_loss = 0.0
    total_l1_error = 0.0
    total_cos_sim = 0.0
    total_emb_norm = 0.0
    total_pred_norm = 0.0
    total_emb_std = 0.0

    num_batches = 0
    target_size = getattr(config, "image_size", None)
    target_size = (target_size, target_size) if isinstance(target_size, int) else target_size

    for batch in dataloader:
        frames = batch["frames"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=(device.type == "cuda" and dtype == torch.bfloat16),
        ):
            preds, embs = model(frames, actions)
            targets = embs[:, 1:]

            pred_loss = nn.functional.mse_loss(preds, targets)
            temporal_embs = embs.transpose(0, 1)
            sr_loss = sigreg(temporal_embs)
            batch_total_loss = pred_loss + config.sigreg_weight * sr_loss

            l1_err = nn.functional.l1_loss(preds, targets)
            cos_sim = nn.functional.cosine_similarity(preds, targets, dim=-1).mean()

            emb_norm = torch.linalg.vector_norm(embs, dim=-1).mean()
            pred_norm = torch.linalg.vector_norm(preds, dim=-1).mean()
            # Variance across batch and sequence dimensions to detect representation collapse
            emb_std = embs.std(dim=0).mean()

        total_loss += batch_total_loss.item()
        total_pred_loss += pred_loss.item()
        total_sigreg_loss += sr_loss.item()
        total_l1_error += l1_err.item()
        total_cos_sim += cos_sim.item()
        total_emb_norm += emb_norm.item()
        total_pred_norm += pred_norm.item()
        total_emb_std += emb_std.item()

        num_batches += 1

    return {
        "val_loss/total": total_loss / num_batches,
        "val_loss/prediction_loss": total_pred_loss / num_batches,
        "val_loss/sigreg_loss": total_sigreg_loss / num_batches,
        "val_loss/l1_error": total_l1_error / num_batches,
        "val/cosine_similarity": total_cos_sim / num_batches,
        "val/emb_norm": total_emb_norm / num_batches,
        "val/pred_norm": total_pred_norm / num_batches,
        "val/emb_std": total_emb_std / num_batches,
    }
