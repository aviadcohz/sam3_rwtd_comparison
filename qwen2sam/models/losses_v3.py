"""
Loss functions for Qwen2SAM v3.

Reuses v2 segmentation losses directly. The only difference is:
  - Alignment operates on pooled description embeddings (already handled in model)
  - LM loss weight is higher (descriptions are the primary driver)
  - No SEG position validity check needed (extraction handles it in model)
"""

import torch

from qwen2sam.models.losses_v2 import v2_seg_loss
from qwen2sam.models.losses import alignment_loss, alignment_loss_with_bank


def v3_total_loss(
    lm_loss: torch.Tensor,
    seg_loss: torch.Tensor,
    alignment_loss_val: torch.Tensor = None,
    lm_weight: float = 0.5,
    seg_weight: float = 1.0,
    alignment_weight: float = 1.0,
) -> torch.Tensor:
    """Weighted combination of all v3 losses."""
    total = seg_weight * seg_loss
    if lm_loss is not None and lm_weight > 0:
        total = total + lm_weight * lm_loss
    if alignment_loss_val is not None and alignment_weight > 0:
        total = total + alignment_weight * alignment_loss_val
    return total
