"""
Hungarian (bipartite) matching for Phase 3.

Since Qwen may describe textures in either order — e.g.
  "rough brick <SEG_A> to smooth plaster <SEG_B>"
    OR
  "smooth plaster <SEG_A> to rough brick <SEG_B>"
— we need order-invariant loss. Hungarian matching finds the optimal
assignment between the 2 predicted masks and 2 GT masks per sample.

For the 2×2 case this is trivial (compare identity vs swapped cost),
but we implement it cleanly for correctness and potential extension.
"""

import torch
import torch.nn.functional as F


# ===================================================================== #
#  Per-sample matching cost                                               #
# ===================================================================== #

def per_sample_mask_cost(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Compute per-sample matching cost (BCE + Dice) for Hungarian assignment.

    This is NOT the training loss — it's only used to determine which
    assignment is optimal. The actual training loss is computed afterwards
    under the selected assignment.

    Args:
        pred_logits: (B, H, W) raw mask logits
        target: (B, H, W) binary GT

    Returns:
        (B,) cost per sample (lower = better match)
    """
    # Per-sample BCE
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target, reduction="none"
    ).mean(dim=(-1, -2))  # (B,)

    # Per-sample Dice cost
    pred = pred_logits.sigmoid()
    pred_flat = pred.flatten(1)           # (B, H*W)
    target_flat = target.flatten(1)       # (B, H*W)
    intersection = (pred_flat * target_flat).sum(1)
    dice_cost = 1.0 - (2.0 * intersection + smooth) / (
        pred_flat.sum(1) + target_flat.sum(1) + smooth
    )

    return bce + dice_cost


# ===================================================================== #
#  2×2 Hungarian matching                                                 #
# ===================================================================== #

@torch.no_grad()
def hungarian_match_2x2(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    gt_a: torch.Tensor,
    gt_b: torch.Tensor,
    cost_fn=per_sample_mask_cost,
) -> torch.Tensor:
    """
    Find optimal 2×2 assignment between predictions and ground truth.

    Compares two possible assignments per sample:
      Identity: pred_A ↔ gt_A,  pred_B ↔ gt_B
      Swapped:  pred_A ↔ gt_B,  pred_B ↔ gt_A

    Args:
        pred_a: (B, H, W) logits for mask from <SEG_A>
        pred_b: (B, H, W) logits for mask from <SEG_B>
        gt_a: (B, H, W) ground truth mask A
        gt_b: (B, H, W) ground truth mask B
        cost_fn: function(pred, gt) → (B,) per-sample cost

    Returns:
        use_swap: (B,) bool tensor — True where swapped assignment is cheaper
    """
    cost_identity = cost_fn(pred_a, gt_a) + cost_fn(pred_b, gt_b)
    cost_swapped = cost_fn(pred_a, gt_b) + cost_fn(pred_b, gt_a)

    return cost_swapped < cost_identity


def apply_hungarian_match(
    gt_a: torch.Tensor,
    gt_b: torch.Tensor,
    use_swap: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reorder GT masks according to the Hungarian matching result.

    For each sample where use_swap[i] is True, swap gt_a[i] and gt_b[i]
    so that the predictions are paired with their best-matching GT.

    Args:
        gt_a: (B, H, W) ground truth mask A
        gt_b: (B, H, W) ground truth mask B
        use_swap: (B,) bool tensor from hungarian_match_2x2

    Returns:
        matched_gt_a: (B, H, W) GT matched to <SEG_A> prediction
        matched_gt_b: (B, H, W) GT matched to <SEG_B> prediction
    """
    swap_mask = use_swap.view(-1, 1, 1)  # (B, 1, 1) for broadcasting
    matched_gt_a = torch.where(swap_mask, gt_b, gt_a)
    matched_gt_b = torch.where(swap_mask, gt_a, gt_b)
    return matched_gt_a, matched_gt_b
