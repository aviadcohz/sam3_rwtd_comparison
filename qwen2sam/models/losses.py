"""
Loss functions for Qwen2SAM.

Phase 1: BCE + Dice + IoU prediction loss
Phase 2: + Soft-clDice + Interface clDice + Mutual Exclusivity
Phase 3: + Hungarian matching (added later)
"""

import torch
import torch.nn.functional as F

from qwen2sam.utils.soft_skeleton import (
    soft_cldice_loss,
    interface_cldice_loss,
    exclusivity_loss,
)


def sigmoid_bce_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross-entropy loss on raw logits.

    Args:
        pred_logits: (B, H, W) raw mask logits from SAM decoder
        target: (B, H, W) binary ground truth mask {0, 1}
    """
    return F.binary_cross_entropy_with_logits(pred_logits, target, reduction="mean")


def dice_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft Dice loss (1 - Dice coefficient).

    Operates on sigmoid(logits) vs binary target.

    Args:
        pred_logits: (B, H, W) raw mask logits
        target: (B, H, W) binary ground truth {0, 1}
        smooth: Laplace smoothing to avoid division by zero
    """
    pred = pred_logits.sigmoid()
    pred_flat = pred.flatten(1)       # (B, H*W)
    target_flat = target.flatten(1)   # (B, H*W)

    intersection = (pred_flat * target_flat).sum(1)
    cardinality = pred_flat.sum(1) + target_flat.sum(1)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return (1.0 - dice).mean()


def compute_mask_iou(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute IoU between binarized prediction and ground truth.

    Args:
        pred_logits: (B, H, W) raw logits
        target: (B, H, W) binary GT
        threshold: binarization threshold on sigmoid(logits)

    Returns:
        (B,) IoU per sample
    """
    pred = (pred_logits.sigmoid() > threshold).float()
    intersection = (pred * target).sum(dim=(-1, -2))
    union = pred.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) - intersection
    return intersection / (union + 1e-6)


def iou_prediction_loss(
    iou_pred: torch.Tensor,
    pred_logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss between SAM's predicted IoU and the actual IoU.

    This teaches SAM to estimate its own mask quality.

    Args:
        iou_pred: (B,) SAM's predicted IoU scores
        pred_logits: (B, H, W) raw mask logits
        target: (B, H, W) binary GT
    """
    with torch.no_grad():
        actual_iou = compute_mask_iou(pred_logits, target)
    return F.mse_loss(iou_pred, actual_iou)


def focal_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """
    Focal Loss for hard-pixel mining in binary segmentation.

    Down-weights easy pixels and focuses learning on hard pixels
    near texture transitions where the model is uncertain.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        pred_logits: (B, H, W) raw mask logits (pre-sigmoid)
        target: (B, H, W) binary ground truth {0, 1}
        gamma: focusing parameter (higher = more focus on hard pixels)
        alpha: class balance weight for positive class
    """
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target, reduction="none"
    )
    p = pred_logits.sigmoid()
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def phase1_loss(
    pred_logits: torch.Tensor,
    iou_pred: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    iou_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Combined Phase 1 loss: BCE + Dice + IoU prediction.

    Args:
        pred_logits: (B, H, W) raw mask logits
        iou_pred: (B,) predicted IoU from SAM
        target: (B, H, W) binary GT mask
        bce_weight: weight for BCE loss
        dice_weight: weight for Dice loss
        iou_weight: weight for IoU prediction loss

    Returns:
        total_loss: scalar
        metrics: dict with individual loss values for logging
    """
    bce = sigmoid_bce_loss(pred_logits, target)
    dice = dice_loss(pred_logits, target)
    iou = iou_prediction_loss(iou_pred, pred_logits, target)

    total = bce_weight * bce + dice_weight * dice + iou_weight * iou

    metrics = {
        "bce": bce.item(),
        "dice": dice.item(),
        "iou_pred": iou.item(),
        "total": total.item(),
    }
    return total, metrics


def phase1_paired_loss(
    pred_a: torch.Tensor,
    iou_a: torch.Tensor,
    gt_a: torch.Tensor,
    pred_b: torch.Tensor,
    iou_b: torch.Tensor,
    gt_b: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    iou_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Combined loss for both masks in a texture transition pair.

    Returns average loss across Mask A and Mask B.
    """
    loss_a, metrics_a = phase1_loss(pred_a, iou_a, gt_a, bce_weight, dice_weight, iou_weight)
    loss_b, metrics_b = phase1_loss(pred_b, iou_b, gt_b, bce_weight, dice_weight, iou_weight)

    total = (loss_a + loss_b) / 2.0
    metrics = {
        "loss_a": metrics_a["total"],
        "loss_b": metrics_b["total"],
        "bce": (metrics_a["bce"] + metrics_b["bce"]) / 2.0,
        "dice": (metrics_a["dice"] + metrics_b["dice"]) / 2.0,
        "iou_pred": (metrics_a["iou_pred"] + metrics_b["iou_pred"]) / 2.0,
        "total": total.item(),
    }
    return total, metrics


# ===================================================================== #
#  Phase 2: Topological Boundary Losses                                   #
# ===================================================================== #

def phase2_paired_loss(
    pred_a: torch.Tensor,
    iou_a: torch.Tensor,
    gt_a: torch.Tensor,
    pred_b: torch.Tensor,
    iou_b: torch.Tensor,
    gt_b: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    iou_weight: float = 1.0,
    cldice_weight: float = 0.5,
    interface_cldice_weight: float = 1.0,
    exclusivity_weight: float = 0.5,
    skeleton_iters: int = 15,
    interface_dilation_iters: int = 2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Phase 2 combined loss: Phase 1 losses + topological regularizers.

    Components:
      Phase 1 (retained):
        - BCE + Dice + IoU prediction (per mask, averaged)
      Phase 2 (new):
        - Per-mask soft-clDice: topology preservation for each mask
        - Interface clDice: continuous boundary between A and B
        - Mutual exclusivity: penalize mask overlap

    All topological losses operate on sigmoid(logits), not raw logits.

    Args:
        pred_a/pred_b: (B, H, W) raw mask logits
        iou_a/iou_b: (B,) SAM's IoU predictions
        gt_a/gt_b: (B, H, W) binary GT masks
        *_weight: loss component weights
        skeleton_iters: iterations for soft-skeletonization
        interface_dilation_iters: boundary dilation for interface extraction

    Returns:
        total_loss: scalar
        metrics: dict with all component values for logging
    """
    # ---- Phase 1 base losses ---------------------------------------- #
    p1_loss, p1_metrics = phase1_paired_loss(
        pred_a, iou_a, gt_a, pred_b, iou_b, gt_b,
        bce_weight, dice_weight, iou_weight,
    )

    # ---- Phase 2 topological losses --------------------------------- #
    # Soft predictions for topology ops
    soft_a = pred_a.sigmoid()
    soft_b = pred_b.sigmoid()

    # Per-mask clDice
    cldice_a = soft_cldice_loss(soft_a, gt_a, num_iters=skeleton_iters)
    cldice_b = soft_cldice_loss(soft_b, gt_b, num_iters=skeleton_iters)
    cldice_avg = (cldice_a + cldice_b) / 2.0

    # Interface clDice
    iface_cldice = interface_cldice_loss(
        soft_a, soft_b, gt_a, gt_b,
        skel_iters=skeleton_iters,
        dilation_iters=interface_dilation_iters,
    )

    # Mutual exclusivity
    excl = exclusivity_loss(soft_a, soft_b)

    # ---- Total ------------------------------------------------------- #
    total = (
        p1_loss
        + cldice_weight * cldice_avg
        + interface_cldice_weight * iface_cldice
        + exclusivity_weight * excl
    )

    metrics = {
        **p1_metrics,
        "cldice_a": cldice_a.item(),
        "cldice_b": cldice_b.item(),
        "cldice_avg": cldice_avg.item(),
        "interface_cldice": iface_cldice.item(),
        "exclusivity": excl.item(),
        "total": total.item(),
    }
    return total, metrics


# ===================================================================== #
#  Phase 2 (Mask-Focused): Focal + Exclusivity                           #
# ===================================================================== #

def phase2_mask_loss(
    pred_a: torch.Tensor,
    iou_a: torch.Tensor,
    gt_a: torch.Tensor,
    pred_b: torch.Tensor,
    iou_b: torch.Tensor,
    gt_b: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    iou_weight: float = 1.0,
    focal_weight: float = 1.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    exclusivity_weight: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Phase 2 mask-focused loss: Phase 1 losses + focal + exclusivity.

    Replaces boundary-specific losses (clDice, interface_cldice) with
    focal loss for hard-pixel mining. Retains exclusivity to prevent
    mask overlap.

    Args:
        pred_a/pred_b: (B, H, W) raw mask logits
        iou_a/iou_b: (B,) SAM's IoU predictions
        gt_a/gt_b: (B, H, W) binary GT masks
    """
    # Phase 1 base losses (BCE + Dice + IoU prediction)
    p1_loss, p1_metrics = phase1_paired_loss(
        pred_a, iou_a, gt_a, pred_b, iou_b, gt_b,
        bce_weight, dice_weight, iou_weight,
    )

    # Focal loss on each mask
    focal_a = focal_loss(pred_a, gt_a, gamma=focal_gamma, alpha=focal_alpha)
    focal_b = focal_loss(pred_b, gt_b, gamma=focal_gamma, alpha=focal_alpha)
    focal_avg = (focal_a + focal_b) / 2.0

    # Mutual exclusivity
    soft_a = pred_a.sigmoid()
    soft_b = pred_b.sigmoid()
    excl = exclusivity_loss(soft_a, soft_b)

    total = p1_loss + focal_weight * focal_avg + exclusivity_weight * excl

    metrics = {
        **p1_metrics,
        "focal_avg": focal_avg.item(),
        "exclusivity": excl.item(),
        "total": total.item(),
    }
    return total, metrics


# ===================================================================== #
#  Phase 3: Segmentation Loss (direct assignment)                         #
# ===================================================================== #

def phase3_seg_loss(
    pred_a: torch.Tensor,
    iou_a: torch.Tensor,
    pred_b: torch.Tensor,
    iou_b: torch.Tensor,
    gt_a: torch.Tensor,
    gt_b: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    iou_weight: float = 1.0,
    cldice_weight: float = 0.0,
    interface_cldice_weight: float = 0.0,
    exclusivity_weight: float = 0.0,
    skeleton_iters: int = 15,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Phase 3 segmentation loss with direct assignment.

    Uses fixed assignment: pred_a (from <SEG_A>) ↔ gt_a,
    pred_b (from <SEG_B>) ↔ gt_b. The text template binds each SEG
    token to the correct texture via teacher forcing:
      "The transition is from {texture_a} <SEG_A> to {texture_b} <SEG_B>."
    so Hungarian matching is NOT needed and would prevent the model
    from learning the correct A↔B correspondence.

    Args:
        pred_a/pred_b: (B, H, W) raw mask logits from <SEG_A>/<SEG_B>
        iou_a/iou_b: (B,) SAM IoU predictions
        gt_a/gt_b: (B, H, W) binary GT masks
        *_weight: loss component weights

    Returns:
        total_loss: scalar
        metrics: dict with component values
    """
    # ---- Compute loss under direct assignment ------------------------ #
    if cldice_weight > 0 or interface_cldice_weight > 0 or exclusivity_weight > 0:
        loss, metrics = phase2_paired_loss(
            pred_a, iou_a, gt_a,
            pred_b, iou_b, gt_b,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            iou_weight=iou_weight,
            cldice_weight=cldice_weight,
            interface_cldice_weight=interface_cldice_weight,
            exclusivity_weight=exclusivity_weight,
            skeleton_iters=skeleton_iters,
        )
    else:
        loss, metrics = phase1_paired_loss(
            pred_a, iou_a, gt_a,
            pred_b, iou_b, gt_b,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            iou_weight=iou_weight,
        )

    return loss, metrics


def phase3_total_loss(
    lm_loss: torch.Tensor,
    seg_loss: torch.Tensor,
    alignment_loss_val: torch.Tensor | None = None,
    lm_weight: float = 1.0,
    seg_weight: float = 1.0,
    alignment_weight: float = 1.0,
) -> torch.Tensor:
    """Combine language modeling, segmentation, and alignment losses."""
    total = lm_weight * lm_loss + seg_weight * seg_loss
    if alignment_loss_val is not None:
        total = total + alignment_weight * alignment_loss_val
    return total


# ===================================================================== #
#  Phase 3: Contrastive Alignment Loss                                     #
# ===================================================================== #

def alignment_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    c_a: torch.Tensor,
    c_b: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Contrastive alignment loss between predicted SEG hidden states and
    cached Qwen-base texture embeddings, with within-sample hard negatives.

    For each sample, builds a 2x2 cosine similarity matrix:
        [[cos(z_a, c_a), cos(z_a, c_b)],
         [cos(z_b, c_a), cos(z_b, c_b)]]
    and applies cross-entropy where row i should match column i.

    This gives each SEG embedding 1 positive (its own texture) and
    1 hard negative (the paired texture from the same image).

    Args:
        z_a: (B, D) hidden states at <SEG_A> positions (NOT L2-normalized)
        z_b: (B, D) hidden states at <SEG_B> positions
        c_a: (B, D) L2-normalized cached Qwen-base embeddings for texture_a
        c_b: (B, D) L2-normalized cached Qwen-base embeddings for texture_b
        temperature: scaling factor for cosine similarities

    Returns:
        loss: scalar
        metrics: dict with alignment_loss, cos_sim_pos, cos_sim_neg, alignment_acc
    """
    B = z_a.shape[0]

    # L2-normalize predicted embeddings
    z_a_norm = F.normalize(z_a, dim=-1)
    z_b_norm = F.normalize(z_b, dim=-1)

    # Build 2x2 similarity matrices per sample: (B, 2, 2)
    z_stacked = torch.stack([z_a_norm, z_b_norm], dim=1)  # (B, 2, D)
    c_stacked = torch.stack([c_a, c_b], dim=1)            # (B, 2, D)

    # sim[b, i, j] = cos(z_i, c_j) / temperature
    sim = torch.bmm(z_stacked, c_stacked.transpose(1, 2)) / temperature

    # Target: row 0 → col 0, row 1 → col 1
    targets = torch.arange(2, device=z_a.device).unsqueeze(0).expand(B, -1)  # (B, 2)

    # Cross-entropy over the 2-class dimension
    loss = F.cross_entropy(
        sim.view(B * 2, 2),
        targets.reshape(B * 2),
    )

    # Metrics (detached)
    with torch.no_grad():
        cos_pos = (sim[:, 0, 0] + sim[:, 1, 1]).mean() * temperature / 2.0
        cos_neg = (sim[:, 0, 1] + sim[:, 1, 0]).mean() * temperature / 2.0
        preds = sim.view(B * 2, 2).argmax(dim=1)
        accuracy = (preds == targets.reshape(B * 2)).float().mean()

    metrics = {
        "alignment_loss": loss.item(),
        "cos_sim_pos": cos_pos.item(),
        "cos_sim_neg": cos_neg.item(),
        "alignment_acc": accuracy.item(),
    }
    return loss, metrics


class AlignmentMemoryBank:
    """
    Memory bank for richer contrastive negatives in alignment loss.

    Stores recent target embeddings (c_a, c_b) from past batches to expand
    the similarity matrix beyond the current 2x2 per-sample structure.
    """

    def __init__(self, bank_size: int = 32, embed_dim: int = 768):
        self.bank_size = bank_size
        self.embed_dim = embed_dim
        self.bank = None  # (K, D) tensor of past target embeddings
        self.ptr = 0

    @torch.no_grad()
    def update(self, c_a: torch.Tensor, c_b: torch.Tensor):
        """Add new target embeddings to the bank."""
        new = torch.cat([c_a, c_b], dim=0).detach().cpu()  # (2B, D)
        if self.bank is None:
            self.bank = torch.zeros(self.bank_size, new.shape[-1])
        for emb in new:
            self.bank[self.ptr % self.bank_size] = emb
            self.ptr += 1

    def get_negatives(self, device: torch.device) -> torch.Tensor | None:
        """Return bank embeddings as additional negatives. (K, D)"""
        if self.bank is None or self.ptr == 0:
            return None
        valid = min(self.ptr, self.bank_size)
        return self.bank[:valid].to(device)


def alignment_loss_with_bank(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    c_a: torch.Tensor,
    c_b: torch.Tensor,
    bank: AlignmentMemoryBank | None = None,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Contrastive alignment loss with optional memory bank for richer negatives.

    Without bank: same as alignment_loss (2x2 matrix per sample).
    With bank: expands to 2x(2+K) matrix where K is bank size, providing
    more hard negatives to prevent early saturation.
    """
    B = z_a.shape[0]

    z_a_norm = F.normalize(z_a, dim=-1)
    z_b_norm = F.normalize(z_b, dim=-1)

    # Within-sample 2x2 (always present)
    z_stacked = torch.stack([z_a_norm, z_b_norm], dim=1)  # (B, 2, D)
    c_stacked = torch.stack([c_a, c_b], dim=1)            # (B, 2, D)

    if bank is not None:
        bank_negs = bank.get_negatives(z_a.device)  # (K, D) or None
    else:
        bank_negs = None

    if bank_negs is not None and len(bank_negs) > 0:
        K = bank_negs.shape[0]
        # Expand: each sample sees its 2 targets + K bank negatives
        # c_expanded: (B, 2+K, D)
        bank_expanded = bank_negs.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)
        c_expanded = torch.cat([c_stacked, bank_expanded], dim=1)  # (B, 2+K, D)

        # sim: (B, 2, 2+K) — each z against all targets
        sim = torch.bmm(z_stacked, c_expanded.transpose(1, 2)) / temperature

        # Target: row 0 → col 0, row 1 → col 1 (first 2 columns are positives)
        targets = torch.arange(2, device=z_a.device).unsqueeze(0).expand(B, -1)

        loss = F.cross_entropy(
            sim.view(B * 2, 2 + K),
            targets.reshape(B * 2),
        )
    else:
        # Fallback: standard 2x2
        sim = torch.bmm(z_stacked, c_stacked.transpose(1, 2)) / temperature
        targets = torch.arange(2, device=z_a.device).unsqueeze(0).expand(B, -1)
        loss = F.cross_entropy(sim.view(B * 2, 2), targets.reshape(B * 2))

    # Update bank with current targets
    if bank is not None:
        bank.update(c_a, c_b)

    # Metrics
    with torch.no_grad():
        sim_2x2 = torch.bmm(z_stacked, c_stacked.transpose(1, 2)) / temperature
        cos_pos = (sim_2x2[:, 0, 0] + sim_2x2[:, 1, 1]).mean() * temperature / 2.0
        cos_neg = (sim_2x2[:, 0, 1] + sim_2x2[:, 1, 0]).mean() * temperature / 2.0
        preds = sim_2x2.view(B * 2, 2).argmax(dim=1)
        accuracy = (preds == targets.reshape(B * 2)).float().mean()

    metrics = {
        "alignment_loss": loss.item(),
        "cos_sim_pos": cos_pos.item(),
        "cos_sim_neg": cos_neg.item(),
        "alignment_acc": accuracy.item(),
    }
    return loss, metrics
