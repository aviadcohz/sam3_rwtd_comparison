"""
Loss functions for Qwen2SAM v2_tracker (DETR + SAM3 Tracker).

Combines:
  - DETR segmentation loss (from losses_v2.py) on coarse DETR masks
  - Tracker mask loss (focal + dice) on refined masks
  - Alignment loss (contrastive) on SEG token embeddings
  - A/B Hungarian matching for both DETR and tracker paths
"""

import torch
import torch.nn.functional as F

from qwen2sam.models.losses_v2 import (
    sigmoid_focal_loss,
    dice_loss,
    v2_seg_loss,
)
from qwen2sam.models.losses import alignment_loss


# ===================================================================== #
#  Tracker mask loss (focal + dice on refined masks)                      #
# ===================================================================== #

@torch.no_grad()
def _tracker_quick_iou(refined_mask, gt_mask):
    """Quick IoU between refined mask and GT for A/B matching."""
    # refined_mask: (B, 1, H, W) logits
    # gt_mask: (B, H_gt, W_gt) binary
    pred = refined_mask.squeeze(1)  # (B, H, W)
    if pred.shape[-2:] != gt_mask.shape[-2:]:
        pred = F.interpolate(
            pred.unsqueeze(1).float(), size=gt_mask.shape[-2:],
            mode="bilinear", align_corners=False,
        ).squeeze(1)
    pred_bin = (pred.sigmoid() > 0.5).float()
    intersection = (pred_bin * gt_mask).flatten(1).sum(1)
    union = (pred_bin + gt_mask).flatten(1).clamp(max=1).sum(1)
    return (intersection / union.clamp(min=1)).mean()


def tracker_mask_loss(
    refined_a: torch.Tensor,
    refined_b: torch.Tensor,
    gt_masks_a: torch.Tensor,
    gt_masks_b: torch.Tensor,
    focal_weight: float = 5.0,
    dice_weight: float = 1.0,
    exclusivity_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """
    Focal + Dice loss on tracker refined masks for both textures.

    Includes A/B Hungarian matching: swaps GT if swapped gives better IoU.

    Args:
        refined_a: (B, 1, 288, 288) logits from tracker for texture A
        refined_b: (B, 1, 288, 288) logits from tracker for texture B
        gt_masks_a: (B, H_gt, W_gt) binary GT masks
        gt_masks_b: (B, H_gt, W_gt) binary GT masks
        focal_weight: weight for focal loss
        dice_weight: weight for dice loss
        exclusivity_weight: penalty for mask overlap

    Returns:
        (loss, metrics_dict)
    """
    # ---- A/B Hungarian matching ---------------------------------------- #
    with torch.no_grad():
        iou_direct = (
            _tracker_quick_iou(refined_a, gt_masks_a)
            + _tracker_quick_iou(refined_b, gt_masks_b)
        )
        iou_swapped = (
            _tracker_quick_iou(refined_a, gt_masks_b)
            + _tracker_quick_iou(refined_b, gt_masks_a)
        )

    if iou_swapped > iou_direct:
        gt_masks_a, gt_masks_b = gt_masks_b, gt_masks_a

    # ---- Downsample GT to match refined mask resolution (288×288) ------ #
    target_size = refined_a.shape[-2:]  # (288, 288)

    def _resize_gt(gt):
        if gt.shape[-2:] != target_size:
            return F.interpolate(
                gt.unsqueeze(1).float(), size=target_size,
                mode="bilinear", align_corners=False,
            ).squeeze(1)
        return gt.float()

    gt_a = _resize_gt(gt_masks_a)  # (B, 288, 288)
    gt_b = _resize_gt(gt_masks_b)

    # Squeeze refined masks: (B, 1, H, W) → (B, H, W)
    ref_a = refined_a.squeeze(1)
    ref_b = refined_b.squeeze(1)

    # ---- Focal + Dice loss for each texture ---------------------------- #
    loss_focal_a = sigmoid_focal_loss(ref_a, gt_a, reduction="mean")
    loss_dice_a = dice_loss(ref_a, gt_a, reduction="mean")
    loss_focal_b = sigmoid_focal_loss(ref_b, gt_b, reduction="mean")
    loss_dice_b = dice_loss(ref_b, gt_b, reduction="mean")

    loss_a = focal_weight * loss_focal_a + dice_weight * loss_dice_a
    loss_b = focal_weight * loss_focal_b + dice_weight * loss_dice_b
    total = (loss_a + loss_b) / 2.0

    # ---- Exclusivity loss on refined masks ----------------------------- #
    excl_loss = torch.tensor(0.0, device=refined_a.device)
    if exclusivity_weight > 0:
        excl_loss = (ref_a.sigmoid() * ref_b.sigmoid()).mean()
        total = total + exclusivity_weight * excl_loss

    # ---- Compute tracker IoU for monitoring ---------------------------- #
    with torch.no_grad():
        pred_a_bin = (ref_a.detach().sigmoid() > 0.5).float()
        pred_b_bin = (ref_b.detach().sigmoid() > 0.5).float()
        inter_a = (pred_a_bin * gt_a).sum(dim=(-2, -1))
        union_a = pred_a_bin.sum(dim=(-2, -1)) + gt_a.sum(dim=(-2, -1)) - inter_a
        iou_a = (inter_a / union_a.clamp(min=1)).mean()
        inter_b = (pred_b_bin * gt_b).sum(dim=(-2, -1))
        union_b = pred_b_bin.sum(dim=(-2, -1)) + gt_b.sum(dim=(-2, -1)) - inter_b
        iou_b = (inter_b / union_b.clamp(min=1)).mean()
        tracker_iou = (iou_a + iou_b) / 2.0

    metrics = {
        "tracker_total": total.item(),
        "tracker_focal": (loss_focal_a.item() + loss_focal_b.item()) / 2.0,
        "tracker_dice": (loss_dice_a.item() + loss_dice_b.item()) / 2.0,
        "tracker_exclusivity": excl_loss.item(),
        "tracker_iou": tracker_iou.item(),
    }
    return total, metrics


# ===================================================================== #
#  Combined total loss                                                    #
# ===================================================================== #

def v2_tracker_total_loss(
    outputs: dict,
    gt_masks_a: torch.Tensor,
    gt_masks_b: torch.Tensor,
    gt_boxes_a: torch.Tensor,
    gt_boxes_b: torch.Tensor,
    gt_boxes_xyxy_a: torch.Tensor,
    gt_boxes_xyxy_b: torch.Tensor,
    align_target_a: torch.Tensor = None,
    align_target_b: torch.Tensor = None,
    loss_cfg: dict = None,
) -> tuple[torch.Tensor, dict]:
    """
    Combined loss: DETR + Tracker + Alignment.

    Args:
        outputs: dict from Qwen2SAMv2Tracker.forward()
        gt_masks_a/b: (B, H, W) binary masks
        gt_boxes_a/b: (B, 4) cxcywh, gt_boxes_xyxy_a/b: (B, 4) xyxy
        align_target_a/b: (B, D) L2-normalized cached Qwen embeddings
        loss_cfg: loss weight config dict

    Returns:
        (total_loss, metrics_dict)
    """
    if loss_cfg is None:
        loss_cfg = {}

    detr_weight = loss_cfg.get("detr_weight", 0.5)
    tracker_weight = loss_cfg.get("tracker_weight", 1.0)
    alignment_weight = loss_cfg.get("alignment_weight", 1.0)
    temperature = loss_cfg.get("alignment_temperature", 0.07)

    metrics = {}

    # ---- DETR segmentation loss ---------------------------------------- #
    detr_loss = torch.tensor(0.0, device=gt_masks_a.device)
    if detr_weight > 0:
        detr_loss, detr_met = v2_seg_loss(
            outputs, gt_masks_a, gt_masks_b,
            gt_boxes_a, gt_boxes_b, gt_boxes_xyxy_a, gt_boxes_xyxy_b,
            focal_weight=loss_cfg.get("focal_weight", 5.0),
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            cls_weight=loss_cfg.get("cls_weight", 2.0),
            box_l1_weight=loss_cfg.get("box_l1_weight", 5.0),
            box_giou_weight=loss_cfg.get("box_giou_weight", 2.0),
            exclusivity_weight=loss_cfg.get("exclusivity_weight", 0.5),
        )
        metrics.update({f"detr_{k}": v for k, v in detr_met.items()})
    else:
        # Still compute DETR IoU for monitoring even when loss is disabled
        with torch.no_grad():
            _, detr_met = v2_seg_loss(
                outputs, gt_masks_a, gt_masks_b,
                gt_boxes_a, gt_boxes_b, gt_boxes_xyxy_a, gt_boxes_xyxy_b,
            )
            metrics["detr_mask_iou"] = detr_met.get("mask_iou", 0)

    # ---- Tracker mask loss --------------------------------------------- #
    trk_loss = torch.tensor(0.0, device=gt_masks_a.device)
    if tracker_weight > 0 and "refined_masks_a" in outputs:
        trk_loss, trk_met = tracker_mask_loss(
            outputs["refined_masks_a"], outputs["refined_masks_b"],
            gt_masks_a, gt_masks_b,
            focal_weight=loss_cfg.get("tracker_focal_weight", 5.0),
            dice_weight=loss_cfg.get("tracker_dice_weight", 1.0),
            exclusivity_weight=loss_cfg.get("tracker_exclusivity_weight", 0.5),
        )
        metrics.update(trk_met)

    # ---- Alignment loss ------------------------------------------------ #
    align_loss = torch.tensor(0.0, device=gt_masks_a.device)
    if alignment_weight > 0 and align_target_a is not None:
        align_loss, align_met = alignment_loss(
            outputs["align_a"], outputs["align_b"],
            align_target_a, align_target_b,
            temperature=temperature,
        )
        metrics.update(align_met)

    # ---- Reconstruction loss (autoencoder) ------------------------------ #
    recon_weight = loss_cfg.get("reconstruction_weight", 0.0)
    recon_loss = torch.tensor(0.0, device=gt_masks_a.device)
    if recon_weight > 0 and "point_reconstructed" in outputs:
        recon_loss = F.mse_loss(
            outputs["point_reconstructed"],
            outputs["point_hidden"].detach(),  # detach target to avoid double gradient
        )
        metrics["reconstruction_loss"] = recon_loss.item()

    # ---- LM loss (texture description generation) ----------------------- #
    lm_weight = loss_cfg.get("lm_weight", 0.0)
    lm_loss = torch.tensor(0.0, device=gt_masks_a.device)
    if lm_weight > 0 and outputs.get("lm_loss") is not None:
        lm_loss = outputs["lm_loss"]

    # ---- Total --------------------------------------------------------- #
    total = (
        detr_weight * detr_loss
        + tracker_weight * trk_loss
        + alignment_weight * align_loss
        + recon_weight * recon_loss
        + lm_weight * lm_loss
    )

    metrics["total"] = total.item()
    metrics["detr_loss"] = detr_loss.item()
    metrics["tracker_loss"] = trk_loss.item()
    metrics["align_loss"] = align_loss.item()
    metrics["recon_loss"] = recon_loss.item()
    metrics["lm_loss"] = lm_loss.item()

    return total, metrics
