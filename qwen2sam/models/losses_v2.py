"""
Loss functions for Qwen2SAM v2 (SAM3 DETR-based architecture).

Losses:
  - Mask: Sigmoid Focal + Dice (applied to Hungarian-matched query)
  - Box: L1 + Generalized IoU (applied to matched query)
  - Classification: Sigmoid Focal (1 positive, 199 negatives)
  - Alignment: Contrastive cosine similarity (trains LoRA)

Hungarian matching: Simplified for our case (200 queries → 1 GT target).
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ===================================================================== #
#  Focal Loss                                                             #
# ===================================================================== #

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Sigmoid focal loss (Lin et al., 2017).

    Args:
        inputs: logits, any shape
        targets: same shape, values in [0, 1]
        alpha: balancing factor
        gamma: focusing parameter
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


# ===================================================================== #
#  Dice Loss                                                              #
# ===================================================================== #

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Dice loss for binary segmentation.

    Args:
        inputs: logits, (N, H*W) or (N, H, W)
        targets: binary, same shape
    """
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)

    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    if reduction == "mean":
        return loss.mean()
    return loss.sum()


# ===================================================================== #
#  Box Losses                                                             #
# ===================================================================== #

def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU between two sets of boxes in xyxy format.

    Args:
        boxes1, boxes2: (N, 4) in xyxy format

    Returns:
        (N,) generalized IoU values
    """
    # Intersection
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter_area

    iou = inter_area / union.clamp(min=1e-6)

    # Enclosing box
    enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return giou


def box_loss(
    pred_boxes: torch.Tensor,
    pred_boxes_xyxy: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    l1_weight: float = 5.0,
    giou_weight: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """
    Combined L1 + GIoU box loss on matched queries.

    Args:
        pred_boxes: (N, 4) cxcywh normalized
        pred_boxes_xyxy: (N, 4) xyxy normalized
        gt_boxes: (N, 4) cxcywh normalized
        gt_boxes_xyxy: (N, 4) xyxy normalized

    Returns:
        (loss, metrics_dict)
    """
    loss_l1 = F.l1_loss(pred_boxes, gt_boxes, reduction="mean")
    loss_giou = 1 - generalized_box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
    loss_giou = loss_giou.mean()

    total = l1_weight * loss_l1 + giou_weight * loss_giou
    return total, {"box_l1": loss_l1.item(), "box_giou": loss_giou.item()}


# ===================================================================== #
#  Hungarian Matching (simplified: Q queries → 1 target)                  #
# ===================================================================== #

@torch.no_grad()
def hungarian_match(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_boxes_xyxy: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    cost_class: float = 2.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
) -> torch.Tensor:
    """
    Hungarian matching: find the best query for each GT target.

    Simplified for our case: 1 GT target per sample.

    Args:
        pred_logits: (B, Q, 1) predicted class logits
        pred_boxes: (B, Q, 4) predicted boxes (cxcywh)
        pred_boxes_xyxy: (B, Q, 4) predicted boxes (xyxy)
        gt_boxes: (B, 4) GT boxes (cxcywh)
        gt_boxes_xyxy: (B, 4) GT boxes (xyxy)

    Returns:
        matched_indices: (B,) index of matched query per sample
    """
    B, Q, _ = pred_logits.shape

    # Classification cost (focal)
    out_prob = pred_logits.squeeze(-1).sigmoid()  # (B, Q)
    cost_cls = -(0.25 * (1 - out_prob) ** 2 * torch.log(out_prob + 1e-8))  # (B, Q)

    # L1 box cost: (B, Q, 4) vs (B, 1, 4)
    cost_l1 = torch.cdist(pred_boxes, gt_boxes.unsqueeze(1), p=1).squeeze(-1)  # (B, Q)

    # GIoU cost
    cost_giou_vals = torch.zeros(B, Q, device=pred_logits.device)
    for b in range(B):
        gt_exp = gt_boxes_xyxy[b].unsqueeze(0).expand(Q, -1)  # (Q, 4)
        cost_giou_vals[b] = -generalized_box_iou(pred_boxes_xyxy[b], gt_exp)

    # Total cost
    C = cost_class * cost_cls + cost_bbox * cost_l1 + cost_giou * cost_giou_vals

    # For 1 target, just argmin
    matched_indices = C.argmin(dim=1)  # (B,)
    return matched_indices


# ===================================================================== #
#  DETR Segmentation Loss                                                 #
# ===================================================================== #

def detr_seg_loss(
    pred_masks: torch.Tensor,
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_boxes_xyxy: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    focal_weight: float = 5.0,
    dice_weight: float = 1.0,
    cls_weight: float = 2.0,
    box_l1_weight: float = 5.0,
    box_giou_weight: float = 2.0,
    hires_pixel: torch.Tensor = None,
    hires_queries: torch.Tensor = None,
) -> tuple[torch.Tensor, dict]:
    """
    Full DETR segmentation loss for one texture.

    1. Hungarian match 200 queries to 1 GT
    2. Mask loss on matched query
    3. Box loss on matched query
    4. Classification loss on all queries (1 positive, 199 negative)

    Args:
        pred_masks: (B, Q, H_pred, W_pred) logits
        pred_logits: (B, Q, 1) class logits
        pred_boxes: (B, Q, 4) cxcywh normalized
        pred_boxes_xyxy: (B, Q, 4) xyxy normalized
        gt_masks: (B, H_gt, W_gt) binary masks
        gt_boxes: (B, 4) cxcywh normalized
        gt_boxes_xyxy: (B, 4) xyxy normalized

    Returns:
        (loss, metrics_dict)
    """
    B, Q = pred_logits.shape[:2]
    device = pred_logits.device

    # ---- Hungarian matching ------------------------------------------ #
    matched_idx = hungarian_match(
        pred_logits, pred_boxes, pred_boxes_xyxy,
        gt_boxes, gt_boxes_xyxy,
    )  # (B,)

    batch_idx = torch.arange(B, device=device)

    # ---- Mask loss (focal + dice) on matched queries ----------------- #
    if hires_pixel is not None and hires_queries is not None:
        # High-res path: compute mask at target resolution via dot product
        matched_q = hires_queries[batch_idx, matched_idx]  # (B, thin_dim)
        if hires_pixel.ndim == 3:
            # B=1 squeezed: (thin, H, W)
            matched_masks = torch.einsum("bc,chw->bhw", matched_q, hires_pixel)
        else:
            matched_masks = torch.einsum("bc,bchw->bhw", matched_q, hires_pixel)
    else:
        # Fallback: bilinear upsample from 288
        matched_masks = pred_masks[batch_idx, matched_idx]  # (B, H_pred, W_pred)
        if matched_masks.shape[-2:] != gt_masks.shape[-2:]:
            matched_masks = F.interpolate(
                matched_masks.unsqueeze(1).float(),
                size=gt_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

    loss_focal = sigmoid_focal_loss(matched_masks, gt_masks, reduction="mean")
    loss_dice = dice_loss(matched_masks, gt_masks, reduction="mean")

    # ---- Box loss on matched queries --------------------------------- #
    matched_boxes = pred_boxes[batch_idx, matched_idx]         # (B, 4)
    matched_boxes_xyxy = pred_boxes_xyxy[batch_idx, matched_idx]  # (B, 4)
    loss_box, box_metrics = box_loss(
        matched_boxes, matched_boxes_xyxy,
        gt_boxes, gt_boxes_xyxy,
        l1_weight=box_l1_weight,
        giou_weight=box_giou_weight,
    )

    # ---- Classification loss on all queries -------------------------- #
    target_cls = torch.zeros(B, Q, device=device)
    target_cls[batch_idx, matched_idx] = 1.0
    loss_cls = sigmoid_focal_loss(
        pred_logits.squeeze(-1), target_cls, reduction="mean"
    )

    # ---- Total ------------------------------------------------------- #
    total = (
        focal_weight * loss_focal
        + dice_weight * loss_dice
        + cls_weight * loss_cls
        + loss_box
    )

    # ---- Compute mask IoU for monitoring ----------------------------- #
    with torch.no_grad():
        pred_binary = (matched_masks.detach().sigmoid() > 0.5).float()
        intersection = (pred_binary * gt_masks).sum(dim=(-2, -1))
        union = pred_binary.sum(dim=(-2, -1)) + gt_masks.sum(dim=(-2, -1)) - intersection
        mask_iou = (intersection / union.clamp(min=1)).mean()

    metrics = {
        "focal": loss_focal.item(),
        "dice": loss_dice.item(),
        "cls": loss_cls.item(),
        "mask_iou": mask_iou.item(),
        **box_metrics,
    }
    return total, metrics


# ===================================================================== #
#  Combined loss for both textures                                        #
# ===================================================================== #

@torch.no_grad()
def _quick_mask_iou(pred_masks, pred_logits, gt_masks):
    """Quick IoU between best-query prediction and GT (for Hungarian A/B matching)."""
    B, Q = pred_logits.shape[:2]
    scores = pred_logits.squeeze(-1).sigmoid()
    best_idx = scores.argmax(dim=1)                        # (B,)
    batch_idx = torch.arange(B, device=pred_masks.device)
    matched = pred_masks[batch_idx, best_idx]              # (B, H, W)
    if matched.shape[-2:] != gt_masks.shape[-2:]:
        matched = F.interpolate(
            matched.unsqueeze(1).float(), size=gt_masks.shape[-2:],
            mode="bilinear", align_corners=False,
        ).squeeze(1)
    pred_bin = (matched.sigmoid() > 0.5).float()
    intersection = (pred_bin * gt_masks).flatten(1).sum(1)
    union = (pred_bin + gt_masks).flatten(1).clamp(max=1).sum(1)
    return (intersection / union.clamp(min=1)).mean()


def v2_seg_loss(
    outputs: dict,
    gt_masks_a: torch.Tensor,
    gt_masks_b: torch.Tensor,
    gt_boxes_a: torch.Tensor,
    gt_boxes_b: torch.Tensor,
    gt_boxes_xyxy_a: torch.Tensor,
    gt_boxes_xyxy_b: torch.Tensor,
    focal_weight: float = 5.0,
    dice_weight: float = 1.0,
    cls_weight: float = 2.0,
    box_l1_weight: float = 5.0,
    box_giou_weight: float = 2.0,
    exclusivity_weight: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """
    Compute segmentation loss for both texture A and B.

    Includes A/B Hungarian matching: tries both direct (pred_a→gt_a)
    and swapped (pred_a→gt_b) assignments, uses whichever gives better IoU.
    This prevents the model from learning a consistent A↔B swap.

    Also includes mutual exclusivity loss to penalize mask overlap.
    """
    B = gt_masks_a.shape[0]

    # ---- A/B Hungarian matching: pick best assignment ---- #
    with torch.no_grad():
        # Direct: pred_a→gt_a, pred_b→gt_b
        iou_direct = (
            _quick_mask_iou(outputs["pred_masks_a"], outputs["pred_logits_a"], gt_masks_a) +
            _quick_mask_iou(outputs["pred_masks_b"], outputs["pred_logits_b"], gt_masks_b)
        )
        # Swapped: pred_a→gt_b, pred_b→gt_a
        iou_swapped = (
            _quick_mask_iou(outputs["pred_masks_a"], outputs["pred_logits_a"], gt_masks_b) +
            _quick_mask_iou(outputs["pred_masks_b"], outputs["pred_logits_b"], gt_masks_a)
        )

    # Extract hires features (None if not present)
    hires_pixel_a = outputs.get("hires_pixel_a")
    hires_queries_a = outputs.get("hires_queries_a")
    hires_pixel_b = outputs.get("hires_pixel_b")
    hires_queries_b = outputs.get("hires_queries_b")

    if iou_swapped > iou_direct:
        # Swap GT assignments so loss is computed with the better match
        # Note: hires features are tied to pred_a/pred_b, NOT swapped
        gt_masks_a, gt_masks_b = gt_masks_b, gt_masks_a
        gt_boxes_a, gt_boxes_b = gt_boxes_b, gt_boxes_a
        gt_boxes_xyxy_a, gt_boxes_xyxy_b = gt_boxes_xyxy_b, gt_boxes_xyxy_a

    loss_a, met_a = detr_seg_loss(
        outputs["pred_masks_a"], outputs["pred_logits_a"],
        outputs["pred_boxes_a"], outputs["pred_boxes_xyxy_a"],
        gt_masks_a, gt_boxes_a, gt_boxes_xyxy_a,
        focal_weight, dice_weight, cls_weight, box_l1_weight, box_giou_weight,
        hires_pixel=hires_pixel_a, hires_queries=hires_queries_a,
    )
    loss_b, met_b = detr_seg_loss(
        outputs["pred_masks_b"], outputs["pred_logits_b"],
        outputs["pred_boxes_b"], outputs["pred_boxes_xyxy_b"],
        gt_masks_b, gt_boxes_b, gt_boxes_xyxy_b,
        focal_weight, dice_weight, cls_weight, box_l1_weight, box_giou_weight,
        hires_pixel=hires_pixel_b, hires_queries=hires_queries_b,
    )

    total = (loss_a + loss_b) / 2.0

    # ---- Mutual exclusivity loss --------------------------------- #
    # Penalize overlap between best-scoring mask_a and mask_b.
    # Index selection is non-differentiable but the selected mask logits
    # retain gradient connection.
    excl_loss = torch.tensor(0.0, device=gt_masks_a.device)
    if exclusivity_weight > 0:
        scores_a = outputs["pred_logits_a"].squeeze(-1).sigmoid()
        best_a = scores_a.argmax(dim=1).detach()
        scores_b = outputs["pred_logits_b"].squeeze(-1).sigmoid()
        best_b = scores_b.argmax(dim=1).detach()
        batch_idx = torch.arange(B, device=gt_masks_a.device)
        mask_a_sel = outputs["pred_masks_a"][batch_idx, best_a].sigmoid()  # (B, H, W)
        mask_b_sel = outputs["pred_masks_b"][batch_idx, best_b].sigmoid()  # (B, H, W)
        excl_loss = (mask_a_sel * mask_b_sel).mean()
        total = total + exclusivity_weight * excl_loss

    metrics = {
        "seg_total": total.item(),
        "mask_iou": (met_a["mask_iou"] + met_b["mask_iou"]) / 2.0,
        "focal": (met_a["focal"] + met_b["focal"]) / 2.0,
        "dice": (met_a["dice"] + met_b["dice"]) / 2.0,
        "cls": (met_a["cls"] + met_b["cls"]) / 2.0,
        "box_l1": (met_a["box_l1"] + met_b["box_l1"]) / 2.0,
        "box_giou": (met_a["box_giou"] + met_b["box_giou"]) / 2.0,
        "exclusivity": excl_loss.item(),
    }
    return total, metrics


def v2_total_loss(
    lm_loss: torch.Tensor,
    seg_loss: torch.Tensor,
    alignment_loss_val: torch.Tensor = None,
    lm_weight: float = 0.0,
    seg_weight: float = 1.0,
    alignment_weight: float = 1.0,
) -> torch.Tensor:
    """Weighted combination of all losses."""
    total = seg_weight * seg_loss
    if lm_loss is not None and lm_weight > 0:
        total = total + lm_weight * lm_loss
    if alignment_loss_val is not None and alignment_weight > 0:
        total = total + alignment_weight * alignment_loss_val
    return total
