"""
Differentiable Soft-Skeletonization and Soft-clDice loss.

Phase 2 (Boundary Refinement) enforces crisp, continuous, single-pixel
interfaces between Mask A and Mask B using topology-aware losses.

Reference: Shit et al., "clDice - a Novel Topology-Preserving Loss
Function for Tubular Structure Segmentation" (MICCAI 2021)

Implements:
  Morphological ops:
    - soft_erode()       — MinPool via negated MaxPool
    - soft_dilate()      — MaxPool
    - soft_open()        — dilate(erode(x))

  Skeletonization:
    - soft_skeletonize() — iterative peel-and-accumulate

  Loss functions:
    - soft_cldice_loss()       — clDice on a single mask vs GT
    - extract_boundary()       — differentiable boundary extraction
    - extract_interface()      — interface between two masks
    - interface_cldice_loss()  — clDice on the A/B interface
    - exclusivity_loss()       — penalize mask overlap
"""

import torch
import torch.nn.functional as F


# ===================================================================== #
#  Differentiable Morphological Operations                                #
#  All expect (B, 1, H, W) input and return (B, 1, H, W).                #
# ===================================================================== #

def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is (B, 1, H, W). Accepts (B, H, W) or (B, 1, H, W)."""
    if x.ndim == 3:
        return x.unsqueeze(1)
    return x


def _squeeze_if_needed(x: torch.Tensor, was_3d: bool) -> torch.Tensor:
    """Squeeze back to (B, H, W) if input was 3D."""
    if was_3d:
        return x.squeeze(1)
    return x


def soft_erode(mask: torch.Tensor) -> torch.Tensor:
    """
    Differentiable morphological erosion.

    Uses separated 1D min-pooling (via negated max-pool) for a better
    approximation of disk-shaped erosion than a single 2D square kernel.

    Args:
        mask: (B, 1, H, W) or (B, H, W), soft mask in [0, 1]

    Returns:
        Same shape as input, eroded mask.
    """
    was_3d = mask.ndim == 3
    mask = _ensure_4d(mask)

    # Horizontal erosion (3x1 kernel)
    p1 = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
    # Vertical erosion (1x3 kernel)
    p2 = -F.max_pool2d(-p1, kernel_size=(1, 3), stride=1, padding=(0, 1))

    return _squeeze_if_needed(p2, was_3d)


def soft_dilate(mask: torch.Tensor) -> torch.Tensor:
    """
    Differentiable morphological dilation via 3x3 MaxPool.

    Args:
        mask: (B, 1, H, W) or (B, H, W), soft mask in [0, 1]

    Returns:
        Same shape as input, dilated mask.
    """
    was_3d = mask.ndim == 3
    mask = _ensure_4d(mask)

    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

    return _squeeze_if_needed(dilated, was_3d)


def soft_open(mask: torch.Tensor) -> torch.Tensor:
    """
    Differentiable morphological opening: dilate(erode(x)).

    Removes small foreground protrusions while preserving shape.
    """
    return soft_dilate(soft_erode(mask))


def soft_close(mask: torch.Tensor) -> torch.Tensor:
    """
    Differentiable morphological closing: erode(dilate(x)).

    Fills small background holes while preserving shape.
    """
    return soft_erode(soft_dilate(mask))


# ===================================================================== #
#  Differentiable Soft-Skeletonization                                    #
# ===================================================================== #

def soft_skeletonize(
    mask: torch.Tensor,
    num_iters: int = 10,
) -> torch.Tensor:
    """
    Differentiable soft-skeletonization via iterative erosion.

    At each iteration, extracts the "topological peel" — pixels that
    would be removed by morphological opening but not by erosion alone.
    The union of all peels across scales approximates the skeleton.

    Algorithm (from clDice paper):
        skel = relu(img - open(img))           # initial peel
        for i in range(num_iters):
            img = erode(img)                    # shrink
            skel += relu(img - open(img))       # accumulate peel
        return skel

    Args:
        mask: (B, 1, H, W) or (B, H, W), soft mask in [0, 1]
        num_iters: number of erosion iterations (controls max skeleton
                   width; 10-15 is typical for 256x256, 15-20 for 1024)

    Returns:
        Same shape as input, soft skeleton in [0, ~num_iters].
        Higher values = stronger skeleton response.
    """
    was_3d = mask.ndim == 3
    mask = _ensure_4d(mask)

    # Initial peel
    skel = F.relu(mask - soft_open(mask))

    current = mask
    for _ in range(num_iters):
        current = soft_erode(current)
        skel = skel + F.relu(current - soft_open(current))

    return _squeeze_if_needed(skel, was_3d)


# ===================================================================== #
#  Boundary and Interface Extraction                                      #
# ===================================================================== #

def extract_boundary(mask: torch.Tensor) -> torch.Tensor:
    """
    Extract the 1-pixel boundary of a mask.

    boundary = mask - erode(mask)

    For a binary mask, this gives a ring of 1s along the edge.
    For a soft mask, it gives a soft boundary.

    Args:
        mask: (B, 1, H, W) or (B, H, W), soft mask in [0, 1]

    Returns:
        Same shape, boundary pixels.
    """
    return F.relu(mask - soft_erode(mask))


def extract_interface(
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    dilation_iters: int = 2,
) -> torch.Tensor:
    """
    Extract the interface region where Mask A meets Mask B.

    For complementary masks (A + B ≈ 1 in the active region), the
    interface is the thin line where both boundaries coincide.

    Method:
        1. Extract boundary of each mask
        2. Dilate each boundary slightly (so they overlap at the seam)
        3. Interface = boundary_A * boundary_B  (intersection)

    Args:
        mask_a: (B, 1, H, W) or (B, H, W), soft mask A in [0, 1]
        mask_b: (B, 1, H, W) or (B, H, W), soft mask B in [0, 1]
        dilation_iters: how many times to dilate boundaries before
                        intersection (ensures overlap for complementary masks)

    Returns:
        Same shape, soft interface region.
    """
    boundary_a = extract_boundary(mask_a)
    boundary_b = extract_boundary(mask_b)

    # Dilate so that complementary mask boundaries actually overlap
    for _ in range(dilation_iters):
        boundary_a = soft_dilate(boundary_a)
        boundary_b = soft_dilate(boundary_b)

    return boundary_a * boundary_b


# ===================================================================== #
#  Loss Functions                                                         #
# ===================================================================== #

def soft_cldice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_iters: int = 10,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft Centerline Dice loss for a single mask.

    Measures topology-preserving agreement between prediction and GT
    using their skeletons:

        Tprec = |S_pred ∩ V_gt| / |S_pred|
        Tsens = |S_gt ∩ V_pred| / |S_gt|
        clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
        loss = 1 - clDice

    Where S = skeleton, V = volume (full mask).

    Args:
        pred: (B, H, W) or (B, 1, H, W), soft prediction in [0, 1]
              (apply sigmoid before calling if you have logits)
        target: same shape, binary GT
        num_iters: skeleton iterations
        smooth: Laplace smoothing

    Returns:
        Scalar loss in [0, 1].
    """
    was_3d = pred.ndim == 3
    pred_4d = _ensure_4d(pred)
    target_4d = _ensure_4d(target)

    skel_pred = soft_skeletonize(pred_4d, num_iters)
    skel_target = soft_skeletonize(target_4d, num_iters)

    # Topology precision: how much of predicted skeleton lies within GT volume
    tprec = (
        (skel_pred * target_4d).sum() + smooth
    ) / (skel_pred.sum() + smooth)

    # Topology sensitivity: how much of GT skeleton is covered by prediction
    tsens = (
        (skel_target * pred_4d).sum() + smooth
    ) / (skel_target.sum() + smooth)

    cl_dice = 2.0 * tprec * tsens / (tprec + tsens + smooth)
    return 1.0 - cl_dice


def interface_cldice_loss(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    gt_a: torch.Tensor,
    gt_b: torch.Tensor,
    skel_iters: int = 10,
    dilation_iters: int = 2,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft-clDice on the interface between Mask A and Mask B.

    Extracts the predicted and GT interface regions, then applies
    clDice to enforce a continuous, thin boundary line.

    Args:
        pred_a: (B, H, W) soft prediction for mask A (after sigmoid)
        pred_b: (B, H, W) soft prediction for mask B (after sigmoid)
        gt_a: (B, H, W) binary GT for mask A
        gt_b: (B, H, W) binary GT for mask B
        skel_iters: skeleton iterations
        dilation_iters: boundary dilation before intersection
        smooth: Laplace smoothing

    Returns:
        Scalar loss.
    """
    pred_interface = extract_interface(pred_a, pred_b, dilation_iters)
    gt_interface = extract_interface(gt_a, gt_b, dilation_iters)

    return soft_cldice_loss(pred_interface, gt_interface, skel_iters, smooth)


def exclusivity_loss(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
) -> torch.Tensor:
    """
    Mutual exclusivity loss: penalizes overlap between Mask A and Mask B.

    L = mean(sigmoid(a) * sigmoid(b))

    Drives predictions toward complementary masks within the active region.

    Args:
        pred_a: (B, H, W) soft prediction A (after sigmoid)
        pred_b: (B, H, W) soft prediction B (after sigmoid)

    Returns:
        Scalar loss.
    """
    return (pred_a * pred_b).mean()
