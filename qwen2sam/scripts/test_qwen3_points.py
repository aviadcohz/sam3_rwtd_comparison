"""
Qwen3 text + CLIPSeg/CLIP-Grid point extraction debug tester.

Architecture: Qwen3 generates TEXT descriptions only (no coordinates).
Points are extracted via two text-driven methods:
  1. CLIPSeg: heatmap → threshold → erode → sample
  2. CLIP Grid: patch grid → cosine similarity → top patches → centers

Usage (VS Code "Run Python File" compatible):
  python qwen2sam/scripts/test_qwen3_points.py
  python qwen2sam/scripts/test_qwen3_points.py --samples 1,2,3,16,101
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent  # sam3_rwtd_comparison/
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_parent_root = _project_root.parent  # parent directory
if str(_parent_root) not in sys.path:
    sys.path.insert(0, str(_parent_root))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import adjusted_rand_score

from qwen2sam.models.qwen2sam_v3_tracker import Qwen2SAMv3Tracker
from qwen2sam.training.train_phase1 import load_config, set_seed
from qwen2sam.data.dataset_v2 import preprocess_image_for_sam3


# ===================================================================== #
#  PROMPT — text-only, no coordinates                                     #
# ===================================================================== #

QWEN_SYSTEM_PROMPT = (
    "You analyze surface textures in images. Always respond in the exact "
    "format requested, with no extra text."
)


QWEN_USER_PROMPT = (
    "This image contains exactly TWO main visually distinct regions separated by a boundary "
    "(for example, a prominent foreground object and its background, or two contrasting materials).\n\n"
    "Write a single, highly descriptive phrase (approximately 10-15 words) for each of the two regions. "
    "Include the following precise information:\n"
    "1. Semantic Name: A natural, common name for the material or object.\n"
    "2. Distinct Visual Features: The core visual attributes like color, pattern, or texture "
    "that strongly contrast with the other region.\n"
    "3. Spatial Context: A brief note on its general position (e.g., 'foreground', 'background', 'top-left').\n\n"
    "IMPORTANT: Describe the ENTIRE region as a collective group, NOT individual objects within it. "
    "Think of each region as a surface/area, not as a single object.\n\n"
    "Format your response exactly like this:\n"
    "TEXTURE_A: Texture of<description>\n"
    "TEXTURE_B: Texture of <description>"
)

DEFAULT_SAMPLES = ["1", "16", "101", "102", "13", "219", "303", "322", "380", "450", "76"]


# ===================================================================== #
#  Metrics                                                                #
# ===================================================================== #

def compute_iou(pred, gt):
    pred_b, gt_b = pred > 0.5, gt > 0.5
    inter = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    return 1.0 if union == 0 else float(inter / union)


def compute_dice(pred, gt):
    pred_b, gt_b = pred > 0.5, gt > 0.5
    inter = (pred_b & gt_b).sum()
    total = pred_b.sum() + gt_b.sum()
    return 1.0 if total == 0 else float(2.0 * inter / total)


def compute_ari(pred_a, pred_b, gt_a, gt_b):
    pred_labels = np.zeros(pred_a.shape, dtype=np.int32)
    pred_labels[pred_a > 0.5] = 1
    pred_labels[pred_b > 0.5] = 2
    gt_labels = np.zeros(gt_a.shape, dtype=np.int32)
    gt_labels[gt_a > 0.5] = 1
    gt_labels[gt_b > 0.5] = 2
    return float(adjusted_rand_score(gt_labels.ravel(), pred_labels.ravel()))


def compute_sample_metrics(pred_a, pred_b, gt_a, gt_b, crop_name):
    iou_direct = (compute_iou(pred_a, gt_a) + compute_iou(pred_b, gt_b)) / 2.0
    iou_swapped = (compute_iou(pred_a, gt_b) + compute_iou(pred_b, gt_a)) / 2.0
    if iou_swapped > iou_direct:
        pred_a, pred_b = pred_b, pred_a
    iou_a = compute_iou(pred_a, gt_a)
    iou_b = compute_iou(pred_b, gt_b)
    dice_a = compute_dice(pred_a, gt_a)
    dice_b = compute_dice(pred_b, gt_b)
    ari = compute_ari(pred_a, pred_b, gt_a, gt_b)
    return {
        "crop_name": crop_name,
        "iou_a": iou_a, "iou_b": iou_b,
        "mean_iou": (iou_a + iou_b) / 2.0,
        "dice_a": dice_a, "dice_b": dice_b,
        "mean_dice": (dice_a + dice_b) / 2.0,
        "ari": ari,
    }


def aggregate_metrics(all_metrics, tag):
    return {
        "tag": tag,
        "num_samples": len(all_metrics),
        "mean_iou": float(np.mean([m["mean_iou"] for m in all_metrics])),
        "mean_iou_a": float(np.mean([m["iou_a"] for m in all_metrics])),
        "mean_iou_b": float(np.mean([m["iou_b"] for m in all_metrics])),
        "mean_dice": float(np.mean([m["mean_dice"] for m in all_metrics])),
        "mean_ari": float(np.nanmean([m["ari"] for m in all_metrics])),
    }


# ===================================================================== #
#  Visualization helpers                                                  #
# ===================================================================== #

COLOR_A = (0, 0, 220)
COLOR_B = (220, 80, 0)
COLOR_BOUNDARY = (0, 255, 255)


def mask_overlay(image, mask_a, mask_b, alpha=0.45):
    vis = image.copy()
    overlay = image.copy()
    overlay[mask_a > 0.5] = COLOR_A
    overlay[mask_b > 0.5] = COLOR_B
    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)


def binary_mask_image(mask_a, mask_b, h, w):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[mask_a > 0.5] = COLOR_A
    canvas[mask_b > 0.5] = COLOR_B
    return canvas


def boundary_image(mask_a, mask_b, h, w):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ma = (mask_a > 0.5).astype(np.uint8) * 255
    mb = (mask_b > 0.5).astype(np.uint8) * 255
    bd_a = ma - cv2.erode(ma, kernel, iterations=1)
    bd_b = mb - cv2.erode(mb, kernel, iterations=1)
    da = cv2.dilate(bd_a, kernel, iterations=2)
    db = cv2.dilate(bd_b, kernel, iterations=2)
    interface = ((da > 0) & (db > 0)).astype(np.uint8) * 255
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[bd_a > 0] = (0, 0, 150)
    canvas[bd_b > 0] = (150, 60, 0)
    canvas[interface > 0] = COLOR_BOUNDARY
    return canvas


def heatmap_to_bgr(heatmap, size=None):
    """Convert float heatmap [0,1] to colored BGR image."""
    h = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    if size:
        colored = cv2.resize(colored, size, interpolation=cv2.INTER_LINEAR)
    return colored


# ===================================================================== #
#  SAM3 prediction helpers                                                #
# ===================================================================== #

@torch.no_grad()
def run_detr_with_text(model, backbone_out, text_feat, B, device):
    prompt_bf = text_feat["prompt"].unsqueeze(0).expand(B, -1, -1)
    mask_bf = text_feat["mask"].unsqueeze(0).expand(B, -1)
    out = model.base._run_sam3_from_backbone(backbone_out, prompt_bf, mask_bf)
    scores = out["pred_logits"][0].squeeze(-1).sigmoid()
    best = scores.argmax().item()
    return out["pred_masks"][0, best], out["pred_masks"]


@torch.no_grad()
def run_tracker_refinement(model, trunk_output, coarse_mask_logit,
                           pos_points_abs, neg_points_abs, device):
    image_embed, high_res_feats = model._get_sam2_features(trunk_output)
    coarse = coarse_mask_logit.unsqueeze(0).unsqueeze(0)
    refined = model._refine_one(
        image_embed, high_res_feats, coarse,
        pos_coords=pos_points_abs,
        neg_coords=neg_points_abs,
    )
    return refined


def postprocess_mask(mask_logit, gt_h, gt_w):
    if mask_logit.shape[-2:] != (gt_h, gt_w):
        mask_logit = F.interpolate(
            mask_logit[None, None].float() if mask_logit.ndim == 2
            else mask_logit.unsqueeze(0).float() if mask_logit.ndim == 3
            else mask_logit.float(),
            size=(gt_h, gt_w),
            mode="bilinear", align_corners=False,
        ).squeeze()
    return (mask_logit.sigmoid().cpu().numpy() > 0.5).astype(np.float32)


def scale_points(points_list, orig_size, target_size):
    scale = target_size / orig_size
    return [[p[0] * scale, p[1] * scale] for p in points_list]


# ===================================================================== #
#  Model loaders                                                          #
# ===================================================================== #

def load_qwen3_model(device):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor as AP
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"Loading {model_name}...")
    qwen3 = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    proc = AP.from_pretrained(model_name)
    print(f"  Qwen3-VL-8B loaded on {device}")
    return qwen3, proc


def load_clipseg_model(device):
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    print("Loading CLIPSeg...")
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined"
    ).to(device).eval()
    print("  CLIPSeg loaded")
    return model, proc


def load_clip_model(device):
    from transformers import CLIPModel, CLIPProcessor
    # Use laion CLIP which has safetensors (avoids torch.load vulnerability block)
    model_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    print(f"Loading CLIP ({model_name})...")
    try:
        proc = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
    except Exception:
        # Fallback: openai clip with safetensors-only loading
        model_name = "openai/clip-vit-base-patch16"
        print(f"  Fallback to {model_name}...")
        proc = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
    print(f"  CLIP loaded")
    return model, proc


# ===================================================================== #
#  Qwen3 text generation (no coordinates)                                 #
# ===================================================================== #

@torch.no_grad()
def qwen3_generate(model, processor, image_pil, device):
    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": QWEN_USER_PROMPT},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image_pil], return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs, max_new_tokens=200, do_sample=False, temperature=1.0)
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, input_len:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True)


def parse_text_output(text):
    """Parse text-only output: just TEXTURE_A and TEXTURE_B descriptions."""
    desc_a = desc_b = ""
    match_a = re.search(r'TEXTURE_A:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    match_b = re.search(r'TEXTURE_B:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match_a:
        desc_a = match_a.group(1).strip()
    if match_b:
        desc_b = match_b.group(1).strip()
    ok = bool(desc_a and desc_b)
    return desc_a, desc_b, ok


# ===================================================================== #
#  CLIPSeg: heatmap → threshold → erode → sample points                   #
# ===================================================================== #

@torch.no_grad()
def clipseg_extract_points(clipseg_model, clipseg_proc, image_pil,
                           desc_a, desc_b, img_size=256, device="cuda",
                           n_points=4, erode_iter=3, top_percentile=0.3):
    """
    Use CLIPSeg to generate heatmaps for each texture description,
    subtract opposing heatmap to get exclusive regions,
    threshold, erode to get safe interior, sample spread-apart points.
    Returns: pts_a, pts_b (lists of [x,y] in pixel space), heatmap_a, heatmap_b
    """
    # Generate both heatmaps first
    raw_heatmaps = []
    for desc in [desc_a, desc_b]:
        inputs = clipseg_proc(
            text=[desc], images=[image_pil],
            return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.amp.autocast("cuda", enabled=False):
            outputs = clipseg_model.float()(**inputs)

        logits = outputs.logits.squeeze().float()
        heatmap = torch.sigmoid(logits)
        heatmap = F.interpolate(
            heatmap[None, None], size=(img_size, img_size),
            mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
        raw_heatmaps.append(heatmap)

    # Subtract opposing heatmap: keep only pixels where THIS texture dominates
    diff_maps = [
        np.clip(raw_heatmaps[0] - raw_heatmaps[1], 0, 1),  # A minus B
        np.clip(raw_heatmaps[1] - raw_heatmaps[0], 0, 1),  # B minus A
    ]

    results = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for diff_map in diff_maps:
        # Threshold: keep top percentile of the difference map
        thresh = np.percentile(diff_map, (1.0 - top_percentile) * 100)
        binary = (diff_map >= thresh).astype(np.uint8)

        # Morphological erosion — shrink inward to avoid boundaries
        eroded = cv2.erode(binary, kernel, iterations=erode_iter)

        # Adaptive fallback if erosion killed too much
        for fallback_iter in [erode_iter - 1, 1, 0]:
            if eroded.sum() >= n_points:
                break
            eroded = cv2.erode(binary, kernel, iterations=max(fallback_iter, 0))
        if eroded.sum() < n_points:
            eroded = binary

        # Sample points weighted by difference confidence
        ys, xs = np.where(eroded > 0)
        if len(ys) < n_points:
            ys, xs = np.where(binary > 0)
        if len(ys) < n_points:
            ys, xs = np.where(diff_map > np.median(diff_map))

        weights = diff_map[ys, xs]
        weights = weights / (weights.sum() + 1e-8)

        # Sample spread-apart points
        pts = []
        available = list(range(len(ys)))
        for _ in range(min(n_points, len(available))):
            if not available:
                break
            if not pts:
                idx = available[np.argmax(weights[available])]
            else:
                scores = np.zeros(len(available))
                for ai, av in enumerate(available):
                    w = weights[av]
                    min_dist = min(
                        np.sqrt((xs[av] - p[0])**2 + (ys[av] - p[1])**2)
                        for p in pts
                    )
                    scores[ai] = w * min(min_dist / img_size, 1.0)
                idx = available[np.argmax(scores)]
            pts.append([int(xs[idx]), int(ys[idx])])
            available.remove(idx)

        results.append(pts)

    return results[0], results[1], raw_heatmaps[0], raw_heatmaps[1], diff_maps[0], diff_maps[1]


# ===================================================================== #
#  CLIP Grid: patches → cosine similarity → top patches → centers         #
# ===================================================================== #

@torch.no_grad()
def clip_grid_extract_points(clip_model, clip_proc, image_pil,
                             desc_a, desc_b, img_size=256, device="cuda",
                             grid_size=8, n_points=4):
    """
    Divide image into grid patches, compute CLIP cosine similarity
    with each description, use diff (A-B, B-A) for mutual exclusion,
    select top patches avoiding boundaries.
    Returns: pts_a, pts_b, sim_map_a, sim_map_b
    """
    image_np = np.array(image_pil.resize((img_size, img_size)))
    patch_h = img_size // grid_size
    patch_w = img_size // grid_size

    patches = []
    patch_coords = []
    for row in range(grid_size):
        for col in range(grid_size):
            y1, x1 = row * patch_h, col * patch_w
            y2, x2 = y1 + patch_h, x1 + patch_w
            patch = Image.fromarray(image_np[y1:y2, x1:x2])
            patches.append(patch)
            patch_coords.append((x1 + patch_w // 2, y1 + patch_h // 2))

    # Encode all patches
    patch_inputs = clip_proc(images=patches, return_tensors="pt", padding=True)
    patch_inputs = {k: v.to(device) for k, v in patch_inputs.items()}
    with torch.amp.autocast("cuda", enabled=False):
        patch_feats = clip_model.float().get_image_features(**patch_inputs)
    if not isinstance(patch_feats, torch.Tensor):
        patch_feats = patch_feats.pooler_output if hasattr(patch_feats, 'pooler_output') else patch_feats[0]
    patch_feats = patch_feats.float()
    patch_feats = patch_feats / patch_feats.norm(dim=-1, keepdim=True)

    # Compute similarity for both descriptions
    all_sims = []
    sim_maps = []
    for desc in [desc_a, desc_b]:
        text_inputs = clip_proc(text=[desc], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.amp.autocast("cuda", enabled=False):
            text_feat = clip_model.float().get_text_features(**text_inputs)
        if not isinstance(text_feat, torch.Tensor):
            text_feat = text_feat.pooler_output if hasattr(text_feat, 'pooler_output') else text_feat[0]
        text_feat = text_feat.float()
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        sims = (patch_feats @ text_feat.T).squeeze().float().cpu().numpy()
        all_sims.append(sims)
        sim_maps.append(sims.reshape(grid_size, grid_size))

    # Mutual exclusion: use difference scores
    diff_sims = [
        np.clip(all_sims[0] - all_sims[1], 0, None),  # A dominance
        np.clip(all_sims[1] - all_sims[0], 0, None),  # B dominance
    ]

    results = []
    for diff in diff_sims:
        # Filter boundary patches
        valid = np.ones(len(diff), dtype=bool)
        for idx in range(len(diff)):
            row, col = idx // grid_size, idx % grid_size
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    neighbors.append(diff[nr * grid_size + nc])
            if neighbors:
                # Skip if this patch has high diff but a neighbor is near-zero
                if diff[idx] > np.median(diff) and min(neighbors) < 1e-4:
                    valid[idx] = False

        masked = diff.copy()
        masked[~valid] = -1

        pts = []
        for _ in range(n_points):
            best = np.argmax(masked)
            if masked[best] <= -1:
                best = np.argmax(diff)
            cx, cy = patch_coords[best]
            pts.append([cx, cy])
            masked[best] = -1

        results.append(pts)

    return results[0], results[1], sim_maps[0], sim_maps[1]


# ===================================================================== #
#  Point accuracy check                                                   #
# ===================================================================== #

def check_point_in_mask(pts, mask, img_size=256):
    results = []
    for pt in pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        x = max(0, min(img_size - 1, x))
        y = max(0, min(img_size - 1, y))
        results.append(bool(mask[y, x] > 0.5))
    return results


# ===================================================================== #
#  Visualization                                                          #
# ===================================================================== #

def draw_point(img, pt, color, marker="circle", label="", scale=1.0):
    cx, cy = int(pt[0] * scale), int(pt[1] * scale)
    if marker == "circle":
        cv2.circle(img, (cx, cy), 8, color, -1)
        cv2.circle(img, (cx, cy), 8, (255, 255, 255), 2)
    elif marker == "diamond":
        d = 8
        diamond = np.array([[cx, cy-d], [cx+d, cy], [cx, cy+d], [cx-d, cy]], np.int32)
        cv2.fillPoly(img, [diamond], color)
        cv2.polylines(img, [diamond], True, (255, 255, 255), 2)
    elif marker == "square":
        cv2.rectangle(img, (cx-6, cy-6), (cx+6, cy+6), color, -1)
        cv2.rectangle(img, (cx-6, cy-6), (cx+6, cy+6), (255, 255, 255), 2)
    elif marker == "triangle":
        d = 8
        tri = np.array([[cx, cy-d], [cx+d, cy+d], [cx-d, cy+d]], np.int32)
        cv2.fillPoly(img, [tri], color)
        cv2.polylines(img, [tri], True, (255, 255, 255), 2)
    if label:
        cv2.putText(img, label, (cx + 10, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def draw_debug_grid(image_bgr, gt_a, gt_b, preds, oracle_pts,
                    clipseg_pts, clipgrid_pts, heatmaps, diff_maps,
                    q_desc_a, q_desc_b, title, cell_size=256,
                    coarse_masks=None, detr_masks=None):
    """
    Visualization grid:
      Row 1: overlay (pred on image)
      Row 2: binary masks
      Row 3: coarse mask that was FED to tracker for this method

    Columns: Image+Pts | Diff | GT | Qw3Txt(+DETR) | +CSeg(+DETR) | +Logit(+logit) | +Binary(+binary) | OrcT+P
    """
    h, w = image_bgr.shape[:2]
    s = cell_size / max(h, w)
    ch, cw = int(h * s), int(w * s)

    PT_COLOR_A = (0, 0, 255)
    PT_COLOR_B = (255, 150, 0)

    def ri(img):
        return cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)

    def rm(mask):
        return cv2.resize(mask.astype(np.float32), (cw, ch),
                          interpolation=cv2.INTER_NEAREST)

    img = ri(image_bgr)
    ga, gb = rm(gt_a), rm(gt_b)

    # Image with CLIPSeg points
    img_pts = img.copy()
    cseg_a_pts, cseg_b_pts = clipseg_pts
    for pt in cseg_a_pts:
        draw_point(img_pts, pt, PT_COLOR_A, "circle", "", s)
    for pt in cseg_b_pts:
        draw_point(img_pts, pt, PT_COLOR_B, "circle", "", s)

    # Diff heatmaps
    diff_a, diff_b = diff_maps
    d_max = max(diff_a.max(), diff_b.max(), 1e-8) if diff_a is not None else 1
    diff_a_vis = heatmap_to_bgr(diff_a / d_max, (cw, ch)) if diff_a is not None else np.zeros((ch, cw, 3), dtype=np.uint8)
    diff_b_vis = heatmap_to_bgr(diff_b / d_max, (cw, ch)) if diff_b is not None else np.zeros((ch, cw, 3), dtype=np.uint8)

    # Prepare coarse mask visualizations for each method
    # DETR mask (used by Qw3Txt and +CSeg)
    if detr_masks is not None:
        dm_a, dm_b = detr_masks
        dm_a_np = torch.sigmoid(dm_a.float()).cpu().numpy() if isinstance(dm_a, torch.Tensor) else dm_a
        dm_b_np = torch.sigmoid(dm_b.float()).cpu().numpy() if isinstance(dm_b, torch.Tensor) else dm_b
        detr_a_vis = heatmap_to_bgr(dm_a_np, (cw, ch))
        detr_b_vis = heatmap_to_bgr(dm_b_np, (cw, ch))
    else:
        detr_a_vis = np.zeros((ch, cw, 3), dtype=np.uint8)
        detr_b_vis = np.zeros((ch, cw, 3), dtype=np.uint8)

    # CLIPSeg binary mask (used by +Binary and +GenCSeg)
    if coarse_masks is not None:
        cm_a, cm_b = coarse_masks
        cm_a_vis = heatmap_to_bgr((cm_a + 5.0) / 10.0, (cw, ch))
        cm_b_vis = heatmap_to_bgr((cm_b + 5.0) / 10.0, (cw, ch))
    else:
        cm_a_vis = np.zeros((ch, cw, 3), dtype=np.uint8)
        cm_b_vis = np.zeros((ch, cw, 3), dtype=np.uint8)

    # Map each method to its coarse mask visualization
    # Row 3 shows: texture_a coarse mask on top, texture_b on bottom (but we only have 3 rows)
    # So row 3 = coarse mask for texture A (the primary one to debug)
    coarse_map = {
        "Qw3Txt": detr_a_vis,     # DETR mask (what DETR produced)
        "+CSeg": detr_a_vis,       # uses DETR mask as coarse
        "+GenCSeg": cm_a_vis,      # uses CLIPSeg binary mask (generic "texture" prompt)
        "+Binary": cm_a_vis,       # uses CLIPSeg binary mask (Qwen text prompt)
        "OrcT+P": detr_a_vis,      # uses oracle DETR mask
    }

    # Build columns
    # Col: Image+Pts
    col_img = (img_pts, np.zeros((ch, cw, 3), dtype=np.uint8),
               np.zeros((ch, cw, 3), dtype=np.uint8))
    # Col: Diff heatmap
    col_diff = (diff_a_vis, diff_b_vis,
                np.zeros((ch, cw, 3), dtype=np.uint8))
    # Col: GT
    gt_ov = mask_overlay(img, ga, gb)
    gt_bin = binary_mask_image(ga, gb, ch, cw)
    col_gt = (gt_ov, gt_bin, np.zeros((ch, cw, 3), dtype=np.uint8))

    cols = [col_img, col_diff, col_gt]
    col_labels = [f"{title}", "Diff A|B", "GT"]

    for label, (ma, mb, met) in preds.items():
        ma_r, mb_r = rm(ma), rm(mb)
        ov = mask_overlay(img, ma_r, mb_r)
        bn = binary_mask_image(ma_r, mb_r, ch, cw)
        # Row 3: the coarse mask that was fed to the tracker for this method
        coarse_vis = coarse_map.get(label, np.zeros((ch, cw, 3), dtype=np.uint8))
        cols.append((ov, bn, coarse_vis))
        col_labels.append(f"{label} {met['mean_iou']:.3f}")

    # Assemble grid
    sep = 2
    header_h = 32
    desc_h = 40
    row_labels_h = 16

    rows = []
    for row_idx in range(3):
        cells = []
        for col in cols:
            cells.extend([col[row_idx], np.zeros((ch, sep, 3), dtype=np.uint8)])
        rows.append(np.hstack(cells[:-1]))

    actual_w = rows[0].shape[1]

    # Header
    bar = np.zeros((header_h, actual_w, 3), dtype=np.uint8) + 30
    x = 0
    for lbl in col_labels:
        cv2.putText(bar, lbl, (x + 4, header_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        x += cw + sep

    # Description bar
    desc_bar = np.zeros((desc_h, actual_w, 3), dtype=np.uint8) + 20
    cv2.putText(desc_bar, f"A: {q_desc_a}"[:actual_w // 5], (8, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, PT_COLOR_A, 1, cv2.LINE_AA)
    cv2.putText(desc_bar, f"B: {q_desc_b}"[:actual_w // 5], (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, PT_COLOR_B, 1, cv2.LINE_AA)

    # Row labels bar
    row_label_bar = np.zeros((row_labels_h, actual_w, 3), dtype=np.uint8) + 10
    cv2.putText(row_label_bar, "Row1:Overlay  Row2:Binary  Row3:Coarse mask fed to tracker (texture A)", (8, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (150, 150, 150), 1, cv2.LINE_AA)

    row_sep = np.zeros((sep, actual_w, 3), dtype=np.uint8)
    grid_rows = []
    for r in rows:
        grid_rows.extend([r, row_sep])

    grid = np.vstack([bar, desc_bar, row_label_bar] + grid_rows[:-1])
    return grid


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Qwen3 + CLIPSeg/CLIP-Grid point tester")
    parser.add_argument("--config", type=str,
                        default="qwen2sam/configs/v3_tracker_detexure.yaml")
    parser.add_argument("--data_root", type=str, default="/home/aviad/RWTD")
    parser.add_argument("--samples", type=str, default=None,
                        help="Comma-separated crop names (default: 10 diverse)")
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/qwen3_points_debug")
    args = parser.parse_args()

    sample_names = args.samples.split(",") if args.samples else DEFAULT_SAMPLES
    # None means run all images

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _project_root / config_path
    cfg = load_config(str(config_path))
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg["model"].get("image_size", 1008)

    # Load metadata
    meta_path = Path(args.data_root) / "metadata_phase1.json"
    with open(meta_path) as f:
        all_meta = json.load(f)
    meta_by_name = {e["crop_name"]: e for e in all_meta}
    if sample_names is None:
        samples = all_meta  # run all images
    else:
        samples = [meta_by_name[n] for n in sample_names if n in meta_by_name]
    print(f"Testing {len(samples)} samples")

    # Output dirs
    output_dir = _project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    if "tracker" in cfg:
        cfg["tracker"].pop("v3_checkpoint", None)
    print("Loading SAM3 model...")
    model = Qwen2SAMv3Tracker(cfg, device=str(device))
    model.base.sam3.eval()
    model.sam_prompt_encoder.eval()
    model.sam_mask_decoder.eval()

    qwen3, qwen3_proc = load_qwen3_model(device)
    clipseg_model, clipseg_proc = load_clipseg_model(device)
    # clip_model, clip_proc = load_clip_model(device)  # Removed: CLIPGrid not used

    # Metrics accumulators
    txt_metrics = []
    cseg_metrics = []
    cfull_metrics = []
    cbin_metrics = []
    orc_tp_metrics = []

    # Point accuracy
    cseg_total, cseg_correct = 0, 0
    outputs_log = []
    trunk_cache = {}

    def _trunk_hook(module, input, output):
        trunk_cache["xs"] = output

    t0 = time.time()

    for i, entry in enumerate(samples):
        crop_name = entry["crop_name"]
        orc_pts_a = entry.get("oracle_points", {}).get("point_prompt_mask_a", [])
        orc_pts_b = entry.get("oracle_points", {}).get("point_prompt_mask_b", [])

        image_bgr = cv2.imread(entry["image_path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        gt_a = cv2.imread(entry["mask_a_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_b = cv2.imread(entry["mask_b_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_h, gt_w = gt_a.shape

        image_pil = Image.fromarray(image_rgb)
        sam_img = preprocess_image_for_sam3(image_rgb, image_size).unsqueeze(0).to(device)

        trunk = model.base.sam3.backbone.vision_backbone.trunk
        hook = trunk.register_forward_hook(_trunk_hook)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = model.base.sam3.backbone.forward_image(sam_img)
            backbone_out["img_batch_all_stages"] = sam_img

            # --- Qwen3 text generation ---
            raw = qwen3_generate(qwen3, qwen3_proc, image_pil, device)
            q_desc_a, q_desc_b, ok = parse_text_output(raw)

            if not ok:
                print(f"\n  [{crop_name}] PARSE FAIL: {raw[:100]}")
                hook.remove()
                zero = np.zeros_like(gt_a)
                for ml in [txt_metrics, cseg_metrics, cfull_metrics, cbin_metrics, orc_tp_metrics]:
                    ml.append(compute_sample_metrics(zero, zero, gt_a, gt_b, crop_name))
                continue

            # --- SAM3 DETR with text ---
            t_a_out = model.base.sam3.backbone.forward_text([q_desc_a], device=device)
            t_b_out = model.base.sam3.backbone.forward_text([q_desc_b], device=device)
            feat_a = {"prompt": t_a_out["language_features"].squeeze(1),
                      "mask": t_a_out["language_mask"].squeeze(0)}
            feat_b = {"prompt": t_b_out["language_features"].squeeze(1),
                      "mask": t_b_out["language_mask"].squeeze(0)}
            a_logit, _ = run_detr_with_text(model, backbone_out, feat_a, 1, device)
            b_logit, _ = run_detr_with_text(model, backbone_out, feat_b, 1, device)

            txt_a = postprocess_mask(a_logit, gt_h, gt_w)
            txt_b = postprocess_mask(b_logit, gt_h, gt_w)

            hook.remove()
            trunk_output = trunk_cache["xs"][-1]

            # --- CLIPSeg point extraction ---
            cseg_pts_a, cseg_pts_b, heat_a, heat_b, diff_a, diff_b = clipseg_extract_points(
                clipseg_model, clipseg_proc, image_pil,
                q_desc_a, q_desc_b, img_size=orig_w, device=device)

            cseg_pts_a_s = scale_points(cseg_pts_a, orig_w, image_size)
            cseg_pts_b_s = scale_points(cseg_pts_b, orig_w, image_size)
            pos_a = torch.tensor(cseg_pts_a_s, dtype=torch.float32, device=device).unsqueeze(0)
            neg_a = torch.tensor(cseg_pts_b_s, dtype=torch.float32, device=device).unsqueeze(0)
            ref_a = run_tracker_refinement(model, trunk_output, a_logit, pos_a, neg_a, device)
            ref_b = run_tracker_refinement(model, trunk_output, b_logit, neg_a, pos_a, device)
            cseg_a = postprocess_mask(ref_a.squeeze(), gt_h, gt_w)
            cseg_b = postprocess_mask(ref_b.squeeze(), gt_h, gt_w)

            # --- Gen+CSeg: Generic "texture" DETR + CLIPSeg binary mask + points ---
            # SAM3 DETR with generic prompt, then tracker with CLIPSeg mask
            gen_text_out = model.base.sam3.backbone.forward_text(["texture"], device=device)
            gen_feat = {"prompt": gen_text_out["language_features"].squeeze(1),
                        "mask": gen_text_out["language_mask"].squeeze(0)}
            gen_logit, _ = run_detr_with_text(model, backbone_out, gen_feat, 1, device)
            # Use CLIPSeg binary mask as coarse + CLIPSeg points
            gen_coarse_a = (diff_a > 0.1).astype(np.float32) * 10.0 - 5.0
            gen_coarse_b = (diff_b > 0.1).astype(np.float32) * 10.0 - 5.0
            gen_mask_a = torch.from_numpy(gen_coarse_a).float().to(device)
            gen_mask_b = torch.from_numpy(gen_coarse_b).float().to(device)
            ref_gen_a = run_tracker_refinement(
                model, trunk_output, gen_mask_a, pos_a, neg_a, device)
            ref_gen_b = run_tracker_refinement(
                model, trunk_output, gen_mask_b, neg_a, pos_a, device)
            cfull_a = postprocess_mask(ref_gen_a.squeeze(), gt_h, gt_w)
            cfull_b = postprocess_mask(ref_gen_b.squeeze(), gt_h, gt_w)

            # --- CSeg Binary: threshold diff → binary logit mask + points ---
            coarse_np_a = (diff_a > 0.1).astype(np.float32) * 10.0 - 5.0
            coarse_np_b = (diff_b > 0.1).astype(np.float32) * 10.0 - 5.0
            binary_a = torch.from_numpy(coarse_np_a).float().to(device)
            binary_b = torch.from_numpy(coarse_np_b).float().to(device)
            ref_bin_a = run_tracker_refinement(
                model, trunk_output, binary_a, pos_a, neg_a, device)
            ref_bin_b = run_tracker_refinement(
                model, trunk_output, binary_b, neg_a, pos_a, device)
            cbin_a = postprocess_mask(ref_bin_a.squeeze(), gt_h, gt_w)
            cbin_b = postprocess_mask(ref_bin_b.squeeze(), gt_h, gt_w)

            # --- Oracle Text + Points (reference) ---
            if orc_pts_a and orc_pts_b:
                orc_pts_a_s = scale_points(orc_pts_a, orig_w, image_size)
                orc_pts_b_s = scale_points(orc_pts_b, orig_w, image_size)
                pos_oa = torch.tensor(orc_pts_a_s, dtype=torch.float32, device=device).unsqueeze(0)
                neg_oa = torch.tensor(orc_pts_b_s, dtype=torch.float32, device=device).unsqueeze(0)
                # Use oracle text for oracle T+P
                orc_t_a = model.base.sam3.backbone.forward_text([entry["texture_a"]], device=device)
                orc_t_b = model.base.sam3.backbone.forward_text([entry["texture_b"]], device=device)
                orc_feat_a = {"prompt": orc_t_a["language_features"].squeeze(1),
                              "mask": orc_t_a["language_mask"].squeeze(0)}
                orc_feat_b = {"prompt": orc_t_b["language_features"].squeeze(1),
                              "mask": orc_t_b["language_mask"].squeeze(0)}
                orc_a_logit, _ = run_detr_with_text(model, backbone_out, orc_feat_a, 1, device)
                orc_b_logit, _ = run_detr_with_text(model, backbone_out, orc_feat_b, 1, device)
                ref_oa = run_tracker_refinement(model, trunk_output, orc_a_logit, pos_oa, neg_oa, device)
                ref_ob = run_tracker_refinement(model, trunk_output, orc_b_logit, neg_oa, pos_oa, device)
                orc_a_mask = postprocess_mask(ref_oa.squeeze(), gt_h, gt_w)
                orc_b_mask = postprocess_mask(ref_ob.squeeze(), gt_h, gt_w)
            else:
                orc_a_mask, orc_b_mask = txt_a, txt_b

        # Metrics
        txt_met = compute_sample_metrics(txt_a, txt_b, gt_a, gt_b, crop_name)
        cseg_met = compute_sample_metrics(cseg_a, cseg_b, gt_a, gt_b, crop_name)
        cfull_met = compute_sample_metrics(cfull_a, cfull_b, gt_a, gt_b, crop_name)
        cbin_met = compute_sample_metrics(cbin_a, cbin_b, gt_a, gt_b, crop_name)
        orc_met = compute_sample_metrics(orc_a_mask, orc_b_mask, gt_a, gt_b, crop_name)
        txt_metrics.append(txt_met)
        cseg_metrics.append(cseg_met)
        cfull_metrics.append(cfull_met)
        cbin_metrics.append(cbin_met)
        orc_tp_metrics.append(orc_met)

        # Point accuracy
        cseg_in_a = check_point_in_mask(cseg_pts_a, gt_a, orig_w)
        cseg_in_b = check_point_in_mask(cseg_pts_b, gt_b, orig_w)
        cseg_total += len(cseg_in_a) + len(cseg_in_b)
        cseg_correct += sum(cseg_in_a) + sum(cseg_in_b)

        outputs_log.append({
            "crop_name": crop_name, "raw": raw,
            "desc_a": q_desc_a, "desc_b": q_desc_b,
            "cseg_pts_a": cseg_pts_a, "cseg_pts_b": cseg_pts_b,
        })

        # Print
        print(f"\n  [{crop_name}]")
        print(f"    A: \"{q_desc_a}\"")
        print(f"    B: \"{q_desc_b}\"")
        print(f"    CLIPSeg pts_a={cseg_pts_a} in_mask={cseg_in_a}")
        print(f"    CLIPSeg pts_b={cseg_pts_b} in_mask={cseg_in_b}")
        print(f"    Txt: {txt_met['mean_iou']:.4f} | +CSeg: {cseg_met['mean_iou']:.4f} | "
              f"+GenCSeg: {cfull_met['mean_iou']:.4f} | +Binary: {cbin_met['mean_iou']:.4f} | "
              f"OrcT+P: {orc_met['mean_iou']:.4f}")

        # Visualization
        preds = {
            "Qw3Txt": (txt_a, txt_b, txt_met),
            "+CSeg": (cseg_a, cseg_b, cseg_met),
            "+GenCSeg": (cfull_a, cfull_b, cfull_met),
            "+Binary": (cbin_a, cbin_b, cbin_met),
            "OrcT+P": (orc_a_mask, orc_b_mask, orc_met),
        }
        grid = draw_debug_grid(
            image_bgr, gt_a, gt_b, preds,
            oracle_pts=(orc_pts_a, orc_pts_b),
            clipseg_pts=(cseg_pts_a, cseg_pts_b),
            clipgrid_pts=([], []),
            heatmaps=(heat_a, heat_b),
            diff_maps=(diff_a, diff_b),
            q_desc_a=q_desc_a, q_desc_b=q_desc_b,
            title=crop_name,
            coarse_masks=(coarse_np_a, coarse_np_b),
            detr_masks=(a_logit, b_logit),
        )
        cv2.imwrite(str(vis_dir / f"{crop_name}_debug.png"), grid)

    elapsed = time.time() - t0

    # Summary
    txt_sum = aggregate_metrics(txt_metrics, "qwen3_text")
    cseg_sum = aggregate_metrics(cseg_metrics, "qwen3_clipseg")
    cfull_sum = aggregate_metrics(cfull_metrics, "gen_cseg")
    cbin_sum = aggregate_metrics(cbin_metrics, "qwen3_cseg_binary")
    orc_sum = aggregate_metrics(orc_tp_metrics, "oracle_t+p")
    cseg_acc = 100 * cseg_correct / max(cseg_total, 1)

    print(f"\n{'='*100}")
    print(f"  Qwen3 + CLIPSeg Mask Debug — {len(samples)} samples ({elapsed:.1f}s)")
    print(f"{'='*100}")
    print(f"  {'':15s} {'Qw3Txt':>10s} {'  +CSeg':>10s} {'+GenCSeg':>10s} {'+Binary':>10s} {'OrcT+P':>10s}")
    print(f"  {'─'*80}")
    for label, key in [("mIoU", "mean_iou"), ("mDice", "mean_dice"), ("mARI", "mean_ari")]:
        print(f"  {label:15s} {txt_sum[key]:10.4f} {cseg_sum[key]:10.4f} "
              f"{cfull_sum[key]:10.4f} {cbin_sum[key]:10.4f} {orc_sum[key]:10.4f}")
    print(f"  {'─'*80}")
    print(f"  Point accuracy:  CLIPSeg {cseg_correct}/{cseg_total} ({cseg_acc:.1f}%)")
    print(f"{'='*100}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*100}")

    # Save
    with open(output_dir / "qwen3_outputs.json", "w") as f:
        json.dump(outputs_log, f, indent=2, default=str)
    summary = {
        "qwen3_text": txt_sum,
        "qwen3_clipseg": cseg_sum,
        "gen_cseg": cfull_sum,
        "qwen3_cseg_binary": cbin_sum,
        "oracle_t+p": orc_sum,
        "clipseg_point_accuracy": cseg_acc,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
