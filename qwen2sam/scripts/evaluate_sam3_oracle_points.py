"""
SAM3 Oracle Description + Points Evaluation.

9-way comparison on RWTD:
  1. Generic ZS:        SAM3 text="texture", top-2 queries (baseline)
  2. Oracle text:       SAM3 text=ground-truth descriptions, top-1 per texture
  3. Pts only:          Generic DETR masks → tracker with GT points
  4. Oracle text+pts:   Oracle DETR masks → tracker with GT points
  5. Qwen3 proposal:    Qwen3-VL-8B descriptions → SAM3 DETR top-1 proposal mask
  6. Qwen3 semseg:      Qwen3-VL-8B descriptions → SAM3 semantic seg head (WTA)
  7. Qwen3 text+pts:    Qwen3-VL-8B descriptions + Qwen3 points → tracker (legacy)
  8. Qwen3+CSeg:        Qwen3 text + CLIPSeg points + CLIPSeg binary mask → tracker
  9. Qwen3+CSeg(sem):   Qwen3 text + CLIPSeg points + Semantic Seg mask → tracker

Usage:
  python -m qwen2sam.scripts.evaluate_sam3_oracle_points \
      --config qwen2sam/configs/v3_tracker_detexure.yaml \
      --output_dir eval_results/sam3_oracle_points
"""

import argparse
import json
import re
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qwen2sam.models.qwen2sam_v3_tracker import Qwen2SAMv3Tracker
from qwen2sam.training.train_phase1 import load_config, set_seed

from qwen2sam.scripts.evaluate_v2 import (
    compute_sample_metrics, aggregate_metrics,
    save_metrics_csv, mask_overlay, binary_mask_image, boundary_image,
)
from qwen2sam.data.dataset_v2 import preprocess_image_for_sam3


# ===================================================================== #
#  Prediction helpers                                                      #
# ===================================================================== #

@torch.no_grad()
def get_sam3_backbone_and_text(model, sam_images, text_prompts, device):
    """
    Run SAM3 backbone + text encoder.
    Returns backbone_out and text features for each prompt.
    """
    model.base.sam3.eval()
    backbone_out = model.base.sam3.backbone.forward_image(sam_images)
    backbone_out["img_batch_all_stages"] = sam_images

    text_features = []
    for text in text_prompts:
        text_out = model.base.sam3.backbone.forward_text([text], device=device)
        text_features.append({
            "prompt": text_out["language_features"].squeeze(1),  # (seq_len, 256)
            "mask": text_out["language_mask"].squeeze(0),        # (seq_len,)
        })

    return backbone_out, text_features


@torch.no_grad()
def run_detr_with_text(model, backbone_out, text_feat, B, device):
    """Run DETR with text features, return coarse mask logits and best query mask."""
    prompt_bf = text_feat["prompt"].unsqueeze(0).expand(B, -1, -1)
    mask_bf = text_feat["mask"].unsqueeze(0).expand(B, -1)

    out = model.base._run_sam3_from_backbone(backbone_out, prompt_bf, mask_bf)

    scores = out["pred_logits"][0].squeeze(-1).sigmoid()
    best = scores.argmax().item()
    mask_logit = out["pred_masks"][0, best]

    return mask_logit, out["pred_masks"]


@torch.no_grad()
def run_detr_with_text_full(model, backbone_out, text_feat, B, device):
    """Run DETR with text features, return best proposal mask, all masks, AND semantic mask."""
    prompt_bf = text_feat["prompt"].unsqueeze(0).expand(B, -1, -1)
    mask_bf = text_feat["mask"].unsqueeze(0).expand(B, -1)

    out = model.base._run_sam3_from_backbone(backbone_out, prompt_bf, mask_bf)

    scores = out["pred_logits"][0].squeeze(-1).sigmoid()
    best = scores.argmax().item()
    mask_logit = out["pred_masks"][0, best]
    semantic_mask = out.get("semantic_mask", None)

    return mask_logit, out["pred_masks"], semantic_mask


@torch.no_grad()
def run_detr_generic_top2(model, backbone_out, B, device):
    """Run DETR with generic 'texture' prompt, return top-2 masks."""
    text_out = model.base.sam3.backbone.forward_text(["texture"], device=device)
    prompt = text_out["language_features"].squeeze(1)
    mask = text_out["language_mask"].squeeze(0)
    prompt_bf = prompt.unsqueeze(0).expand(B, -1, -1)
    mask_bf = mask.unsqueeze(0).expand(B, -1)

    out = model.base._run_sam3_from_backbone(backbone_out, prompt_bf, mask_bf)
    scores = out["pred_logits"][0].squeeze(-1).sigmoid()
    top2 = scores.topk(2).indices
    return out["pred_masks"][0, top2[0].item()], out["pred_masks"][0, top2[1].item()]


@torch.no_grad()
def run_tracker_refinement(model, trunk_output, coarse_mask_logit,
                           pos_points_abs, neg_points_abs, device):
    """
    Run SAM tracker refinement with points + coarse mask.

    Args:
        coarse_mask_logit: (H, W) raw logits from DETR
        pos_points_abs: (1, N, 2) absolute coords in 1008 space
        neg_points_abs: (1, N, 2) absolute coords in 1008 space
    """
    image_embed, high_res_feats = model._get_sam2_features(trunk_output)

    coarse = coarse_mask_logit.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    refined = model._refine_one(
        image_embed, high_res_feats, coarse,
        pos_coords=pos_points_abs,
        neg_coords=neg_points_abs,
    )
    return refined  # (1, 1, 288, 288) or similar


def postprocess_mask(mask_logit, gt_h, gt_w):
    """Resize and binarize a mask logit."""
    if mask_logit.shape[-2:] != (gt_h, gt_w):
        mask_logit = F.interpolate(
            mask_logit[None, None].float() if mask_logit.ndim == 2
            else mask_logit.unsqueeze(0).float() if mask_logit.ndim == 3
            else mask_logit.float(),
            size=(gt_h, gt_w),
            mode="bilinear", align_corners=False,
        ).squeeze()
    return (mask_logit.sigmoid().cpu().numpy() > 0.5).astype(np.float32)


def postprocess_semantic_mask(mask_logit, gt_h, gt_w):
    """Resize mask logit to probability map (no thresholding) for WTA comparison."""
    if mask_logit.shape[-2:] != (gt_h, gt_w):
        mask_logit = F.interpolate(
            mask_logit[None, None].float() if mask_logit.ndim == 2
            else mask_logit.unsqueeze(0).float(),
            size=(gt_h, gt_w),
            mode="bilinear", align_corners=False,
        ).squeeze()
    return mask_logit.sigmoid().float().cpu().numpy()


def _iou(pred, gt):
    """Compute IoU between two binary masks."""
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return float(inter / max(union, 1e-8))


def scale_points(points_list, orig_size, target_size):
    """Scale point coordinates from original image space to target space."""
    scale = target_size / orig_size
    return [[p[0] * scale, p[1] * scale] for p in points_list]


# ===================================================================== #
#  Qwen inference (zero-shot generation)                                   #
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




@torch.no_grad()
def qwen_generate(model, processor, image_pil, device):
    """
    Run Qwen VL (2.5 or 3) in generation mode on an image.
    Returns raw text output.
    """
    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QWEN_USER_PROMPT},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text], images=[image_pil], return_tensors="pt", padding=True,
    )
    # Qwen3-VL may produce token_type_ids that aren't needed
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        temperature=1.0,
    )

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, input_len:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True)


def load_qwen3_model(device):
    """Load Qwen3-VL-8B-Instruct as a separate model for comparison."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor as AP

    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"Loading {model_name} for comparison...")
    qwen3_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    qwen3_processor = AP.from_pretrained(model_name)
    print(f"  Qwen3-VL-8B loaded on {device}")
    return qwen3_model, qwen3_processor


def load_clipseg_model(device):
    """Load CLIPSeg for text-guided heatmap generation."""
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    print("Loading CLIPSeg...")
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined"
    ).to(device).eval()
    print("  CLIPSeg loaded")
    return model, proc


@torch.no_grad()
def clipseg_extract_points_and_mask(clipseg_model, clipseg_proc, image_pil,
                                     desc_a, desc_b, img_size=256, device="cuda",
                                     n_points=4, erode_iter=3, top_percentile=0.3):
    """
    Use CLIPSeg to generate heatmaps, compute diff maps for mutual exclusion,
    extract spread-apart points and binary masks for each texture.

    Returns: pts_a, pts_b, binary_mask_a, binary_mask_b
      pts: list of [x, y] in pixel space
      binary_mask: numpy array (img_size, img_size) with values +5/-5 (logit scale for SAM3)
    """
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

    # Mutual exclusion: diff maps
    diff_maps = [
        np.clip(raw_heatmaps[0] - raw_heatmaps[1], 0, 1),
        np.clip(raw_heatmaps[1] - raw_heatmaps[0], 0, 1),
    ]

    # Binary masks in logit scale for SAM3: +5 inside, -5 outside
    binary_masks = []
    for diff in diff_maps:
        binary = (diff > 0.1).astype(np.float32)
        binary_masks.append(binary * 10.0 - 5.0)  # logit scale

    # Extract points from diff maps
    results = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for diff_map in diff_maps:
        thresh = np.percentile(diff_map, (1.0 - top_percentile) * 100)
        binary = (diff_map >= thresh).astype(np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=erode_iter)
        for fallback_iter in [erode_iter - 1, 1, 0]:
            if eroded.sum() >= n_points:
                break
            eroded = cv2.erode(binary, kernel, iterations=max(fallback_iter, 0))
        if eroded.sum() < n_points:
            eroded = binary

        ys, xs = np.where(eroded > 0)
        if len(ys) < n_points:
            ys, xs = np.where(binary > 0)
        if len(ys) < n_points:
            ys, xs = np.where(diff_map > np.median(diff_map))

        weights = diff_map[ys, xs]
        weights = weights / (weights.sum() + 1e-8)

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

    return results[0], results[1], binary_masks[0], binary_masks[1]


def parse_qwen_text_only(text):
    """Parse text-only Qwen output (no points)."""
    desc_a = desc_b = ""
    match_a = re.search(r'TEXTURE_A:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    match_b = re.search(r'TEXTURE_B:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match_a:
        desc_a = match_a.group(1).strip()
    if match_b:
        desc_b = match_b.group(1).strip()
    return desc_a, desc_b, bool(desc_a and desc_b)


def parse_qwen_output(text, img_size=256):
    """
    Parse Qwen's structured output into descriptions and points.
    Handles both normalized (0.0-1.0) and pixel (integer) coordinates.
    Returns points in pixel space (img_size).

    Returns:
        desc_a, desc_b: str descriptions
        points_a, points_b: list of [x, y] pixel coordinates (validated)
        success: bool
    """
    desc_a = desc_b = ""
    points_a = []
    points_b = []

    # Try to extract TEXTURE_A/B descriptions
    match_a = re.search(r'TEXTURE_A:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    match_b = re.search(r'TEXTURE_B:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match_a:
        desc_a = match_a.group(1).strip()
    if match_b:
        desc_b = match_b.group(1).strip()

    # Extract coordinates — supports both float (normalized) and int (pixel)
    def extract_points(pattern, text):
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return []
        coords_str = match.group(1)
        # Match float or int pairs: (0.3, 0.7) or (80, 120)
        pairs = re.findall(r'\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)', coords_str)
        if not pairs:
            return []
        result = []
        for xs, ys in pairs:
            x, y = float(xs), float(ys)
            # If values are <= 1.0, they're normalized — convert to pixel space
            if x <= 1.0 and y <= 1.0:
                x = x * img_size
                y = y * img_size
            result.append([x, y])
        return result

    raw_points_a = extract_points(r'POINTS_A:\s*(.+?)(?:\n|$)', text)
    raw_points_b = extract_points(r'POINTS_B:\s*(.+?)(?:\n|$)', text)

    # Validate: clamp to safe range
    margin = 10
    lo, hi = margin, img_size - margin

    def validate_points(pts):
        valid = []
        for p in pts:
            x = max(lo, min(hi, p[0]))
            y = max(lo, min(hi, p[1]))
            valid.append([x, y])
        return valid

    points_a = validate_points(raw_points_a)
    points_b = validate_points(raw_points_b)

    success = bool(desc_a and desc_b and points_a and points_b)
    return desc_a, desc_b, points_a, points_b, success


# ===================================================================== #
#  Visualization                                                          #
# ===================================================================== #

def _draw_points_on_image(img, pts_a, pts_b, scale, marker="circle"):
    """Draw point markers on an image copy. marker='circle' or 'diamond'."""
    out = img.copy()
    if pts_a:
        for pt in pts_a:
            cx, cy = int(pt[0] * scale), int(pt[1] * scale)
            if marker == "circle":
                cv2.circle(out, (cx, cy), 9, (0, 0, 255), -1)       # red (BGR)
                cv2.circle(out, (cx, cy), 9, (255, 255, 255), 2)
            else:
                d = 7
                diamond = np.array([[cx, cy-d], [cx+d, cy], [cx, cy+d], [cx-d, cy]], np.int32)
                cv2.fillPoly(out, [diamond], (0, 0, 255))
                cv2.polylines(out, [diamond], True, (255, 255, 255), 2)
    if pts_b:
        for pt in pts_b:
            cx, cy = int(pt[0] * scale), int(pt[1] * scale)
            if marker == "circle":
                cv2.circle(out, (cx, cy), 9, (255, 0, 0), -1)       # blue (BGR)
                cv2.circle(out, (cx, cy), 9, (255, 255, 255), 2)
            else:
                d = 7
                diamond = np.array([[cx, cy-d], [cx+d, cy], [cx, cy+d], [cx-d, cy]], np.int32)
                cv2.fillPoly(out, [diamond], (255, 0, 0))
                cv2.polylines(out, [diamond], True, (255, 255, 255), 2)
    return out


def _make_label_cell(text, h, w, font_scale=0.4, bg=30):
    """Create a dark label cell with centered text."""
    cell = np.zeros((h, w, 3), dtype=np.uint8) + bg
    cv2.putText(cell, text, (6, h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 1, cv2.LINE_AA)
    return cell


def _hstack_cells(cells, sep=3):
    """Horizontally stack cells with separator columns."""
    parts = []
    for c in cells:
        parts.append(c)
        parts.append(np.zeros((c.shape[0], sep, 3), dtype=np.uint8))
    return np.hstack(parts[:-1]) if parts else np.zeros((1, 1, 3), dtype=np.uint8)


def _make_section_header(text, width, h=28, bg=40):
    """Create a section header bar."""
    bar = np.zeros((h, width, 3), dtype=np.uint8) + bg
    cv2.putText(bar, text, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)
    return bar


def create_comparison_figures(
    image_bgr, gt_a, gt_b,
    preds_dict,  # {label: (mask_a, mask_b, metrics)}
    title="", desc_a="", desc_b="",
    cell_size=320,
    oracle_pts_a=None, oracle_pts_b=None,
):
    """
    Generate separate, clean figures — one per logical group.
    Returns dict of {suffix: numpy_image}:
      "01_gt"       — Image + Oracle Points | GT Overlay | GT Masks
      "02_baseline"  — Generic ZS: overlay | masks
      "03_oracle"    — OracleTxt | PtsOnly | Orc T+P (overlay | masks each)
      "04_qwen"      — Qw3Prop | Qw3Sem (overlay | masks each)
    """
    h, w = image_bgr.shape[:2]
    s = cell_size / max(h, w)
    ch, cw = int(h * s), int(w * s)
    sep = 4
    font = 0.5
    lbl_h = 30

    def ri(img):
        return cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)

    def rm(mask):
        return cv2.resize(mask.astype(np.float32), (cw, ch),
                          interpolation=cv2.INTER_NEAREST)

    img = ri(image_bgr)
    ga, gb = rm(gt_a), rm(gt_b)

    def label_bar(text, width):
        bar = np.zeros((lbl_h, width, 3), dtype=np.uint8) + 35
        cv2.putText(bar, text, (8, lbl_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (230, 230, 230), 1, cv2.LINE_AA)
        return bar

    def approach_block(ma, mb, label, met):
        """Single approach: label + overlay | masks."""
        iou = met.get("mean_iou", 0)
        ari = met.get("ari", met.get("mean_ari", 0))
        overlay = mask_overlay(img, ma, mb)
        masks = binary_mask_image(ma, mb, ch, cw)
        pair = _hstack_cells([overlay, masks], sep)
        lbl = label_bar(f"{label}   mIoU: {iou:.3f}   ARI: {ari:.3f}", pair.shape[1])
        return np.vstack([lbl, pair])

    def group_row(blocks):
        """Horizontally join multiple approach blocks."""
        return _hstack_cells(blocks, sep * 3)

    figures = {}

    # ---- 01: GT row ----------------------------------------------------------
    img_pts = _draw_points_on_image(img, oracle_pts_a, oracle_pts_b, s, "circle")
    gt_over = mask_overlay(img, ga, gb)
    gt_mask = binary_mask_image(ga, gb, ch, cw)
    gt_row = _hstack_cells([img_pts, gt_over, gt_mask], sep)
    gt_labels = _hstack_cells([
        label_bar("Image + Oracle Pts", cw),
        label_bar("GT Overlay", cw),
        label_bar("GT Masks", cw),
    ], sep)
    # Description header
    desc = f"{title}  |  A(red): {desc_a[:60]}  |  B(blue): {desc_b[:60]}"
    hdr = label_bar(desc, gt_row.shape[1])
    figures["01_gt"] = np.vstack([hdr, gt_labels, gt_row])

    # ---- 02: Baseline --------------------------------------------------------
    gen = preds_dict.get("Generic")
    if gen:
        ma, mb, met = gen
        figures["02_baseline"] = approach_block(rm(ma), rm(mb), "Generic ZS", met)

    # ---- 03: Oracle approaches -----------------------------------------------
    orc_blocks = []
    for key in ["OracleTxt", "PtsOnly", "Orc T+P"]:
        data = preds_dict.get(key)
        if data:
            ma, mb, met = data
            orc_blocks.append(approach_block(rm(ma), rm(mb), key, met))
    if orc_blocks:
        figures["03_oracle"] = group_row(orc_blocks)

    # ---- 04: Qwen automated --------------------------------------------------
    qw_blocks = []
    for key in ["Qw3Prop", "Qw3Sem"]:
        data = preds_dict.get(key)
        if data:
            ma, mb, met = data
            qw_blocks.append(approach_block(rm(ma), rm(mb), key, met))
    if qw_blocks:
        figures["04_qwen"] = group_row(qw_blocks)

    return figures


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Oracle Descriptions + Points Evaluation on RWTD")
    parser.add_argument("--config", type=str,
                        default="qwen2sam/configs/v3_tracker_detexure.yaml")
    parser.add_argument("--data_root", type=str, default="/home/aviad/RWTD")
    parser.add_argument("--metadata_file", type=str,
                        default="metadata_phase1.json")
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/sam3_oracle_points")
    parser.add_argument("--cell_size", type=int, default=320)
    parser.add_argument("--no_vis", action="store_true")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    cfg = load_config(str(config_path))
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = cfg["model"].get("image_size", 1008)

    # ---- Load metadata ------------------------------------------------ #
    metadata_path = Path(args.data_root) / args.metadata_file
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} samples from {metadata_path}")

    # ---- Output dirs -------------------------------------------------- #
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build model -------------------------------------------------- #
    # Remove v3_checkpoint so the constructor doesn't try to load a trained ckpt
    # (this is a pure zero-shot SAM3 test — no training involved)
    if "tracker" in cfg:
        cfg["tracker"].pop("v3_checkpoint", None)
    print("Building Qwen2SAMv3Tracker model (SAM3 + tracker heads)...")
    model = Qwen2SAMv3Tracker(cfg, device=str(device))
    model.base.sam3.eval()
    model.sam_prompt_encoder.eval()
    model.sam_mask_decoder.eval()

    # ---- Load Qwen3 model ---------------------------------------------- #
    qwen3_model, qwen3_processor = load_qwen3_model(device)

    # ---- Load CLIPSeg model --------------------------------------------- #
    clipseg_model, clipseg_proc = load_clipseg_model(device)

    # ---- Evaluate ----------------------------------------------------- #
    generic_metrics = []
    oracle_text_metrics = []
    points_only_metrics = []
    oracle_text_points_metrics = []
    qwen3_text_metrics = []
    qwen3_semseg_metrics = []
    qwen3_infer_metrics = []
    qwen3_cseg_metrics = []
    qwen3_cseg_sem_metrics = []
    qwen3_parse_failures = 0
    qwen3_outputs_log = []

    t0 = time.time()
    trunk_hook_cache = {}

    def _trunk_hook(module, input, output):
        trunk_hook_cache["xs"] = output

    for i, entry in enumerate(metadata):
        crop_name = entry.get("crop_name", f"sample_{i}")
        desc_a = entry["texture_a"]
        desc_b = entry["texture_b"]
        oracle_pts = entry.get("oracle_points", {})
        pts_a = oracle_pts.get("point_prompt_mask_a", [])
        pts_b = oracle_pts.get("point_prompt_mask_b", [])

        # Load image
        image_bgr = cv2.imread(entry["image_path"])
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        # Load GT masks
        gt_a = cv2.imread(entry["mask_a_path"], cv2.IMREAD_GRAYSCALE)
        gt_b = cv2.imread(entry["mask_b_path"], cv2.IMREAD_GRAYSCALE)
        if gt_a is None or gt_b is None:
            continue
        gt_a = gt_a.astype(np.float32) / 255.0
        gt_b = gt_b.astype(np.float32) / 255.0
        gt_h, gt_w = gt_a.shape

        # Preprocess for SAM3
        sam_img = preprocess_image_for_sam3(image_rgb, image_size).unsqueeze(0).to(device)

        # Register trunk hook to capture ViT features for tracker
        trunk = model.base.sam3.backbone.vision_backbone.trunk
        hook = trunk.register_forward_hook(_trunk_hook)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # --- SAM3 backbone (shared) ---
            backbone_out = model.base.sam3.backbone.forward_image(sam_img)
            backbone_out["img_batch_all_stages"] = sam_img

            # --- 1. Generic ZS: "texture" top-2 ---
            gen_a_logit, gen_b_logit = run_detr_generic_top2(
                model, backbone_out, B=1, device=device)
            gen_a = postprocess_mask(gen_a_logit, gt_h, gt_w)
            gen_b = postprocess_mask(gen_b_logit, gt_h, gt_w)

            # --- 2. Oracle text: per-texture descriptions ---
            text_feat_a_out = model.base.sam3.backbone.forward_text([desc_a], device=device)
            text_feat_b_out = model.base.sam3.backbone.forward_text([desc_b], device=device)
            text_feat_a = {
                "prompt": text_feat_a_out["language_features"].squeeze(1),
                "mask": text_feat_a_out["language_mask"].squeeze(0),
            }
            text_feat_b = {
                "prompt": text_feat_b_out["language_features"].squeeze(1),
                "mask": text_feat_b_out["language_mask"].squeeze(0),
            }

            orc_a_logit, _ = run_detr_with_text(model, backbone_out, text_feat_a, 1, device)
            orc_b_logit, _ = run_detr_with_text(model, backbone_out, text_feat_b, 1, device)
            orc_a = postprocess_mask(orc_a_logit, gt_h, gt_w)
            orc_b = postprocess_mask(orc_b_logit, gt_h, gt_w)

            # --- 3. Oracle text + points: tracker refinement ---
            hook.remove()
            trunk_output = trunk_hook_cache["xs"][-1]

            if pts_a and pts_b:
                # Scale points from original image space (256) to SAM space (1008)
                pts_a_scaled = scale_points(pts_a, orig_w, image_size)
                pts_b_scaled = scale_points(pts_b, orig_w, image_size)

                # Format: (1, N, 2) — [x, y] in absolute 1008 coords
                pos_a = torch.tensor(pts_a_scaled, dtype=torch.float32,
                                     device=device).unsqueeze(0)
                pos_b = torch.tensor(pts_b_scaled, dtype=torch.float32,
                                     device=device).unsqueeze(0)

                # --- 3a. Points only: generic DETR coarse mask + oracle points ---
                pto_a_out = run_tracker_refinement(
                    model, trunk_output, gen_a_logit, pos_a, pos_b, device)
                pto_b_out = run_tracker_refinement(
                    model, trunk_output, gen_b_logit, pos_b, pos_a, device)
                pto_a = postprocess_mask(pto_a_out.squeeze(), gt_h, gt_w)
                pto_b = postprocess_mask(pto_b_out.squeeze(), gt_h, gt_w)

                # --- 3b. Oracle text + points: oracle DETR coarse mask + oracle points ---
                refined_a_out = run_tracker_refinement(
                    model, trunk_output, orc_a_logit, pos_a, pos_b, device)
                refined_b_out = run_tracker_refinement(
                    model, trunk_output, orc_b_logit, pos_b, pos_a, device)
                ref_a = postprocess_mask(refined_a_out.squeeze(), gt_h, gt_w)
                ref_b = postprocess_mask(refined_b_out.squeeze(), gt_h, gt_w)
            else:
                # No oracle points available, fall back
                pto_a, pto_b = gen_a, gen_b
                ref_a, ref_b = orc_a, orc_b

            # --- 5. Qwen3-VL-8B inference (text-only prompt) ---
            image_pil = Image.fromarray(image_rgb)
            q3_raw = qwen_generate(
                qwen3_model, qwen3_processor, image_pil, device)
            q3_desc_a, q3_desc_b, q3_ok = parse_qwen_text_only(q3_raw)

            qwen3_outputs_log.append({
                "crop_name": crop_name,
                "raw_output": q3_raw,
                "parsed": {"desc_a": q3_desc_a, "desc_b": q3_desc_b},
                "parse_ok": q3_ok,
            })

            if q3_ok:
                q3_text_a_out = model.base.sam3.backbone.forward_text(
                    [q3_desc_a], device=device)
                q3_text_b_out = model.base.sam3.backbone.forward_text(
                    [q3_desc_b], device=device)
                q3_feat_a = {
                    "prompt": q3_text_a_out["language_features"].squeeze(1),
                    "mask": q3_text_a_out["language_mask"].squeeze(0),
                }
                q3_feat_b = {
                    "prompt": q3_text_b_out["language_features"].squeeze(1),
                    "mask": q3_text_b_out["language_mask"].squeeze(0),
                }
                # Single DETR pass: get proposal mask + semantic mask
                q3_a_logit, _, q3_sem_a = run_detr_with_text_full(
                    model, backbone_out, q3_feat_a, 1, device)
                q3_b_logit, _, q3_sem_b = run_detr_with_text_full(
                    model, backbone_out, q3_feat_b, 1, device)

                # Qwen3 text — proposal (top-1 DETR query)
                q3t_a = postprocess_mask(q3_a_logit, gt_h, gt_w)
                q3t_b = postprocess_mask(q3_b_logit, gt_h, gt_w)

                # Qwen3 text + Qwen3 points (legacy — no points in new prompt)
                q3_a, q3_b = q3t_a, q3t_b

                # --- Qwen3 Semantic Segmentation (WTA) ---
                if q3_sem_a is not None and q3_sem_b is not None:
                    sem_prob_a = postprocess_semantic_mask(q3_sem_a[0, 0], gt_h, gt_w)
                    sem_prob_b = postprocess_semantic_mask(q3_sem_b[0, 0], gt_h, gt_w)

                    q3sem_a = (sem_prob_a > sem_prob_b).astype(np.float32)
                    q3sem_b = (sem_prob_b > sem_prob_a).astype(np.float32)

                    # Degenerate fallback (< 2% pixels)
                    if min(q3sem_a.mean(), q3sem_b.mean()) < 0.02:
                        q3sem_a = (sem_prob_a > 0.5).astype(np.float32)
                        q3sem_b = (sem_prob_b > 0.5).astype(np.float32)

                    # Label swap check
                    iou_d = _iou(q3sem_a, gt_a) + _iou(q3sem_b, gt_b)
                    iou_s = _iou(q3sem_a, gt_b) + _iou(q3sem_b, gt_a)
                    if iou_s > iou_d:
                        q3sem_a, q3sem_b = q3sem_b, q3sem_a
                else:
                    q3sem_a, q3sem_b = gen_a, gen_b

                # --- 7. Qwen3 + CLIPSeg: text + CLIPSeg points + binary mask ---
                cseg_pts_a, cseg_pts_b, cseg_mask_a, cseg_mask_b = \
                    clipseg_extract_points_and_mask(
                        clipseg_model, clipseg_proc, image_pil,
                        q3_desc_a, q3_desc_b,
                        img_size=orig_w, device=device)

                cseg_pts_a_s = scale_points(cseg_pts_a, orig_w, image_size)
                cseg_pts_b_s = scale_points(cseg_pts_b, orig_w, image_size)
                cseg_pos_a = torch.tensor(
                    cseg_pts_a_s, dtype=torch.float32, device=device).unsqueeze(0)
                cseg_pos_b = torch.tensor(
                    cseg_pts_b_s, dtype=torch.float32, device=device).unsqueeze(0)

                # Use CLIPSeg binary mask (logit scale) as coarse mask
                cseg_coarse_a = torch.from_numpy(cseg_mask_a).float().to(device)
                cseg_coarse_b = torch.from_numpy(cseg_mask_b).float().to(device)

                cseg_ref_a = run_tracker_refinement(
                    model, trunk_output, cseg_coarse_a,
                    cseg_pos_a, cseg_pos_b, device)
                cseg_ref_b = run_tracker_refinement(
                    model, trunk_output, cseg_coarse_b,
                    cseg_pos_b, cseg_pos_a, device)
                cseg_a = postprocess_mask(cseg_ref_a.squeeze(), gt_h, gt_w)
                cseg_b = postprocess_mask(cseg_ref_b.squeeze(), gt_h, gt_w)

                # --- 8. Qw3+CSeg(sem): CLIPSeg points + Semantic Seg coarse mask → tracker ---
                if q3_sem_a is not None and q3_sem_b is not None:
                    # Use semantic seg mask as coarse input (logit scale)
                    sem_coarse_a = q3_sem_a[0, 0].float()   # (H, W) logits
                    sem_coarse_b = q3_sem_b[0, 0].float()

                    cseg_sem_ref_a = run_tracker_refinement(
                        model, trunk_output, sem_coarse_a,
                        cseg_pos_a, cseg_pos_b, device)
                    cseg_sem_ref_b = run_tracker_refinement(
                        model, trunk_output, sem_coarse_b,
                        cseg_pos_b, cseg_pos_a, device)
                    cseg_sem_a = postprocess_mask(cseg_sem_ref_a.squeeze(), gt_h, gt_w)
                    cseg_sem_b = postprocess_mask(cseg_sem_ref_b.squeeze(), gt_h, gt_w)
                else:
                    cseg_sem_a, cseg_sem_b = cseg_a, cseg_b
            else:
                qwen3_parse_failures += 1
                q3t_a, q3t_b = gen_a, gen_b
                q3_a, q3_b = gen_a, gen_b
                q3sem_a, q3sem_b = gen_a, gen_b
                cseg_a, cseg_b = gen_a, gen_b
                cseg_sem_a, cseg_sem_b = gen_a, gen_b
                cseg_pts_a, cseg_pts_b = [], []

        # --- Compute metrics ---
        gen_met = compute_sample_metrics(gen_a, gen_b, gt_a, gt_b, crop_name)
        orc_met = compute_sample_metrics(orc_a, orc_b, gt_a, gt_b, crop_name)
        pto_met = compute_sample_metrics(pto_a, pto_b, gt_a, gt_b, crop_name)
        ref_met = compute_sample_metrics(ref_a, ref_b, gt_a, gt_b, crop_name)
        q3t_met = compute_sample_metrics(q3t_a, q3t_b, gt_a, gt_b, crop_name)
        q3sem_met = compute_sample_metrics(q3sem_a, q3sem_b, gt_a, gt_b, crop_name)
        q3_met = compute_sample_metrics(q3_a, q3_b, gt_a, gt_b, crop_name)
        cseg_met = compute_sample_metrics(cseg_a, cseg_b, gt_a, gt_b, crop_name)
        cseg_sem_met = compute_sample_metrics(cseg_sem_a, cseg_sem_b, gt_a, gt_b, crop_name)

        generic_metrics.append(gen_met)
        oracle_text_metrics.append(orc_met)
        points_only_metrics.append(pto_met)
        oracle_text_points_metrics.append(ref_met)
        qwen3_text_metrics.append(q3t_met)
        qwen3_semseg_metrics.append(q3sem_met)
        qwen3_infer_metrics.append(q3_met)
        qwen3_cseg_metrics.append(cseg_met)
        qwen3_cseg_sem_metrics.append(cseg_sem_met)

        # --- Visualization ---
        if not args.no_vis:
            preds = {
                "Generic": (gen_a, gen_b, gen_met),
                "OracleTxt": (orc_a, orc_b, orc_met),
                "PtsOnly": (pto_a, pto_b, pto_met),
                "Orc T+P": (ref_a, ref_b, ref_met),
                "Qw3Prop": (q3t_a, q3t_b, q3t_met),
                "Qw3Sem": (q3sem_a, q3sem_b, q3sem_met),
            }
            figs = create_comparison_figures(
                image_bgr, gt_a, gt_b, preds,
                title=crop_name, desc_a=desc_a, desc_b=desc_b,
                cell_size=args.cell_size,
                oracle_pts_a=pts_a, oracle_pts_b=pts_b,
            )
            for suffix, fig_img in figs.items():
                cv2.imwrite(str(vis_dir / f"{crop_name}_{suffix}.png"), fig_img)

        if (i + 1) % 20 == 0 or (i + 1) == len(metadata):
            g_iou = np.mean([m["mean_iou"] for m in generic_metrics])
            o_iou = np.mean([m["mean_iou"] for m in oracle_text_metrics])
            r_iou = np.mean([m["mean_iou"] for m in oracle_text_points_metrics])
            q3t_iou = np.mean([m["mean_iou"] for m in qwen3_text_metrics])
            q3sem_iou = np.mean([m["mean_iou"] for m in qwen3_semseg_metrics])
            cseg_iou = np.mean([m["mean_iou"] for m in qwen3_cseg_metrics])
            csegsem_iou = np.mean([m["mean_iou"] for m in qwen3_cseg_sem_metrics])
            print(f"  {i+1}/{len(metadata)} | Gen: {g_iou:.4f} | OrcTxt: {o_iou:.4f} | "
                  f"OrcT+P: {r_iou:.4f} | Qw3Prop: {q3t_iou:.4f} | "
                  f"Qw3Sem: {q3sem_iou:.4f} | Qw3+CSeg: {cseg_iou:.4f} | "
                  f"CSeg(sem): {csegsem_iou:.4f}")

    elapsed = time.time() - t0

    # ---- Aggregate ---------------------------------------------------- #
    gen_sum = aggregate_metrics(generic_metrics, "generic_texture")
    orc_sum = aggregate_metrics(oracle_text_metrics, "oracle_text")
    pto_sum = aggregate_metrics(points_only_metrics, "points_only")
    ref_sum = aggregate_metrics(oracle_text_points_metrics, "oracle_text_points")
    q3t_sum = aggregate_metrics(qwen3_text_metrics, "qwen3_text_proposal")
    q3sem_sum = aggregate_metrics(qwen3_semseg_metrics, "qwen3_text_semseg")
    q3_sum = aggregate_metrics(qwen3_infer_metrics, "qwen3_text_points")
    cseg_sum = aggregate_metrics(qwen3_cseg_metrics, "qwen3_clipseg")
    cseg_sem_sum = aggregate_metrics(qwen3_cseg_sem_metrics, "qwen3_clipseg_sem")

    # ---- Improvements ------------------------------------------------- #
    def compute_improvement(base, target):
        imp, pct = {}, {}
        for k in ["mean_iou", "mean_dice", "mean_ari"]:
            delta = target[k] - base[k]
            imp[k] = delta
            pct[k] = round(100 * delta / base[k], 2) if abs(base[k]) > 1e-8 else 0.0
        return imp, pct

    q3t_vs_gen_imp, q3t_vs_gen_pct = compute_improvement(gen_sum, q3t_sum)
    q3t_vs_orc_imp, q3t_vs_orc_pct = compute_improvement(orc_sum, q3t_sum)

    # ---- Save --------------------------------------------------------- #
    save_metrics_csv(generic_metrics, output_dir / "metrics_generic.csv")
    save_metrics_csv(oracle_text_metrics, output_dir / "metrics_oracle_text.csv")
    save_metrics_csv(points_only_metrics, output_dir / "metrics_points_only.csv")
    save_metrics_csv(oracle_text_points_metrics, output_dir / "metrics_oracle_text_points.csv")
    save_metrics_csv(qwen3_text_metrics, output_dir / "metrics_qwen3_text_proposal.csv")
    save_metrics_csv(qwen3_semseg_metrics, output_dir / "metrics_qwen3_semseg.csv")
    save_metrics_csv(qwen3_infer_metrics, output_dir / "metrics_qwen3_text_points.csv")
    save_metrics_csv(qwen3_cseg_metrics, output_dir / "metrics_qwen3_clipseg.csv")
    save_metrics_csv(qwen3_cseg_sem_metrics, output_dir / "metrics_qwen3_clipseg_sem.csv")

    with open(output_dir / "qwen3_outputs.json", "w") as f:
        json.dump(qwen3_outputs_log, f, indent=2)

    summary = {
        "generic_texture": gen_sum,
        "oracle_text": orc_sum,
        "points_only": pto_sum,
        "oracle_text_points": ref_sum,
        "qwen3_text_proposal": q3t_sum,
        "qwen3_text_semseg": q3sem_sum,
        "qwen3_text_points": q3_sum,
        "qwen3_clipseg": cseg_sum,
        "qwen3_clipseg_sem": cseg_sem_sum,
        "qwen3_parse_failures": qwen3_parse_failures,
        "qwen3_parse_success_rate": round(
            100 * (len(metadata) - qwen3_parse_failures) / max(len(metadata), 1), 1),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Print -------------------------------------------------------- #
    cseg_vs_gen_imp, cseg_vs_gen_pct = compute_improvement(gen_sum, cseg_sum)

    n = len(metadata)
    w = 120
    print(f"\n{'='*w}")
    print(f"  SAM3 9-Way Evaluation — {n} samples ({elapsed:.1f}s)")
    print(f"{'='*w}")
    print(f"  {'':20s} {'Generic':>8s} {'OrcTxt':>8s} {'PtsOnly':>8s} {'Orc T+P':>8s}"
          f" {'Qw3Prop':>8s} {'Qw3Sem':>8s} {'Qw3+CSeg':>9s} {'CSeg(sem)':>10s}")
    print(f"  {'─'*w}")
    for label, key in [("Mean IoU", "mean_iou"), ("Mean IoU (A)", "mean_iou_a"),
                        ("Mean IoU (B)", "mean_iou_b"), ("Mean Dice", "mean_dice"),
                        ("Mean ARI", "mean_ari")]:
        vals = [gen_sum, orc_sum, pto_sum, ref_sum, q3t_sum, q3sem_sum, cseg_sum, cseg_sem_sum]
        parts = " ".join(f"{v[key]:8.4f}" for v in vals)
        print(f"  {label:20s} {parts}")
    print(f"  {'Samples':20s} " + " ".join(f"{v['num_samples']:8d}" for v in
          [gen_sum, orc_sum, pto_sum, ref_sum, q3t_sum, q3sem_sum, cseg_sum, cseg_sem_sum]))
    print(f"{'='*w}")
    print(f"    vs Generic: {cseg_vs_gen_imp['mean_iou']:+.4f} ({cseg_vs_gen_pct['mean_iou']:+.1f}%)")
    print(f"    Parse success: {len(metadata) - qwen3_parse_failures}/{len(metadata)}")
    print(f"    CSeg impact: Txt={q3t_sum['mean_iou']:.4f} → +CSeg={cseg_sum['mean_iou']:.4f} "
          f"({cseg_sum['mean_iou'] - q3t_sum['mean_iou']:+.4f})")

    print(f"\n{'='*w}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*w}")


if __name__ == "__main__":
    main()
