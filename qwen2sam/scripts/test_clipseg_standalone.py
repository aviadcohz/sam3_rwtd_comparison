"""
CLIPSeg standalone vs CLIPSeg+SAM2 vs QwenTxt comparison tester.

Tests three approaches side-by-side:
  1. CSeg-Only:    Qwen3 text → CLIPSeg heatmaps → WTA binary masks (NO SAM at all)
  2. CSeg+SAM2:    CLIPSeg coarse mask + 2 CLIPSeg points → SAM2 refinement
  3. QwenTxt:      Qwen3 text → SAM3 DETR text masks (existing baseline)

The motivation: CLIPSeg already produces good masks on its own. Can SAM2
refine them further using CLIPSeg's coarse mask as a prompt?

Usage:
  python qwen2sam/scripts/test_clipseg_standalone.py
  python qwen2sam/scripts/test_clipseg_standalone.py --samples 1,2,3,16,101
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
#  PROMPT                                                                 #
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
    "TEXTURE_A: Texture of <description>\n"
    "TEXTURE_B: <description>"
)

DEFAULT_SAMPLES = None  # None = run all


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


def heatmap_to_bgr(heatmap, size=None):
    h = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    if size:
        colored = cv2.resize(colored, size, interpolation=cv2.INTER_LINEAR)
    return colored


# ===================================================================== #
#  SAM3 DETR helpers (for QwenTxt baseline)                               #
# ===================================================================== #

@torch.no_grad()
def run_detr_with_text(model, backbone_out, text_feat, B, device):
    prompt_bf = text_feat["prompt"].unsqueeze(0).expand(B, -1, -1)
    mask_bf = text_feat["mask"].unsqueeze(0).expand(B, -1)
    out = model.base._run_sam3_from_backbone(backbone_out, prompt_bf, mask_bf)
    scores = out["pred_logits"][0].squeeze(-1).sigmoid()
    best = scores.argmax().item()
    return out["pred_masks"][0, best], out["pred_masks"]


# ===================================================================== #
#  SAM2 tracker refinement (for CSeg+SAM2)                                #
# ===================================================================== #

@torch.no_grad()
def run_tracker_refinement(model, trunk_output, coarse_mask_logit,
                           pos_points_abs, neg_points_abs, device):
    """Run SAM2 mask decoder refinement with coarse mask + point prompts."""
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


# ===================================================================== #
#  Qwen3 text generation                                                  #
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
#  CLIPSeg standalone mask generation (NO SAM3)                           #
# ===================================================================== #

@torch.no_grad()
def clipseg_standalone_masks(clipseg_model, clipseg_proc, image_pil,
                             desc_a, desc_b, target_h, target_w, device="cuda",
                             threshold=0.0, morph_close_iter=3, morph_open_iter=1):
    """
    Generate final binary masks using CLIPSeg alone — no SAM3 at all.

    Pipeline:
      1. Run CLIPSeg with desc_a and desc_b to get per-pixel heatmaps
      2. Compute diff maps (A-B, B-A) for mutual exclusion
      3. Assign each pixel to whichever texture has higher activation
      4. Apply morphological cleanup
      5. Resize to ground-truth resolution

    Returns: mask_a, mask_b (np float32 binary), raw heatmaps, diff maps
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
        raw_heatmaps.append(heatmap.cpu().numpy())

    heat_a, heat_b = raw_heatmaps

    # Diff maps for mutual exclusion
    diff_a = np.clip(heat_a - heat_b, 0, 1)
    diff_b = np.clip(heat_b - heat_a, 0, 1)

    # Winner-takes-all assignment: each pixel goes to the dominant texture
    mask_a_raw = (heat_a > heat_b + threshold).astype(np.uint8)
    mask_b_raw = (heat_b > heat_a + threshold).astype(np.uint8)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if morph_close_iter > 0:
        mask_a_raw = cv2.morphologyEx(mask_a_raw, cv2.MORPH_CLOSE, kernel,
                                       iterations=morph_close_iter)
        mask_b_raw = cv2.morphologyEx(mask_b_raw, cv2.MORPH_CLOSE, kernel,
                                       iterations=morph_close_iter)
    if morph_open_iter > 0:
        mask_a_raw = cv2.morphologyEx(mask_a_raw, cv2.MORPH_OPEN, kernel,
                                       iterations=morph_open_iter)
        mask_b_raw = cv2.morphologyEx(mask_b_raw, cv2.MORPH_OPEN, kernel,
                                       iterations=morph_open_iter)

    # Resize to ground truth resolution
    mask_a_final = cv2.resize(mask_a_raw.astype(np.float32), (target_w, target_h),
                               interpolation=cv2.INTER_NEAREST)
    mask_b_final = cv2.resize(mask_b_raw.astype(np.float32), (target_w, target_h),
                               interpolation=cv2.INTER_NEAREST)

    return mask_a_final, mask_b_final, heat_a, heat_b, diff_a, diff_b


@torch.no_grad()
def clipseg_extract_points(clipseg_model, clipseg_proc, image_pil,
                            desc_a, desc_b, img_size=256, device="cuda",
                            n_points=4, erode_iter=3, top_percentile=0.3):
    """
    Extract spread-apart point prompts from CLIPSeg diff heatmaps.
    Returns: pts_a, pts_b (lists of [x,y] in pixel space)
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

    diff_maps = [
        np.clip(raw_heatmaps[0] - raw_heatmaps[1], 0, 1),
        np.clip(raw_heatmaps[1] - raw_heatmaps[0], 0, 1),
    ]

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

    return results[0], results[1]


# ===================================================================== #
#  Visualization grid                                                     #
# ===================================================================== #

def draw_debug_grid(image_bgr, gt_a, gt_b, preds,
                    heatmaps, diff_maps, q_desc_a, q_desc_b,
                    title, cell_size=256):
    """
    Visualization grid:
      Row 1: overlay (pred on image)
      Row 2: binary masks
      Row 3: heatmap / diff used for this method

    Columns: Image | Heatmap A|B | Diff A|B | GT | CSeg-WTA | CSeg+SAM2 | QwenTxt
    """
    h, w = image_bgr.shape[:2]
    s = cell_size / max(h, w)
    ch, cw = int(h * s), int(w * s)

    def ri(img):
        return cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)

    def rm(mask):
        return cv2.resize(mask.astype(np.float32), (cw, ch),
                          interpolation=cv2.INTER_NEAREST)

    img = ri(image_bgr)
    ga, gb = rm(gt_a), rm(gt_b)

    heat_a, heat_b = heatmaps
    diff_a, diff_b = diff_maps

    # Heatmaps vis
    heat_a_vis = heatmap_to_bgr(heat_a, (cw, ch))
    heat_b_vis = heatmap_to_bgr(heat_b, (cw, ch))
    # Diff vis
    d_max = max(diff_a.max(), diff_b.max(), 1e-8)
    diff_a_vis = heatmap_to_bgr(diff_a / d_max, (cw, ch))
    diff_b_vis = heatmap_to_bgr(diff_b / d_max, (cw, ch))

    # GT column
    gt_ov = mask_overlay(img, ga, gb)
    gt_bin = binary_mask_image(ga, gb, ch, cw)

    # Build columns: (row1_overlay, row2_binary, row3_extra)
    blank = np.zeros((ch, cw, 3), dtype=np.uint8)

    cols = [
        (img, blank, blank),                          # Image
        (heat_a_vis, heat_b_vis, blank),              # Heatmap A | B
        (diff_a_vis, diff_b_vis, blank),              # Diff A | B
        (gt_ov, gt_bin, blank),                       # GT
    ]
    col_labels = ["Image", "Heat A|B", "Diff A|B", "GT"]

    for label, (ma, mb, met) in preds.items():
        ma_r, mb_r = rm(ma), rm(mb)
        ov = mask_overlay(img, ma_r, mb_r)
        bn = binary_mask_image(ma_r, mb_r, ch, cw)
        cols.append((ov, bn, blank))
        col_labels.append(f"{label} {met['mean_iou']:.3f}")

    # Assemble
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
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(desc_bar, f"B: {q_desc_b}"[:actual_w // 5], (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 150, 0), 1, cv2.LINE_AA)

    # Row labels
    row_label_bar = np.zeros((row_labels_h, actual_w, 3), dtype=np.uint8) + 10
    cv2.putText(row_label_bar,
                "Row1: Overlay on image   Row2: Binary masks   Row3: (unused)",
                (8, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (150, 150, 150), 1, cv2.LINE_AA)

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
    parser = argparse.ArgumentParser(
        description="CLIPSeg standalone vs QwenTxt comparison")
    parser.add_argument("--config", type=str,
                        default="qwen2sam/configs/v3_tracker_detexure.yaml")
    parser.add_argument("--data_root", type=str, default="/home/aviad/RWTD")
    parser.add_argument("--samples", type=str, default=None,
                        help="Comma-separated crop names (default: all)")
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/clipseg_standalone_debug")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="WTA margin threshold (default: 0.0 = pure argmax)")
    args = parser.parse_args()

    sample_names = args.samples.split(",") if args.samples else DEFAULT_SAMPLES

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
        samples = all_meta
    else:
        samples = [meta_by_name[n] for n in sample_names if n in meta_by_name]
    print(f"Testing {len(samples)} samples")

    # Output dirs
    output_dir = _project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load models — SAM3 only for QwenTxt baseline
    if "tracker" in cfg:
        cfg["tracker"].pop("v3_checkpoint", None)
    print("Loading SAM3 model (for QwenTxt baseline)...")
    model = Qwen2SAMv3Tracker(cfg, device=str(device))
    model.base.sam3.eval()
    model.sam_prompt_encoder.eval()
    model.sam_mask_decoder.eval()

    qwen3, qwen3_proc = load_qwen3_model(device)
    clipseg_model, clipseg_proc = load_clipseg_model(device)

    # Metrics accumulators
    cseg_wta_metrics = []     # CLIPSeg winner-takes-all (no SAM)
    cseg_sam2_metrics = []    # CLIPSeg coarse mask + 2pts → SAM2
    txt_metrics = []          # QwenTxt (SAM3 DETR)

    outputs_log = []
    trunk_cache = {}

    def _trunk_hook(module, input, output):
        trunk_cache["xs"] = output

    t0 = time.time()

    for i, entry in enumerate(samples):
        crop_name = entry["crop_name"]

        image_bgr = cv2.imread(entry["image_path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        gt_a = cv2.imread(entry["mask_a_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_b = cv2.imread(entry["mask_b_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_h, gt_w = gt_a.shape

        image_pil = Image.fromarray(image_rgb)
        sam_img = preprocess_image_for_sam3(image_rgb, image_size).unsqueeze(0).to(device)

        # Register trunk hook for SAM2 features
        trunk = model.base.sam3.backbone.vision_backbone.trunk
        hook = trunk.register_forward_hook(_trunk_hook)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # --- Qwen3 text generation ---
            raw = qwen3_generate(qwen3, qwen3_proc, image_pil, device)
            q_desc_a, q_desc_b, ok = parse_text_output(raw)

            if not ok:
                print(f"\n  [{crop_name}] PARSE FAIL: {raw[:100]}")
                hook.remove()
                zero = np.zeros_like(gt_a)
                for ml in [cseg_wta_metrics, cseg_sam2_metrics, txt_metrics]:
                    ml.append(compute_sample_metrics(zero, zero, gt_a, gt_b, crop_name))
                continue

            # ============================================================ #
            #  APPROACH 1: CLIPSeg standalone — winner-takes-all (NO SAM)   #
            # ============================================================ #
            wta_a, wta_b, heat_a, heat_b, diff_a, diff_b = clipseg_standalone_masks(
                clipseg_model, clipseg_proc, image_pil,
                q_desc_a, q_desc_b, gt_h, gt_w, device=device,
                threshold=args.threshold)

            # ============================================================ #
            #  SAM3 backbone forward (needed for both CSeg+SAM2 & QwenTxt)  #
            # ============================================================ #
            backbone_out = model.base.sam3.backbone.forward_image(sam_img)
            backbone_out["img_batch_all_stages"] = sam_img

            hook.remove()
            trunk_output = trunk_cache["xs"][-1]

            # ============================================================ #
            #  APPROACH 2: CLIPSeg coarse mask + 2 points → SAM2 refinement #
            # ============================================================ #

            # Extract 2 CLIPSeg points per texture for SAM2 point prompts
            cseg_pts_a, cseg_pts_b = clipseg_extract_points(
                clipseg_model, clipseg_proc, image_pil,
                q_desc_a, q_desc_b, img_size=orig_w, device=device,
                n_points=2)

            cseg_pts_a_s = scale_points(cseg_pts_a, orig_w, image_size)
            cseg_pts_b_s = scale_points(cseg_pts_b, orig_w, image_size)
            pos_a = torch.tensor(cseg_pts_a_s, dtype=torch.float32, device=device).unsqueeze(0)
            neg_a = torch.tensor(cseg_pts_b_s, dtype=torch.float32, device=device).unsqueeze(0)

            # Convert CLIPSeg WTA mask to coarse logit for SAM2
            # Scale: binary {0,1} → logit-like {-5, +5} so SAM2 prompt encoder
            # gets a clear signal
            wta_a_resized = cv2.resize(wta_a, (orig_w, orig_h),
                                        interpolation=cv2.INTER_NEAREST)
            wta_b_resized = cv2.resize(wta_b, (orig_w, orig_h),
                                        interpolation=cv2.INTER_NEAREST)
            coarse_logit_a = torch.from_numpy(
                wta_a_resized * 10.0 - 5.0).float().to(device)
            coarse_logit_b = torch.from_numpy(
                wta_b_resized * 10.0 - 5.0).float().to(device)

            ref_a = run_tracker_refinement(
                model, trunk_output, coarse_logit_a, pos_a, neg_a, device)
            ref_b = run_tracker_refinement(
                model, trunk_output, coarse_logit_b, neg_a, pos_a, device)
            sam2_a = postprocess_mask(ref_a.squeeze(), gt_h, gt_w)
            sam2_b = postprocess_mask(ref_b.squeeze(), gt_h, gt_w)

            # ============================================================ #
            #  APPROACH 3: QwenTxt — SAM3 DETR with text (baseline)         #
            # ============================================================ #
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

        # Metrics
        wta_met = compute_sample_metrics(wta_a, wta_b, gt_a, gt_b, crop_name)
        sam2_met = compute_sample_metrics(sam2_a, sam2_b, gt_a, gt_b, crop_name)
        txt_met = compute_sample_metrics(txt_a, txt_b, gt_a, gt_b, crop_name)
        cseg_wta_metrics.append(wta_met)
        cseg_sam2_metrics.append(sam2_met)
        txt_metrics.append(txt_met)

        outputs_log.append({
            "crop_name": crop_name, "raw": raw,
            "desc_a": q_desc_a, "desc_b": q_desc_b,
            "cseg_wta_iou": wta_met["mean_iou"],
            "cseg_sam2_iou": sam2_met["mean_iou"],
            "qwen_txt_iou": txt_met["mean_iou"],
        })

        # Print
        print(f"\n  [{crop_name}] ({i+1}/{len(samples)})")
        print(f"    A: \"{q_desc_a}\"")
        print(f"    B: \"{q_desc_b}\"")
        print(f"    CSeg-WTA: {wta_met['mean_iou']:.4f} | "
              f"CSeg+SAM2: {sam2_met['mean_iou']:.4f} | "
              f"QwenTxt: {txt_met['mean_iou']:.4f}")

        # Visualization
        preds = {
            "CSeg-WTA": (wta_a, wta_b, wta_met),
            "CSeg+SAM2": (sam2_a, sam2_b, sam2_met),
            "QwenTxt": (txt_a, txt_b, txt_met),
        }
        grid = draw_debug_grid(
            image_bgr, gt_a, gt_b, preds,
            heatmaps=(heat_a, heat_b),
            diff_maps=(diff_a, diff_b),
            q_desc_a=q_desc_a, q_desc_b=q_desc_b,
            title=crop_name,
        )
        cv2.imwrite(str(vis_dir / f"{crop_name}_debug.png"), grid)

    elapsed = time.time() - t0

    # Summary
    wta_sum = aggregate_metrics(cseg_wta_metrics, "cseg_wta")
    sam2_sum = aggregate_metrics(cseg_sam2_metrics, "cseg_sam2")
    txt_sum = aggregate_metrics(txt_metrics, "qwen3_text")

    print(f"\n{'='*80}")
    print(f"  CLIPSeg Standalone vs CSeg+SAM2 vs QwenTxt — {len(samples)} samples ({elapsed:.1f}s)")
    print(f"{'='*80}")
    print(f"  {'':15s} {'CSeg-WTA':>12s} {'CSeg+SAM2':>12s} {'QwenTxt':>12s}")
    print(f"  {'─'*55}")
    for label, key in [("mIoU", "mean_iou"), ("mIoU-A", "mean_iou_a"),
                        ("mIoU-B", "mean_iou_b"), ("mDice", "mean_dice"),
                        ("mARI", "mean_ari")]:
        print(f"  {label:15s} {wta_sum[key]:12.4f} {sam2_sum[key]:12.4f} {txt_sum[key]:12.4f}")
    print(f"  {'─'*55}")
    print(f"{'='*80}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*80}")

    # Per-sample head-to-head comparisons
    def h2h(m1, m2):
        w = sum(1 for a, b in zip(m1, m2) if a["mean_iou"] > b["mean_iou"])
        l = sum(1 for a, b in zip(m1, m2) if b["mean_iou"] > a["mean_iou"])
        return w, l, len(m1) - w - l

    wta_v_txt = h2h(cseg_wta_metrics, txt_metrics)
    sam2_v_txt = h2h(cseg_sam2_metrics, txt_metrics)
    sam2_v_wta = h2h(cseg_sam2_metrics, cseg_wta_metrics)

    print(f"\n  Head-to-head:")
    print(f"    CSeg-WTA  vs QwenTxt:  {wta_v_txt[0]}W / {wta_v_txt[1]}L / {wta_v_txt[2]}T")
    print(f"    CSeg+SAM2 vs QwenTxt:  {sam2_v_txt[0]}W / {sam2_v_txt[1]}L / {sam2_v_txt[2]}T")
    print(f"    CSeg+SAM2 vs CSeg-WTA: {sam2_v_wta[0]}W / {sam2_v_wta[1]}L / {sam2_v_wta[2]}T")

    # Save
    with open(output_dir / "outputs.json", "w") as f:
        json.dump(outputs_log, f, indent=2, default=str)
    summary = {
        "cseg_wta": wta_sum,
        "cseg_sam2": sam2_sum,
        "qwen3_text": txt_sum,
        "head_to_head": {
            "cseg_wta_vs_txt": {"wins": wta_v_txt[0], "losses": wta_v_txt[1], "ties": wta_v_txt[2]},
            "cseg_sam2_vs_txt": {"wins": sam2_v_txt[0], "losses": sam2_v_txt[1], "ties": sam2_v_txt[2]},
            "cseg_sam2_vs_wta": {"wins": sam2_v_wta[0], "losses": sam2_v_wta[1], "ties": sam2_v_wta[2]},
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
