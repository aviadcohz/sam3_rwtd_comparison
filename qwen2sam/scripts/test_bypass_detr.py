"""
Bypass DETR: Use CLIPSeg mask directly with SAM3's boundary refinement.

Instead of using DETR proposals, feed CLIPSeg's binary heatmap directly
to SAM3's mask decoder as the coarse mask. SAM3 acts purely as a
boundary-sharpening tool.

Strategies compared:
  1. Qw3Txt:      Standard DETR top-1 (baseline)
  2. CSeg-Raw:     CLIPSeg binary mask directly (no SAM3 at all)
  3. CSeg-Refine:  CLIPSeg mask → SAM3 mask decoder (mask-only, no points)
  4. CSeg-Pts:     CLIPSeg mask + points → SAM3 mask decoder
  5. CSeg-Multi:   CLIPSeg mask + points → SAM3 multimask → pick best by IoU

Usage (VS Code "Run Python File"):
  python qwen2sam/scripts/test_bypass_detr.py
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_parent_root = _project_root.parent
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
#  Prompts                                                                #
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

DEFAULT_SAMPLES = None # ["13", "22", "44", "47", "48", "59", "107"]


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
    pl = np.zeros(pred_a.shape, dtype=np.int32)
    pl[pred_a > 0.5] = 1
    pl[pred_b > 0.5] = 2
    gl = np.zeros(gt_a.shape, dtype=np.int32)
    gl[gt_a > 0.5] = 1
    gl[gt_b > 0.5] = 2
    return float(adjusted_rand_score(gl.ravel(), pl.ravel()))


def compute_sample_metrics(pred_a, pred_b, gt_a, gt_b, name):
    d = (compute_iou(pred_a, gt_a) + compute_iou(pred_b, gt_b)) / 2.0
    s = (compute_iou(pred_a, gt_b) + compute_iou(pred_b, gt_a)) / 2.0
    if s > d:
        pred_a, pred_b = pred_b, pred_a
    ia = compute_iou(pred_a, gt_a)
    ib = compute_iou(pred_b, gt_b)
    return {
        "crop_name": name, "iou_a": ia, "iou_b": ib,
        "mean_iou": (ia + ib) / 2.0,
        "mean_dice": (compute_dice(pred_a, gt_a) + compute_dice(pred_b, gt_b)) / 2.0,
        "ari": compute_ari(pred_a, pred_b, gt_a, gt_b),
    }


def aggregate_metrics(mets, tag):
    return {
        "tag": tag, "num_samples": len(mets),
        "mean_iou": float(np.mean([m["mean_iou"] for m in mets])),
        "mean_dice": float(np.mean([m["mean_dice"] for m in mets])),
        "mean_ari": float(np.nanmean([m["ari"] for m in mets])),
    }


# ===================================================================== #
#  Visualization                                                          #
# ===================================================================== #

COLOR_A = (0, 0, 220)
COLOR_B = (220, 80, 0)


def mask_overlay(image, ma, mb, alpha=0.45):
    vis = image.copy()
    ov = image.copy()
    ov[ma > 0.5] = COLOR_A
    ov[mb > 0.5] = COLOR_B
    return cv2.addWeighted(ov, alpha, vis, 1 - alpha, 0)


def binary_mask_image(ma, mb, h, w):
    c = np.zeros((h, w, 3), dtype=np.uint8)
    c[ma > 0.5] = COLOR_A
    c[mb > 0.5] = COLOR_B
    return c


def heatmap_to_bgr(hm, size=None):
    h = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
    c = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    if size:
        c = cv2.resize(c, size, interpolation=cv2.INTER_LINEAR)
    return c


def enforce_mutual_exclusion(ma, mb, diff_a=None, diff_b=None):
    ba, bb = ma > 0.5, mb > 0.5
    ov = ba & bb
    gap = ~ba & ~bb
    ra, rb = ba.copy(), bb.copy()
    if ov.any():
        if diff_a is not None and diff_b is not None:
            h, w = ma.shape
            da = cv2.resize(diff_a, (w, h)) if diff_a.shape != (h, w) else diff_a
            db = cv2.resize(diff_b, (w, h)) if diff_b.shape != (h, w) else diff_b
            ra[ov & (da < db)] = False
            rb[ov & (da >= db)] = False
        else:
            rb[ov] = False
    if gap.any():
        if diff_a is not None and diff_b is not None:
            h, w = ma.shape
            da = cv2.resize(diff_a, (w, h)) if diff_a.shape != (h, w) else diff_a
            db = cv2.resize(diff_b, (w, h)) if diff_b.shape != (h, w) else diff_b
            ra[gap & (da >= db)] = True
            rb[gap & (da < db)] = True
        else:
            ra[gap] = True
    return ra.astype(np.float32), rb.astype(np.float32)


# ===================================================================== #
#  SAM3 helpers                                                           #
# ===================================================================== #

@torch.no_grad()
def run_detr_with_text(model, backbone_out, text_feat, B, device):
    prompt_bf = text_feat["prompt"].unsqueeze(0).expand(B, -1, -1)
    mask_bf = text_feat["mask"].unsqueeze(0).expand(B, -1)
    out = model.base._run_sam3_from_backbone(backbone_out, prompt_bf, mask_bf)
    scores = out["pred_logits"][0].squeeze(-1).sigmoid()
    best = scores.argmax().item()
    return out["pred_masks"][0, best]


@torch.no_grad()
def run_mask_decoder(model, image_embed, high_res_feats, coarse_mask,
                     pos_points=None, neg_points=None, multimask=False, device="cuda"):
    """
    Run SAM3's mask decoder with a coarse mask prompt (and optional points).
    Bypasses DETR entirely — uses SAM3 only as boundary refinement.
    """
    B = image_embed.shape[0]

    # Prepare mask
    coarse = coarse_mask
    if coarse.ndim == 2:
        coarse = coarse.unsqueeze(0).unsqueeze(0)
    elif coarse.ndim == 3:
        coarse = coarse.unsqueeze(0)

    mask_input_size = model.sam_prompt_encoder.mask_input_size
    if coarse.shape[-2:] != mask_input_size:
        sam_mask = F.interpolate(
            coarse.float(), size=mask_input_size,
            mode="bilinear", align_corners=False, antialias=True)
    else:
        sam_mask = coarse.float()

    # Prepare points (optional)
    if pos_points is not None and neg_points is not None:
        all_coords = torch.cat([pos_points, neg_points], dim=1)
        pos_labels = torch.ones(B, pos_points.shape[1], dtype=torch.int32, device=device)
        neg_labels = torch.zeros(B, neg_points.shape[1], dtype=torch.int32, device=device)
        all_labels = torch.cat([pos_labels, neg_labels], dim=1)
        points = (all_coords, all_labels)
    else:
        # No points — mask-only prompt
        points = None

    sparse_emb, dense_emb = model.sam_prompt_encoder(
        points=points, boxes=None, masks=sam_mask)

    image_pe = model.sam_prompt_encoder.get_dense_pe()

    masks_out, ious, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=multimask,
        repeat_image=False,
        high_res_features=high_res_feats,
    )
    return masks_out.squeeze(0), ious.squeeze(0)  # (N, H, W), (N,)


def postprocess_mask(mask_logit, gt_h, gt_w):
    if mask_logit.shape[-2:] != (gt_h, gt_w):
        mask_logit = F.interpolate(
            mask_logit[None, None].float() if mask_logit.ndim == 2
            else mask_logit.unsqueeze(0).float(),
            size=(gt_h, gt_w),
            mode="bilinear", align_corners=False,
        ).squeeze()
    return (mask_logit.sigmoid().cpu().numpy() > 0.5).astype(np.float32)


def scale_points(pts, orig_size, target_size):
    s = target_size / orig_size
    return [[p[0] * s, p[1] * s] for p in pts]


# ===================================================================== #
#  CLIPSeg                                                                #
# ===================================================================== #

@torch.no_grad()
def clipseg_extract(clipseg_model, clipseg_proc, image_pil,
                     desc_a, desc_b, img_size=256, device="cuda",
                     n_points=4, erode_iter=3, top_pct=0.3):
    """Get CLIPSeg diff maps, points, and binary masks."""
    raw = []
    for desc in [desc_a, desc_b]:
        inp = clipseg_proc(text=[desc], images=[image_pil],
                            return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.amp.autocast("cuda", enabled=False):
            out = clipseg_model.float()(**inp)
        hm = torch.sigmoid(out.logits.squeeze().float())
        hm = F.interpolate(hm[None, None], size=(img_size, img_size),
                           mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        raw.append(hm)

    diff_a = np.clip(raw[0] - raw[1], 0, 1)
    diff_b = np.clip(raw[1] - raw[0], 0, 1)

    # Points extraction
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    all_pts = []
    for dm in [diff_a, diff_b]:
        thresh = np.percentile(dm, (1.0 - top_pct) * 100)
        binary = (dm >= thresh).astype(np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=erode_iter)
        for fb in [erode_iter - 1, 1, 0]:
            if eroded.sum() >= n_points:
                break
            eroded = cv2.erode(binary, kernel, iterations=max(fb, 0))
        if eroded.sum() < n_points:
            eroded = binary
        ys, xs = np.where(eroded > 0)
        if len(ys) < n_points:
            ys, xs = np.where(binary > 0)
        if len(ys) < n_points:
            ys, xs = np.where(dm > np.median(dm))
        weights = dm[ys, xs]
        weights = weights / (weights.sum() + 1e-8)
        pts = []
        avail = list(range(len(ys)))
        for _ in range(min(n_points, len(avail))):
            if not avail:
                break
            if not pts:
                idx = avail[np.argmax(weights[avail])]
            else:
                sc = np.zeros(len(avail))
                for ai, av in enumerate(avail):
                    w = weights[av]
                    md = min(np.sqrt((xs[av]-p[0])**2 + (ys[av]-p[1])**2) for p in pts)
                    sc[ai] = w * min(md / img_size, 1.0)
                idx = avail[np.argmax(sc)]
            pts.append([int(xs[idx]), int(ys[idx])])
            avail.remove(idx)
        all_pts.append(pts)

    # Binary masks in logit scale
    bin_a = (diff_a > 0.1).astype(np.float32) * 10.0 - 5.0
    bin_b = (diff_b > 0.1).astype(np.float32) * 10.0 - 5.0

    return all_pts[0], all_pts[1], diff_a, diff_b, raw[0], raw[1], bin_a, bin_b


# ===================================================================== #
#  Model loaders                                                          #
# ===================================================================== #

def load_qwen3(device):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor as AP
    name = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"Loading {name}...")
    m = Qwen3VLForConditionalGeneration.from_pretrained(
        name, torch_dtype=torch.bfloat16).to(device).eval()
    p = AP.from_pretrained(name)
    print("  Qwen3 loaded")
    return m, p


def load_clipseg(device):
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    print("Loading CLIPSeg...")
    p = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    m = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined").to(device).eval()
    print("  CLIPSeg loaded")
    return m, p


@torch.no_grad()
def qwen3_generate(model, proc, image_pil, device):
    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": QWEN_USER_PROMPT},
        ]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image_pil], return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False, temperature=1.0)
    gen = out[0, inputs["input_ids"].shape[1]:]
    return proc.tokenizer.decode(gen, skip_special_tokens=True)


def parse_text(text):
    da = db = ""
    ma = re.search(r'TEXTURE_A:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    mb = re.search(r'TEXTURE_B:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if ma: da = ma.group(1).strip()
    if mb: db = mb.group(1).strip()
    return da, db, bool(da and db)


# ===================================================================== #
#  Visualization                                                          #
# ===================================================================== #

def draw_grid(image_bgr, gt_a, gt_b, preds, diff_maps,
              cseg_binary_masks, q_desc_a, q_desc_b, title, cell_size=256):
    h, w = image_bgr.shape[:2]
    s = cell_size / max(h, w)
    ch, cw = int(h * s), int(w * s)

    def ri(img): return cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)
    def rm(m): return cv2.resize(m.astype(np.float32), (cw, ch), interpolation=cv2.INTER_NEAREST)

    img = ri(image_bgr)
    ga, gb = rm(gt_a), rm(gt_b)
    diff_a, diff_b = diff_maps
    d_max = max(diff_a.max(), diff_b.max(), 1e-8)
    bin_a, bin_b = cseg_binary_masks

    # Columns: Image | Diff | CSeg Mask | GT | predictions...
    # Row 1: overlay, Row 2: binary, Row 3: input mask to method
    col_img = (img, np.zeros((ch, cw, 3), dtype=np.uint8),
               np.zeros((ch, cw, 3), dtype=np.uint8))
    col_diff = (heatmap_to_bgr(diff_a / d_max, (cw, ch)),
                heatmap_to_bgr(diff_b / d_max, (cw, ch)),
                np.zeros((ch, cw, 3), dtype=np.uint8))
    # CSeg binary mask visualization
    cseg_a_vis = heatmap_to_bgr((bin_a + 5) / 10, (cw, ch))
    cseg_b_vis = heatmap_to_bgr((bin_b + 5) / 10, (cw, ch))
    col_cseg = (cseg_a_vis, cseg_b_vis,
                np.zeros((ch, cw, 3), dtype=np.uint8))

    gt_ov = mask_overlay(img, ga, gb)
    gt_bin = binary_mask_image(ga, gb, ch, cw)
    col_gt = (gt_ov, gt_bin, np.zeros((ch, cw, 3), dtype=np.uint8))

    cols = [col_img, col_diff, col_cseg, col_gt]
    col_labels = [title, "Diff A|B", "CSeg Mask", "GT"]

    for label, (ma, mb, met) in preds.items():
        ov = mask_overlay(img, rm(ma), rm(mb))
        bn = binary_mask_image(rm(ma), rm(mb), ch, cw)
        cols.append((ov, bn, np.zeros((ch, cw, 3), dtype=np.uint8)))
        col_labels.append(f"{label} {met['mean_iou']:.3f}")

    sep = 2
    header_h = 32
    desc_h = 40

    rows = []
    for ri_idx in range(2):  # just 2 rows: overlay + binary
        cells = []
        for col in cols:
            cells.extend([col[ri_idx], np.zeros((ch, sep, 3), dtype=np.uint8)])
        rows.append(np.hstack(cells[:-1]))

    actual_w = rows[0].shape[1]
    bar = np.zeros((header_h, actual_w, 3), dtype=np.uint8) + 30
    x = 0
    for lbl in col_labels:
        cv2.putText(bar, lbl, (x + 4, header_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)
        x += cw + sep

    desc_bar = np.zeros((desc_h, actual_w, 3), dtype=np.uint8) + 20
    cv2.putText(desc_bar, f"A: {q_desc_a}"[:actual_w // 5], (8, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(desc_bar, f"B: {q_desc_b}"[:actual_w // 5], (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 150, 0), 1, cv2.LINE_AA)

    row_sep = np.zeros((sep, actual_w, 3), dtype=np.uint8)
    grid_rows = []
    for r in rows:
        grid_rows.extend([r, row_sep])

    return np.vstack([bar, desc_bar] + grid_rows[:-1])


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Bypass DETR test")
    parser.add_argument("--config", type=str,
                        default="qwen2sam/configs/v3_tracker_detexure.yaml")
    parser.add_argument("--data_root", type=str, default="/home/aviad/RWTD")
    parser.add_argument("--samples", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/bypass_detr")
    args = parser.parse_args()

    sample_names = args.samples.split(",") if args.samples else DEFAULT_SAMPLES

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _project_root / config_path
    cfg = load_config(str(config_path))
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg["model"].get("image_size", 1008)

    meta = json.load(open(Path(args.data_root) / "metadata_phase1.json"))
    meta_by_name = {e["crop_name"]: e for e in meta}
    if sample_names is None:
        samples = meta  # run all images
    else:
        samples = [meta_by_name[n] for n in sample_names if n in meta_by_name]
    print(f"Testing {len(samples)} samples")

    output_dir = _project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    if "tracker" in cfg:
        cfg["tracker"].pop("v3_checkpoint", None)
    print("Loading SAM3...")
    model = Qwen2SAMv3Tracker(cfg, device=str(device))
    model.base.sam3.eval()
    model.sam_prompt_encoder.eval()
    model.sam_mask_decoder.eval()

    qwen3, qwen3_proc = load_qwen3(device)
    cseg_model, cseg_proc = load_clipseg(device)

    methods = ["Qw3Txt", "CSeg-Raw", "CSeg-Refine", "CSeg-Pts", "CSeg-Multi"]
    all_metrics = {m: [] for m in methods}
    trunk_cache = {}

    def _trunk_hook(module, inp, out):
        trunk_cache["xs"] = out

    t0 = time.time()

    for entry in samples:
        crop_name = entry["crop_name"]
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

            # Qwen3 text
            raw = qwen3_generate(qwen3, qwen3_proc, image_pil, device)
            q_desc_a, q_desc_b, ok = parse_text(raw)

            if not ok:
                print(f"  [{crop_name}] PARSE FAIL")
                hook.remove()
                zero = np.zeros_like(gt_a)
                met = compute_sample_metrics(zero, zero, gt_a, gt_b, crop_name)
                for m in methods:
                    all_metrics[m].append(met)
                continue

            # === 1. Qw3Txt: DETR baseline ===
            t_a = model.base.sam3.backbone.forward_text([q_desc_a], device=device)
            t_b = model.base.sam3.backbone.forward_text([q_desc_b], device=device)
            feat_a = {"prompt": t_a["language_features"].squeeze(1),
                      "mask": t_a["language_mask"].squeeze(0)}
            feat_b = {"prompt": t_b["language_features"].squeeze(1),
                      "mask": t_b["language_mask"].squeeze(0)}
            detr_a = run_detr_with_text(model, backbone_out, feat_a, 1, device)
            detr_b = run_detr_with_text(model, backbone_out, feat_b, 1, device)
            txt_a = postprocess_mask(detr_a, gt_h, gt_w)
            txt_b = postprocess_mask(detr_b, gt_h, gt_w)

            hook.remove()
            trunk_output = trunk_cache["xs"][-1]

            # CLIPSeg extraction
            (cseg_pts_a, cseg_pts_b, diff_a, diff_b,
             heat_a, heat_b, bin_a, bin_b) = clipseg_extract(
                cseg_model, cseg_proc, image_pil,
                q_desc_a, q_desc_b, img_size=orig_w, device=device)

            # Get SAM3 image features (bypassing DETR)
            image_embed, high_res_feats = model._get_sam2_features(trunk_output)

            # Coarse masks as tensors
            coarse_a = torch.from_numpy(bin_a).float().to(device)
            coarse_b = torch.from_numpy(bin_b).float().to(device)

            # Points
            pts_a_s = scale_points(cseg_pts_a, orig_w, image_size)
            pts_b_s = scale_points(cseg_pts_b, orig_w, image_size)
            pos_a = torch.tensor(pts_a_s, dtype=torch.float32, device=device).unsqueeze(0)
            pos_b = torch.tensor(pts_b_s, dtype=torch.float32, device=device).unsqueeze(0)

            # === 2. CSeg-Raw: CLIPSeg binary directly (no SAM3) ===
            raw_a = cv2.resize((diff_a > 0.1).astype(np.float32), (gt_w, gt_h),
                               interpolation=cv2.INTER_NEAREST)
            raw_b = cv2.resize((diff_b > 0.1).astype(np.float32), (gt_w, gt_h),
                               interpolation=cv2.INTER_NEAREST)

            # === 3. CSeg-Refine: mask only, no points ===
            ref_a, _ = run_mask_decoder(model, image_embed, high_res_feats,
                                        coarse_a, device=device)
            ref_b, _ = run_mask_decoder(model, image_embed, high_res_feats,
                                        coarse_b, device=device)
            refine_a = postprocess_mask(ref_a[0], gt_h, gt_w)
            refine_b = postprocess_mask(ref_b[0], gt_h, gt_w)

            # === 4. CSeg-Pts: mask + points ===
            pts_a_out, _ = run_mask_decoder(model, image_embed, high_res_feats,
                                            coarse_a, pos_a, pos_b, device=device)
            pts_b_out, _ = run_mask_decoder(model, image_embed, high_res_feats,
                                            coarse_b, pos_b, pos_a, device=device)
            pts_ref_a = postprocess_mask(pts_a_out[0], gt_h, gt_w)
            pts_ref_b = postprocess_mask(pts_b_out[0], gt_h, gt_w)

            # === 5. CSeg-Multi: mask + points + multimask, pick best by IoU ===
            multi_a, ious_a = run_mask_decoder(model, image_embed, high_res_feats,
                                               coarse_a, pos_a, pos_b,
                                               multimask=True, device=device)
            multi_b, ious_b = run_mask_decoder(model, image_embed, high_res_feats,
                                               coarse_b, pos_b, pos_a,
                                               multimask=True, device=device)
            best_a = ious_a.float().argmax().item()
            best_b = ious_b.float().argmax().item()
            multi_ref_a = postprocess_mask(multi_a[best_a], gt_h, gt_w)
            multi_ref_b = postprocess_mask(multi_b[best_b], gt_h, gt_w)

        # Enforce mutual exclusion on all
        txt_a, txt_b = enforce_mutual_exclusion(txt_a, txt_b, diff_a, diff_b)
        raw_a, raw_b = enforce_mutual_exclusion(raw_a, raw_b, diff_a, diff_b)
        refine_a, refine_b = enforce_mutual_exclusion(refine_a, refine_b, diff_a, diff_b)
        pts_ref_a, pts_ref_b = enforce_mutual_exclusion(pts_ref_a, pts_ref_b, diff_a, diff_b)
        multi_ref_a, multi_ref_b = enforce_mutual_exclusion(multi_ref_a, multi_ref_b, diff_a, diff_b)

        # Metrics
        results = {
            "Qw3Txt": (txt_a, txt_b),
            "CSeg-Raw": (raw_a, raw_b),
            "CSeg-Refine": (refine_a, refine_b),
            "CSeg-Pts": (pts_ref_a, pts_ref_b),
            "CSeg-Multi": (multi_ref_a, multi_ref_b),
        }

        preds_vis = {}
        for method, (ma, mb) in results.items():
            met = compute_sample_metrics(ma, mb, gt_a, gt_b, crop_name)
            all_metrics[method].append(met)
            preds_vis[method] = (ma, mb, met)

        # Print
        line = " | ".join(f"{m}: {preds_vis[m][2]['mean_iou']:.3f}" for m in methods)
        print(f"  [{crop_name}] {line}")
        print(f"    A: \"{q_desc_a}\"")
        print(f"    B: \"{q_desc_b}\"")
        if multi_a.shape[0] == 3:
            print(f"    Multi IoU scores A: {ious_a.float().cpu().numpy()} pick={best_a}")
            print(f"    Multi IoU scores B: {ious_b.float().cpu().numpy()} pick={best_b}")

        # Visualization
        grid = draw_grid(
            image_bgr, gt_a, gt_b, preds_vis,
            diff_maps=(diff_a, diff_b),
            cseg_binary_masks=(bin_a, bin_b),
            q_desc_a=q_desc_a, q_desc_b=q_desc_b,
            title=crop_name,
        )
        cv2.imwrite(str(vis_dir / f"{crop_name}_bypass.png"), grid)

    elapsed = time.time() - t0

    summaries = {m: aggregate_metrics(all_metrics[m], m) for m in methods}

    print(f"\n{'='*100}")
    print(f"  Bypass DETR — {len(samples)} samples ({elapsed:.1f}s)")
    print(f"{'='*100}")
    print(f"  {'':15s}" + "".join(f"{m:>14s}" for m in methods))
    print(f"  {'─'*85}")
    for label, key in [("mIoU", "mean_iou"), ("mDice", "mean_dice"), ("mARI", "mean_ari")]:
        vals = "".join(f"{summaries[m][key]:14.4f}" for m in methods)
        print(f"  {label:15s}{vals}")
    print(f"{'='*100}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*100}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
