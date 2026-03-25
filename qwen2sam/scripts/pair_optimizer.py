"""
Pairing Optimization Algorithm for CLIPSeg heatmap selection.

Given two lists of heatmaps (one per texture), this module:
  1. Filters out dead/collapsed maps (sanity check)
  2. Scores all valid NxM pairs on quality + minimal overlap
  3. Visualizes the winning pair

Standalone module — import and call, or run directly as a test.

Usage:
  # As a module:
  from pair_optimizer import filter_valid_heatmaps, find_best_pair, visualize_pairing_test

  # As a standalone test on the diversity output:
  python qwen2sam/scripts/pair_optimizer.py
  python qwen2sam/scripts/pair_optimizer.py --samples 3,12,13,15,18,22
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_parent_root = _project_root.parent
if str(_parent_root) not in sys.path:
    sys.path.insert(0, str(_parent_root))


# ===================================================================== #
#  1. Sanity Check Filter                                                 #
# ===================================================================== #

def filter_valid_heatmaps(heatmaps, descriptions,
                          min_std=0.05, min_coverage=0.10, coverage_thresh=0.5):
    """
    Filter heatmaps by two hard gates.

    Args:
        heatmaps: list of numpy arrays, each normalized to [0, 1]
        descriptions: list of str, same length as heatmaps
        min_std: drop if std < this (dead map gate)
        min_coverage: drop if fraction of pixels > coverage_thresh is below this
        coverage_thresh: threshold for the salience mask

    Returns:
        valid_heatmaps: list of numpy arrays that passed both gates
        valid_descriptions: corresponding descriptions
        filter_log: list of dicts with per-map stats and pass/fail reason
    """
    valid_heatmaps = []
    valid_descriptions = []
    filter_log = []

    for i, (hm, desc) in enumerate(zip(heatmaps, descriptions)):
        std = float(np.std(hm))
        salience_mask = hm > coverage_thresh
        coverage = float(salience_mask.sum() / hm.size)

        entry = {
            "index": i,
            "description": desc,
            "std": std,
            "coverage": coverage,
            "passed": True,
            "reject_reason": None,
        }

        if std < min_std:
            entry["passed"] = False
            entry["reject_reason"] = f"dead_map (std={std:.4f} < {min_std})"
        elif coverage < min_coverage:
            entry["passed"] = False
            entry["reject_reason"] = f"low_salience (coverage={coverage:.3f} < {min_coverage})"
        else:
            valid_heatmaps.append(hm)
            valid_descriptions.append(desc)

        filter_log.append(entry)

    return valid_heatmaps, valid_descriptions, filter_log


# ===================================================================== #
#  2. Scoring Matrix                                                      #
# ===================================================================== #

def _quality(hm, top_pct=0.10):
    """Mean value of the top `top_pct` hottest pixels."""
    flat = hm.ravel()
    k = max(1, int(len(flat) * top_pct))
    top_vals = np.partition(flat, -k)[-k:]
    return float(np.mean(top_vals))


def find_best_pair(valid_A, valid_B, desc_A, desc_B, gamma=2.5):
    """
    Score all NxM pairs and return sorted list.

    Score = Quality(A_i) + Quality(B_j) - gamma * Overlap(A_i, B_j)

    Args:
        valid_A: list of numpy heatmaps for texture A
        valid_B: list of numpy heatmaps for texture B
        desc_A: descriptions for A
        desc_B: descriptions for B
        gamma: penalty weight for overlap

    Returns:
        sorted_pairs: list of dicts sorted by score (highest first), each with:
            idx_a, idx_b, desc_a, desc_b, quality_a, quality_b, overlap, score
    """
    pairs = []

    for i, (hm_a, da) in enumerate(zip(valid_A, desc_A)):
        qa = _quality(hm_a)
        for j, (hm_b, db) in enumerate(zip(valid_B, desc_B)):
            qb = _quality(hm_b)
            overlap = float(np.mean(hm_a * hm_b))
            score = qa + qb - gamma * overlap

            pairs.append({
                "idx_a": i,
                "idx_b": j,
                "desc_a": da,
                "desc_b": db,
                "quality_a": qa,
                "quality_b": qb,
                "overlap": overlap,
                "score": score,
            })

    pairs.sort(key=lambda p: p["score"], reverse=True)
    return pairs


# ===================================================================== #
#  3. Visualization                                                       #
# ===================================================================== #

def visualize_pairing_test(image_rgb, heatmaps_A, heatmaps_B, desc_A, desc_B,
                           gt_a=None, gt_b=None, gamma=2.5, save_path=None,
                           crop_name="", skip_vis=False):
    """
    Run full pipeline and optionally produce matplotlib visualization.

    Args:
        image_rgb: numpy HxWx3 RGB image
        heatmaps_A: list of numpy heatmaps for texture A
        heatmaps_B: list of numpy heatmaps for texture B
        desc_A: list of 5 descriptions for A
        desc_B: list of 5 descriptions for B
        gt_a, gt_b: optional ground truth masks for IoU display
        gamma: overlap penalty
        save_path: if set, save figure here
        crop_name: sample name for title

    Returns:
        best_pair: the winning pair dict
        sorted_pairs: all pairs sorted
        filter_log_a, filter_log_b: filter results
    """
    # --- Filter ---
    valid_A, vdesc_A, flog_a = filter_valid_heatmaps(heatmaps_A, desc_A)
    valid_B, vdesc_B, flog_b = filter_valid_heatmaps(heatmaps_B, desc_B)

    n_valid_a = len(valid_A)
    n_valid_b = len(valid_B)

    if n_valid_a == 0 or n_valid_b == 0:
        print(f"  WARNING: Not enough valid heatmaps (A={n_valid_a}, B={n_valid_b})")
        return None, [], flog_a, flog_b

    # --- Score ---
    sorted_pairs = find_best_pair(valid_A, valid_B, vdesc_A, vdesc_B, gamma=gamma)
    best = sorted_pairs[0]

    if skip_vis:
        return best, sorted_pairs, flog_a, flog_b

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    win_a = valid_A[best["idx_a"]]
    win_b = valid_B[best["idx_b"]]

    # Map winning valid-index back to original heatmap index
    valid_indices_a = [fl["index"] for fl in flog_a if fl["passed"]]
    valid_indices_b = [fl["index"] for fl in flog_b if fl["passed"]]
    winner_orig_a = valid_indices_a[best["idx_a"]]
    winner_orig_b = valid_indices_b[best["idx_b"]]

    # Resize heatmaps to image size for overlay
    h, w = image_rgb.shape[:2]
    win_a_full = cv2.resize(win_a, (w, h), interpolation=cv2.INTER_LINEAR)
    win_b_full = cv2.resize(win_b, (w, h), interpolation=cv2.INTER_LINEAR)

    # --- Build figure ---
    # Layout: 2 rows x 4 cols
    # Row 1: Original | All A heatmaps (valid/rejected) | Winner A | Winner B
    # Row 2: GT(opt)  | Score matrix heatmap             | Boundary overlay | Combined RB
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f"Pair Optimizer — {crop_name}    |    "
                 f"Valid: A={n_valid_a}/{len(heatmaps_A)}, B={n_valid_b}/{len(heatmaps_B)}    |    "
                 f"Best Score: {best['score']:.3f}",
                 fontsize=13, fontweight="bold")

    # (0,0) Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original Image", fontsize=10)
    axes[0, 0].axis("off")

    # (0,1) Filter summary — show all A heatmaps, winner marked with red border
    _draw_filter_summary(axes[0, 1], heatmaps_A, desc_A, flog_a, "Texture A",
                         winner_idx=winner_orig_a)

    # (0,2) Winning heatmap A
    axes[0, 2].imshow(win_a_full, cmap="jet", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Winner A (Q={best['quality_a']:.3f})\n\"{best['desc_a'][:50]}\"",
                         fontsize=8, color="red")
    axes[0, 2].axis("off")

    # (0,3) Winning heatmap B
    axes[0, 3].imshow(win_b_full, cmap="jet", vmin=0, vmax=1)
    axes[0, 3].set_title(f"Winner B (Q={best['quality_b']:.3f})\n\"{best['desc_b'][:50]}\"",
                         fontsize=8, color="blue")
    axes[0, 3].axis("off")

    # (1,0) GT masks — same color scheme as other testers
    # COLOR_A = blue (RGB 0,0,220), COLOR_B = orange (RGB 220,80,0)
    if gt_a is not None and gt_b is not None:
        gt_a_r = cv2.resize(gt_a, (w, h), interpolation=cv2.INTER_NEAREST)
        gt_b_r = cv2.resize(gt_b, (w, h), interpolation=cv2.INTER_NEAREST)
        gt_overlay = (image_rgb.astype(np.float32) / 255.0).copy()
        color_a = np.array([0, 0, 220]) / 255.0   # blue
        color_b = np.array([220, 80, 0]) / 255.0   # orange
        gt_overlay[gt_a_r > 0.5] = gt_overlay[gt_a_r > 0.5] * 0.55 + color_a * 0.45
        gt_overlay[gt_b_r > 0.5] = gt_overlay[gt_b_r > 0.5] * 0.55 + color_b * 0.45
        axes[1, 0].imshow(np.clip(gt_overlay, 0, 1))
        axes[1, 0].set_title("GT (Blue=A, Orange=B)", fontsize=10)
    else:
        axes[1, 0].text(0.5, 0.5, "No GT", ha="center", va="center",
                        fontsize=14, color="gray", transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("Ground Truth", fontsize=10)
    axes[1, 0].axis("off")

    # (1,1) Filter summary B, winner marked with red border
    _draw_filter_summary(axes[1, 1], heatmaps_B, desc_B, flog_b, "Texture B",
                         winner_idx=winner_orig_b)

    # (1,2) Score matrix heatmap
    if n_valid_a > 0 and n_valid_b > 0:
        score_matrix = np.zeros((n_valid_a, n_valid_b))
        for p in sorted_pairs:
            score_matrix[p["idx_a"], p["idx_b"]] = p["score"]
        im = axes[1, 2].imshow(score_matrix, cmap="viridis", aspect="auto")
        axes[1, 2].set_xlabel("B index", fontsize=8)
        axes[1, 2].set_ylabel("A index", fontsize=8)
        axes[1, 2].set_title(f"Score Matrix (γ={gamma})\nBest: A[{best['idx_a']}]×B[{best['idx_b']}]",
                             fontsize=9)
        # Mark best
        axes[1, 2].plot(best["idx_b"], best["idx_a"], "r*", markersize=15)
        fig.colorbar(im, ax=axes[1, 2], fraction=0.046)
    else:
        axes[1, 2].axis("off")

    # (1,3) Combined overlay: WTA binary masks on image, same colors as other testers
    wta_a_vis = (win_a_full > win_b_full).astype(np.float32)
    wta_b_vis = (win_b_full > win_a_full).astype(np.float32)
    color_a = np.array([0, 0, 220]) / 255.0   # blue
    color_b = np.array([220, 80, 0]) / 255.0   # orange
    combined = (image_rgb.astype(np.float32) / 255.0).copy()
    combined[wta_a_vis > 0.5] = combined[wta_a_vis > 0.5] * 0.55 + color_a * 0.45
    combined[wta_b_vis > 0.5] = combined[wta_b_vis > 0.5] * 0.55 + color_b * 0.45

    axes[1, 3].imshow(np.clip(combined, 0, 1))
    axes[1, 3].set_title(f"WTA Overlay (Blue=A, Orange=B)\n"
                         f"Overlap={best['overlap']:.3f}  Score={best['score']:.3f}",
                         fontsize=9)
    axes[1, 3].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)

    return best, sorted_pairs, flog_a, flog_b


def _draw_filter_summary(ax, heatmaps, descriptions, filter_log, title,
                         winner_idx=None):
    """Draw a mini summary of all heatmaps with pass/fail and winner marking."""
    n = len(heatmaps)
    cell_h, cell_w = 40, 40
    border = 3
    padded_h = cell_h + 2 * border
    padded_w = cell_w + 2 * border
    gap = 2
    strip = np.zeros((padded_h, padded_w * n + gap * (n - 1), 3), dtype=np.uint8)

    for i, (hm, fl) in enumerate(zip(heatmaps, filter_log)):
        mini = cv2.resize(hm, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
        mini_color = cv2.applyColorMap((mini * 255).astype(np.uint8), cv2.COLORMAP_JET)
        mini_color = cv2.cvtColor(mini_color, cv2.COLOR_BGR2RGB)

        # Create padded cell with border
        cell = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)

        if i == winner_idx:
            # WINNER: thick red border
            cell[:, :] = [255, 0, 0]
        elif not fl["passed"]:
            # Rejected: dark gray border with X pattern
            cell[:, :] = [80, 80, 80]
        else:
            # Valid but not chosen: black border
            cell[:, :] = [0, 0, 0]

        cell[border:border + cell_h, border:border + cell_w] = mini_color

        # Dim rejected maps
        if not fl["passed"]:
            cell[border:border + cell_h, border:border + cell_w] = (
                cell[border:border + cell_h, border:border + cell_w] * 0.4
            ).astype(np.uint8)

        x_start = i * (padded_w + gap)
        strip[:, x_start:x_start + padded_w] = cell

    ax.imshow(strip)
    ax.set_title(title, fontsize=10)

    # Add text annotations below — mark winner with ★
    txt_parts = []
    for i, fl in enumerate(filter_log):
        if i == winner_idx:
            status = "★"
        elif fl["passed"]:
            status = "✓"
        else:
            status = "✗"
        txt_parts.append(f"{status} std={fl['std']:.3f} cov={fl['coverage']:.2f}")
    summary_text = "\n".join(txt_parts)
    ax.text(0.0, -0.15, summary_text, transform=ax.transAxes,
            fontsize=6, verticalalignment="top", fontfamily="monospace")
    ax.axis("off")


# ===================================================================== #
#  Standalone test: generate heatmaps + run optimizer                     #
# ===================================================================== #

DEFAULT_SAMPLES = None #["3", "12", "13", "15", "18", "22"]

# --- Single-description prompt (for QwenTxt baseline, same as other testers) ---
QWEN_SINGLE_PROMPT = (
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
    "TEXTURE_B: Texture of <description>"
)

QWEN_SYSTEM_PROMPT = (
    "You analyze surface textures in images. Always respond in the exact "
    "format requested, with no extra text."
)

QWEN_DIVERSE_PROMPT = (
    "This image contains exactly TWO main visually distinct regions separated by a boundary.\n\n"
    "For each region, provide 5 DIFFERENT descriptions. Each description must be a single "
    "phrase of 10-15 words that captures the region, but each one MUST use a COMPLETELY DIFFERENT "
    "angle or vocabulary:\n"
    "  1. Focus on COLOR and TONE\n"
    "  2. Focus on TEXTURE and PATTERN\n"
    "  3. Focus on MATERIAL and SURFACE type\n"
    "  4. Focus on SPATIAL CONTEXT and SHAPE\n"
    "  5. Use EVERYDAY LANGUAGE (how a non-expert would describe it)\n\n"
    "IMPORTANT: Each description must be genuinely different — not just rephrasing the same words. "
    "Use diverse vocabulary. Describe the ENTIRE region as a surface/area.\n\n"
    "Format your response exactly like this:\n"
    "TEXTURE_A_1: Texture of <color/tone>\n"
    "TEXTURE_A_2: Texture of <texture/pattern>\n"
    "TEXTURE_A_3: Texture of <material/surface>\n"
    "TEXTURE_A_4: Texture of <spatial/shape>\n"
    "TEXTURE_A_5: Texture of <everyday language>\n"
    "TEXTURE_B_1: Texture of <color/tone>\n"
    "TEXTURE_B_2: Texture of <texture/pattern>\n"
    "TEXTURE_B_3: Texture of <material/surface>\n"
    "TEXTURE_B_4: Texture of <spatial/shape>\n"
    "TEXTURE_B_5: Texture of <everyday language>"
)

N_DESCRIPTIONS = 5
DESC_LABELS = ["Color/Tone", "Texture/Pattern", "Material/Surface",
               "Spatial/Shape", "Everyday"]


def load_qwen3_model(device):
    import torch
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
    import torch
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    print("Loading CLIPSeg...")
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined"
    ).to(device).eval()
    print("  CLIPSeg loaded")
    return model, proc


def qwen3_generate_diverse(model, processor, image_pil, device):
    import torch
    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": QWEN_DIVERSE_PROMPT},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image_pil], return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=800, do_sample=False, temperature=1.0)
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, input_len:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True)


def parse_diverse_output(text, n_desc=N_DESCRIPTIONS):
    descs_a, descs_b = [], []
    for i in range(1, n_desc + 1):
        ma = re.search(rf'TEXTURE_A_{i}:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        mb = re.search(rf'TEXTURE_B_{i}:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        descs_a.append(ma.group(1).strip() if ma else "")
        descs_b.append(mb.group(1).strip() if mb else "")
    ok = all(descs_a) and all(descs_b)
    return descs_a, descs_b, ok


def qwen3_generate_single(model, processor, image_pil, device):
    """Generate a single TEXTURE_A / TEXTURE_B description (for QwenTxt baseline)."""
    import torch
    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": QWEN_SINGLE_PROMPT},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image_pil], return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=200, do_sample=False, temperature=1.0)
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, input_len:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True)


def parse_single_output(text):
    da = db = ""
    ma = re.search(r'TEXTURE_A:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    mb = re.search(r'TEXTURE_B:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if ma: da = ma.group(1).strip()
    if mb: db = mb.group(1).strip()
    return da, db, bool(da and db)


# --- SAM3 DETR helpers (for QwenTxt baseline) ---

def run_detr_with_text(model, backbone_out, text_feat, B, device):
    import torch
    prompt_bf = text_feat["prompt"].unsqueeze(0).expand(B, -1, -1)
    mask_bf = text_feat["mask"].unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        out = model.base._run_sam3_from_backbone(backbone_out, prompt_bf, mask_bf)
    scores = out["pred_logits"][0].squeeze(-1).sigmoid()
    best = scores.argmax().item()
    return out["pred_masks"][0, best]


def run_detr_full(model, backbone_out, text_feat, B, device):
    """Run DETR, return ALL proposals with scores + semantic mask."""
    import torch
    prompt_bf = text_feat["prompt"].unsqueeze(0).expand(B, -1, -1)
    mask_bf = text_feat["mask"].unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        out = model.base._run_sam3_from_backbone(backbone_out, prompt_bf, mask_bf)
    scores = out["pred_logits"][0].squeeze(-1).sigmoid()
    all_masks = out["pred_masks"][0]
    semantic_mask = out.get("semantic_mask", None)
    return scores, all_masks, semantic_mask


def postprocess_mask(mask_logit, gt_h, gt_w):
    import torch
    import torch.nn.functional as F
    if mask_logit.shape[-2:] != (gt_h, gt_w):
        mask_logit = F.interpolate(
            mask_logit[None, None].float() if mask_logit.ndim == 2
            else mask_logit.unsqueeze(0).float() if mask_logit.ndim == 3
            else mask_logit.float(),
            size=(gt_h, gt_w),
            mode="bilinear", align_corners=False,
        ).squeeze()
    return (mask_logit.sigmoid().cpu().numpy() > 0.5).astype(np.float32)


def clipseg_heatmap(clipseg_model, clipseg_proc, image_pil, desc, device="cuda"):
    import torch
    import torch.nn.functional as F
    inputs = clipseg_proc(
        text=[desc], images=[image_pil],
        return_tensors="pt", padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        outputs = clipseg_model.float()(**inputs)
    logits = outputs.logits.squeeze().float()
    heatmap = torch.sigmoid(logits).cpu().numpy()
    return heatmap


def compute_iou(pred, gt):
    pred_b, gt_b = pred > 0.5, gt > 0.5
    inter = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    return 0.0 if union == 0 else float(inter / union)


def compute_dice(pred, gt):
    pred_b, gt_b = pred > 0.5, gt > 0.5
    inter = (pred_b & gt_b).sum()
    total = pred_b.sum() + gt_b.sum()
    return 1.0 if total == 0 else float(2.0 * inter / total)


def postprocess_mask_to_np(mask_logit, gt_h, gt_w):
    """Convert mask logit to numpy probability map (no thresholding)."""
    import torch
    import torch.nn.functional as F
    if mask_logit.shape[-2:] != (gt_h, gt_w):
        mask_logit = F.interpolate(
            mask_logit[None, None].float() if mask_logit.ndim == 2
            else mask_logit.unsqueeze(0).float(),
            size=(gt_h, gt_w),
            mode="bilinear", align_corners=False,
        ).squeeze()
    return mask_logit.sigmoid().float().cpu().numpy()


# ===================================================================== #
#  SAM3 Proposal Visualization (same style as test_sam3_proposal_diversity) #
# ===================================================================== #

COLOR_A = (0, 0, 220)
COLOR_B = (220, 80, 0)


def draw_proposal_row(image_bgr, gt_mask, proposals, desc, label,
                      cell_size=180, gt_color=COLOR_A):
    """
    Single row: GT | Prop1 | Prop2 | ... | PropK
    Each proposal overlay + confidence + GT IoU.
    """
    h, w = image_bgr.shape[:2]
    s = cell_size / max(h, w)
    ch, cw = int(h * s), int(w * s)

    def ri(img):
        return cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)

    def rm(m):
        return cv2.resize(m, (cw, ch), interpolation=cv2.INTER_NEAREST)

    img = ri(image_bgr)

    # GT cell
    gt_vis = img.copy()
    overlay = img.copy()
    overlay[rm(gt_mask) > 0.5] = gt_color
    gt_vis = cv2.addWeighted(overlay, 0.45, gt_vis, 0.55, 0)

    cells = [gt_vis]

    for mask_np, conf, gt_iou, rank in proposals:
        mask_r = rm(mask_np)
        cell = img.copy()
        ov = img.copy()
        ov[mask_r > 0.5] = gt_color
        cell = cv2.addWeighted(ov, 0.5, cell, 0.5, 0)

        # Border color by GT IoU
        g = int(min(gt_iou * 255, 255))
        r = int(max(0, (1 - gt_iou) * 255))
        cv2.rectangle(cell, (0, 0), (cw - 1, ch - 1), (0, g, r), 2)

        # Scores
        cv2.putText(cell, f"c:{conf:.3f}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(cell, f"IoU:{gt_iou:.3f}", (4, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
        area = (mask_np > 0.5).mean() * 100
        cv2.putText(cell, f"{area:.0f}%", (4, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1, cv2.LINE_AA)

        cells.append(cell)

    sep = 2
    row_parts = []
    for c in cells:
        row_parts.extend([c, np.zeros((ch, sep, 3), dtype=np.uint8)])
    row_img = np.hstack(row_parts[:-1])

    actual_w = row_img.shape[1]

    # Label bar
    label_bar = np.zeros((18, actual_w, 3), dtype=np.uint8) + 25
    cv2.putText(label_bar, f"{label}: \"{desc}\""[:actual_w // 4], (4, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 200, 255), 1, cv2.LINE_AA)

    return np.vstack([label_bar, row_img])


def draw_sam3_proposal_grid(image_bgr, gt_a, gt_b, sections,
                            crop_name, cell_size=180, top_k=10):
    """
    Build full grid: title + proposal rows.

    sections: list of (label, desc, gt_mask, proposals, gt_color)
    """
    h, w = image_bgr.shape[:2]
    s = cell_size / max(h, w)
    ch, cw = int(h * s), int(w * s)

    rows = []

    # Title
    n_cols = 1 + top_k  # GT + proposals
    est_w = n_cols * (cw + 2)
    title_bar = np.zeros((26, est_w, 3), dtype=np.uint8) + 40
    cv2.putText(title_bar,
                f"Sample: {crop_name}  |  Top-{top_k} SAM3 DETR proposals per description",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    rows.append(title_bar)

    sep_h = 4
    for label, desc, gt_mask, proposals, gt_color in sections:
        row = draw_proposal_row(image_bgr, gt_mask, proposals, desc, label,
                                cell_size=cell_size, gt_color=gt_color)
        if row.shape[1] < est_w:
            pad = np.zeros((row.shape[0], est_w - row.shape[1], 3), dtype=np.uint8)
            row = np.hstack([row, pad])
        elif row.shape[1] > est_w:
            est_w = row.shape[1]
            if rows[0].shape[1] < est_w:
                pad = np.zeros((rows[0].shape[0], est_w - rows[0].shape[1], 3), dtype=np.uint8) + 40
                rows[0] = np.hstack([rows[0], pad])
        rows.append(row)
        rows.append(np.zeros((sep_h, row.shape[1], 3), dtype=np.uint8))

    # Make all rows same width
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    return np.vstack(padded)


def compute_ari(pred_a, pred_b, gt_a, gt_b):
    from sklearn.metrics import adjusted_rand_score
    pred_labels = np.zeros(pred_a.shape, dtype=np.int32)
    pred_labels[pred_a > 0.5] = 1
    pred_labels[pred_b > 0.5] = 2
    gt_labels = np.zeros(gt_a.shape, dtype=np.int32)
    gt_labels[gt_a > 0.5] = 1
    gt_labels[gt_b > 0.5] = 2
    return float(adjusted_rand_score(gt_labels.ravel(), pred_labels.ravel()))


def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from qwen2sam.models.qwen2sam_v3_tracker import Qwen2SAMv3Tracker
    from qwen2sam.training.train_phase1 import load_config, set_seed
    from qwen2sam.data.dataset_v2 import preprocess_image_for_sam3

    parser = argparse.ArgumentParser(
        description="Pair Optimizer — CLIPSeg heatmap pairing test")
    parser.add_argument("--config", type=str,
                        default="qwen2sam/configs/v3_tracker_detexure.yaml")
    parser.add_argument("--data_root", type=str, default="/home/aviad/RWTD")
    parser.add_argument("--samples", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/pair_optimizer")
    parser.add_argument("--gamma", type=float, default=2.5,
                        help="Overlap penalty weight")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip all visualization (fast mode)")
    args = parser.parse_args()

    sample_names = args.samples.split(",") if args.samples else DEFAULT_SAMPLES
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _project_root / config_path
    cfg = load_config(str(config_path))
    image_size = cfg["model"].get("image_size", 1008)

    meta_path = Path(args.data_root) / "metadata_phase1.json"
    with open(meta_path) as f:
        all_meta = json.load(f)
    meta_by_name = {e["crop_name"]: e for e in all_meta}
    if sample_names is None:
        samples = all_meta  # run all
    else:
        samples = [meta_by_name[n] for n in sample_names if n in meta_by_name]
    print(f"Testing {len(samples)} samples, gamma={args.gamma}")

    output_dir = _project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SAM3 for QwenTxt baseline
    if "tracker" in cfg:
        cfg["tracker"].pop("v3_checkpoint", None)
    print("Loading SAM3 (for QwenTxt baseline)...")
    sam3_model = Qwen2SAMv3Tracker(cfg, device=str(device))
    sam3_model.base.sam3.eval()

    qwen3, qwen3_proc = load_qwen3_model(device)
    clipseg_model, clipseg_proc = load_clipseg_model(device)

    results_log = []
    txt_results_log = []  # QwenTxt baseline metrics
    sam3_results_log = []  # SAM3 diverse baseline metrics
    # (SemSeg is now merged into SAM3+SemSeg combined pool)
    sam3_proposals_dir = output_dir / "sam3_proposals"
    sam3_proposals_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for i, entry in enumerate(samples):
        crop_name = entry["crop_name"]
        print(f"\n{'='*70}")
        print(f"  [{crop_name}] ({i+1}/{len(samples)})")
        print(f"{'='*70}")

        image_bgr = cv2.imread(entry["image_path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        orig_h, orig_w = image_rgb.shape[:2]

        gt_a = cv2.imread(entry["mask_a_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_b = cv2.imread(entry["mask_b_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_h, gt_w = gt_a.shape

        # ============================================================ #
        #  SAM3 backbone (shared by QwenTxt + SAM3 Diverse)              #
        # ============================================================ #
        sam_img = preprocess_image_for_sam3(image_rgb, image_size).unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = sam3_model.base.sam3.backbone.forward_image(sam_img)
            backbone_out["img_batch_all_stages"] = sam_img

        # ============================================================ #
        #  QwenTxt baseline: single description → SAM3 DETR              #
        # ============================================================ #

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            raw_single = qwen3_generate_single(qwen3, qwen3_proc, image_pil, device)
            txt_desc_a, txt_desc_b, txt_ok = parse_single_output(raw_single)

            if txt_ok:
                t_a_out = sam3_model.base.sam3.backbone.forward_text([txt_desc_a], device=device)
                t_b_out = sam3_model.base.sam3.backbone.forward_text([txt_desc_b], device=device)
                feat_a = {"prompt": t_a_out["language_features"].squeeze(1),
                          "mask": t_a_out["language_mask"].squeeze(0)}
                feat_b = {"prompt": t_b_out["language_features"].squeeze(1),
                          "mask": t_b_out["language_mask"].squeeze(0)}
                a_logit = run_detr_with_text(sam3_model, backbone_out, feat_a, 1, device)
                b_logit = run_detr_with_text(sam3_model, backbone_out, feat_b, 1, device)

                txt_pred_a = postprocess_mask(a_logit, gt_h, gt_w)
                txt_pred_b = postprocess_mask(b_logit, gt_h, gt_w)

                # Handle label swap
                iou_d = compute_iou(txt_pred_a, gt_a) + compute_iou(txt_pred_b, gt_b)
                iou_s = compute_iou(txt_pred_a, gt_b) + compute_iou(txt_pred_b, gt_a)
                if iou_s > iou_d:
                    txt_pred_a, txt_pred_b = txt_pred_b, txt_pred_a

                txt_iou_a = compute_iou(txt_pred_a, gt_a)
                txt_iou_b = compute_iou(txt_pred_b, gt_b)
                txt_dice_a = compute_dice(txt_pred_a, gt_a)
                txt_dice_b = compute_dice(txt_pred_b, gt_b)
                txt_ari = compute_ari(txt_pred_a, txt_pred_b, gt_a, gt_b)
            else:
                txt_iou_a = txt_iou_b = txt_dice_a = txt_dice_b = txt_ari = 0.0
                print(f"    QwenTxt PARSE FAIL: {raw_single[:100]}")

        txt_results_log.append({
            "crop_name": crop_name,
            "iou_a": txt_iou_a, "iou_b": txt_iou_b,
            "mean_iou": (txt_iou_a + txt_iou_b) / 2.0,
            "dice_a": txt_dice_a, "dice_b": txt_dice_b,
            "mean_dice": (txt_dice_a + txt_dice_b) / 2.0,
            "ari": txt_ari,
        })
        print(f"    QwenTxt: \"{txt_desc_a[:40]}\" / \"{txt_desc_b[:40]}\"")
        print(f"    QwenTxt mIoU={((txt_iou_a+txt_iou_b)/2):.4f} "
              f"mDice={((txt_dice_a+txt_dice_b)/2):.4f} ARI={txt_ari:.4f}")

        # ============================================================ #
        #  Qwen3: diverse descriptions (shared by CLIPSeg + SAM3)        #
        # ============================================================ #

        raw = qwen3_generate_diverse(qwen3, qwen3_proc, image_pil, device)
        descs_a, descs_b, ok = parse_diverse_output(raw)

        if not ok:
            print(f"    PARSE FAIL: {raw[:200]}")
            descs_a = [d if d else f"texture A variant {j+1}" for j, d in enumerate(descs_a)]
            descs_b = [d if d else f"texture B variant {j+1}" for j, d in enumerate(descs_b)]

        for j, d in enumerate(descs_a):
            print(f"    A[{j}] {DESC_LABELS[j]:20s}: {d}")
        for j, d in enumerate(descs_b):
            print(f"    B[{j}] {DESC_LABELS[j]:20s}: {d}")

        # ============================================================ #
        #  SAM3 Diverse: 10 descriptions → SAM3 DETR → top-1 per desc   #
        #  + top-10 proposal vis + STD vis                               #
        # ============================================================ #

        SAM3_TOP_K = 10

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Reuse backbone_out from QwenTxt section above
            sam3_masks_a = []   # conf-weighted prob maps (same role as CLIPSeg heatmaps)
            sam3_masks_b = []   # total: N_DESCRIPTIONS * SAM3_TOP_K per texture
            sam3_descs_a = []   # matching description labels
            sam3_descs_b = []
            vis_sections = []   # for proposal grid visualization
            std_stats_a = []    # mean STD per description (for summary)
            std_stats_b = []
            semantic_masks_a = []  # semantic mask per description (text-dependent)
            semantic_masks_b = []

            for j in range(N_DESCRIPTIONS):
                label = DESC_LABELS[j] if j < len(DESC_LABELS) else f"Desc {j+1}"

                # --- Texture A ---
                t_out_a = sam3_model.base.sam3.backbone.forward_text(
                    [descs_a[j]], device=device)
                feat_a_j = {
                    "prompt": t_out_a["language_features"].squeeze(1),
                    "mask": t_out_a["language_mask"].squeeze(0),
                }
                scores_a, all_masks_a, sem_mask_a = run_detr_full(
                    sam3_model, backbone_out, feat_a_j, 1, device)

                # Capture semantic mask for this A description
                if sem_mask_a is not None:
                    semantic_masks_a.append(postprocess_mask_to_np(
                        sem_mask_a[0, 0], gt_h, gt_w))

                # Top-K proposals for visualization + pair optimizer
                top_indices_a = scores_a.topk(
                    min(SAM3_TOP_K, len(scores_a))).indices
                proposals_a = []
                for rank, idx in enumerate(top_indices_a):
                    idx_val = idx.item()
                    conf = scores_a[idx_val].item()
                    prob_map = postprocess_mask_to_np(
                        all_masks_a[idx_val], gt_h, gt_w)
                    mask_bin = (prob_map > 0.5).astype(np.float32)
                    gt_iou = compute_iou(mask_bin, gt_a)
                    proposals_a.append((mask_bin, conf, gt_iou, rank + 1))

                    # Collect confidence-weighted probability map for scoring
                    # Weight by DETR confidence so _quality() prefers high-confidence proposals
                    sam3_masks_a.append(prob_map * conf)
                    sam3_descs_a.append(
                        f"{descs_a[j]} [r{rank+1} c={conf:.3f}]")

                # STD across all 200 masks (stat only, no visualization)
                all_bin_a = np.stack([
                    (postprocess_mask_to_np(all_masks_a[k], gt_h, gt_w) > 0.5
                     ).astype(np.float32)
                    for k in range(all_masks_a.shape[0])
                ], axis=0)
                mean_std_a = float(np.mean(np.std(all_bin_a, axis=0)))
                std_stats_a.append(mean_std_a)

                vis_sections.append(
                    (f"A-{label}", descs_a[j], gt_a, proposals_a, COLOR_A))

                print(f"    SAM3 A[{j}] {label}: top1 conf={proposals_a[0][1]:.3f} "
                      f"IoU={proposals_a[0][2]:.3f}  mask_std={mean_std_a:.4f}")

                # --- Texture B ---
                t_out_b = sam3_model.base.sam3.backbone.forward_text(
                    [descs_b[j]], device=device)
                feat_b_j = {
                    "prompt": t_out_b["language_features"].squeeze(1),
                    "mask": t_out_b["language_mask"].squeeze(0),
                }
                scores_b, all_masks_b, sem_mask_b = run_detr_full(
                    sam3_model, backbone_out, feat_b_j, 1, device)

                # Capture semantic mask for this B description
                if sem_mask_b is not None:
                    semantic_masks_b.append(postprocess_mask_to_np(
                        sem_mask_b[0, 0], gt_h, gt_w))

                top_indices_b = scores_b.topk(
                    min(SAM3_TOP_K, len(scores_b))).indices
                proposals_b = []
                for rank, idx in enumerate(top_indices_b):
                    idx_val = idx.item()
                    conf = scores_b[idx_val].item()
                    prob_map = postprocess_mask_to_np(
                        all_masks_b[idx_val], gt_h, gt_w)
                    mask_bin = (prob_map > 0.5).astype(np.float32)
                    gt_iou = compute_iou(mask_bin, gt_b)
                    proposals_b.append((mask_bin, conf, gt_iou, rank + 1))

                    # Collect confidence-weighted probability map for scoring
                    sam3_masks_b.append(prob_map * conf)
                    sam3_descs_b.append(
                        f"{descs_b[j]} [r{rank+1} c={conf:.3f}]")

                all_bin_b = np.stack([
                    (postprocess_mask_to_np(all_masks_b[k], gt_h, gt_w) > 0.5
                     ).astype(np.float32)
                    for k in range(all_masks_b.shape[0])
                ], axis=0)
                mean_std_b = float(np.mean(np.std(all_bin_b, axis=0)))
                std_stats_b.append(mean_std_b)

                vis_sections.append(
                    (f"B-{label}", descs_b[j], gt_b, proposals_b, COLOR_B))

                print(f"    SAM3 B[{j}] {label}: top1 conf={proposals_b[0][1]:.3f} "
                      f"IoU={proposals_b[0][2]:.3f}  mask_std={mean_std_b:.4f}")

        # Print STD summary for this sample
        avg_std_a = float(np.mean(std_stats_a))
        avg_std_b = float(np.mean(std_stats_b))
        print(f"    Mean STD across 200 masks: A={avg_std_a:.4f}  B={avg_std_b:.4f}")
        print(f"    Semantic masks captured: A={len(semantic_masks_a)} B={len(semantic_masks_b)}")

        # --- Save SAM3 proposal grid visualization (flat in sam3_proposals/) ---
        if not args.no_vis:
            grid = draw_sam3_proposal_grid(
                image_bgr, gt_a, gt_b, vis_sections,
                crop_name, cell_size=180, top_k=SAM3_TOP_K)
            cv2.imwrite(str(sam3_proposals_dir / f"{crop_name}_proposals.png"), grid)
            print(f"    Saved SAM3 proposals: {sam3_proposals_dir / f'{crop_name}_proposals.png'}")

        # --- Save semantic mask visualization (both textures A and B) ---
        if not args.no_vis and semantic_masks_a and semantic_masks_b:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Average semantic masks across descriptions (one per texture)
            sem_a_avg = np.mean(semantic_masks_a, axis=0)
            sem_b_avg = np.mean(semantic_masks_b, axis=0)

            n_desc_vis = len(semantic_masks_a)
            # Layout: 2 rows x (2 + n_desc) cols
            # Row 1: Original | GT | Avg-A heatmap | per-desc A heatmaps...
            # Row 2: empty    |    | Avg-B heatmap | per-desc B heatmaps...
            n_cols = 3 + n_desc_vis
            fig_sem, axes_sem = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
            fig_sem.suptitle(f"Semantic Seg Head — {crop_name}  |  "
                             f"Texture A (blue) vs B (orange)",
                             fontsize=13, fontweight="bold")

            img_norm = image_rgb.astype(np.float32) / 255.0
            c_a = np.array([0, 0, 220]) / 255.0
            c_b = np.array([220, 80, 0]) / 255.0

            # (0,0) Original image
            axes_sem[0, 0].imshow(image_rgb)
            axes_sem[0, 0].set_title("Original", fontsize=9)
            axes_sem[0, 0].axis("off")
            axes_sem[1, 0].axis("off")

            # (0,1) GT overlay
            gt_a_r = cv2.resize(gt_a, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            gt_b_r = cv2.resize(gt_b, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            gt_ov = img_norm.copy()
            gt_ov[gt_a_r > 0.5] = gt_ov[gt_a_r > 0.5] * 0.55 + c_a * 0.45
            gt_ov[gt_b_r > 0.5] = gt_ov[gt_b_r > 0.5] * 0.55 + c_b * 0.45
            axes_sem[0, 1].imshow(np.clip(gt_ov, 0, 1))
            axes_sem[0, 1].set_title("GT (Blue=A, Orange=B)", fontsize=9)
            axes_sem[0, 1].axis("off")
            axes_sem[1, 1].axis("off")

            # (0,2) Average semantic mask A
            sem_a_full = cv2.resize(sem_a_avg, (orig_w, orig_h),
                                    interpolation=cv2.INTER_LINEAR)
            axes_sem[0, 2].imshow(sem_a_full, cmap="jet", vmin=0, vmax=1)
            axes_sem[0, 2].set_title(f"Avg Semantic A\nmean={sem_a_full.mean():.3f}", fontsize=9)
            axes_sem[0, 2].axis("off")

            # (1,2) Average semantic mask B
            sem_b_full = cv2.resize(sem_b_avg, (orig_w, orig_h),
                                    interpolation=cv2.INTER_LINEAR)
            axes_sem[1, 2].imshow(sem_b_full, cmap="jet", vmin=0, vmax=1)
            axes_sem[1, 2].set_title(f"Avg Semantic B\nmean={sem_b_full.mean():.3f}", fontsize=9)
            axes_sem[1, 2].axis("off")

            # Per-description semantic masks
            for j in range(n_desc_vis):
                col = 3 + j
                dlabel = DESC_LABELS[j] if j < len(DESC_LABELS) else f"D{j+1}"

                # Row 0: Texture A
                sa = cv2.resize(semantic_masks_a[j], (orig_w, orig_h),
                                interpolation=cv2.INTER_LINEAR)
                sa_ov = img_norm.copy()
                sa_ov[sa > 0.5] = sa_ov[sa > 0.5] * 0.5 + c_a * 0.5
                axes_sem[0, col].imshow(np.clip(sa_ov, 0, 1))
                axes_sem[0, col].set_title(f"A: {dlabel}\n{sa.mean():.3f}", fontsize=8)
                axes_sem[0, col].axis("off")

                # Row 1: Texture B
                sb = cv2.resize(semantic_masks_b[j], (orig_w, orig_h),
                                interpolation=cv2.INTER_LINEAR)
                sb_ov = img_norm.copy()
                sb_ov[sb > 0.5] = sb_ov[sb > 0.5] * 0.5 + c_b * 0.5
                axes_sem[1, col].imshow(np.clip(sb_ov, 0, 1))
                axes_sem[1, col].set_title(f"B: {dlabel}\n{sb.mean():.3f}", fontsize=8)
                axes_sem[1, col].axis("off")

            plt.tight_layout()
            sem_path = sam3_proposals_dir / f"{crop_name}_semantic.png"
            fig_sem.savefig(str(sem_path), dpi=150, bbox_inches="tight")
            plt.close(fig_sem)
            print(f"    Saved semantic masks: {sem_path}")

        # --- SAM3+SemSeg combined pair optimizer ---
        # Pool: 50 DETR conf-weighted masks + 5 semantic masks = 55 per texture
        # When DETR masks all get filtered, semantic masks survive as fallback
        # When semantic masks are too broad, DETR provides sharp boundaries
        combined_a = sam3_masks_a + semantic_masks_a
        combined_b = sam3_masks_b + semantic_masks_b
        combined_descs_a = sam3_descs_a + [
            f"SemSeg: {descs_a[j]}" for j in range(len(semantic_masks_a))]
        combined_descs_b = sam3_descs_b + [
            f"SemSeg: {descs_b[j]}" for j in range(len(semantic_masks_b))]

        n_detr_a, n_sem_a = len(sam3_masks_a), len(semantic_masks_a)
        n_detr_b, n_sem_b = len(sam3_masks_b), len(semantic_masks_b)
        print(f"    Combined pool: A={n_detr_a}+{n_sem_a}={len(combined_a)}, "
              f"B={n_detr_b}+{n_sem_b}={len(combined_b)}")

        # Same flow as CSeg: visualize_pairing_test does filter → score → WTA → vis
        sam3_best, sam3_sorted, sam3_flog_a, sam3_flog_b = visualize_pairing_test(
            image_rgb, combined_a, combined_b,
            combined_descs_a, combined_descs_b,
            gt_a=gt_a, gt_b=gt_b, gamma=args.gamma,
            save_path=None if args.no_vis else sam3_proposals_dir / f"{crop_name}_pairs.png",
            crop_name=f"{crop_name} (SAM3+SemSeg)",
            skip_vis=args.no_vis,
        )

        n_valid_sam3_a = sum(1 for f in sam3_flog_a if f["passed"]) if sam3_flog_a else 0
        n_valid_sam3_b = sum(1 for f in sam3_flog_b if f["passed"]) if sam3_flog_b else 0
        print(f"    Combined filter: A={n_valid_sam3_a}/{len(combined_a)}, "
              f"B={n_valid_sam3_b}/{len(combined_b)}")

        if sam3_best is not None:
            # Evaluate WTA — same logic as CSeg section below
            valid_indices_a = [fl["index"] for fl in sam3_flog_a if fl["passed"]]
            valid_indices_b = [fl["index"] for fl in sam3_flog_b if fl["passed"]]
            win_idx_a = valid_indices_a[sam3_best["idx_a"]]
            win_idx_b = valid_indices_b[sam3_best["idx_b"]]
            win_a_sam3 = combined_a[win_idx_a]
            win_b_sam3 = combined_b[win_idx_b]

            # Report source of winning masks
            src_a = "SemSeg" if win_idx_a >= n_detr_a else "DETR"
            src_b = "SemSeg" if win_idx_b >= n_detr_b else "DETR"
            print(f"    Winner source: A={src_a}, B={src_b}")

            h_gt, w_gt = gt_a.shape
            wa = cv2.resize(win_a_sam3, (w_gt, h_gt), interpolation=cv2.INTER_LINEAR)
            wb = cv2.resize(win_b_sam3, (w_gt, h_gt), interpolation=cv2.INTER_LINEAR)
            sam3_wta_a = (wa > wb).astype(np.float32)
            sam3_wta_b = (wb > wa).astype(np.float32)

            # Degenerate WTA fallback
            pix_a = sam3_wta_a.sum() / sam3_wta_a.size
            pix_b = sam3_wta_b.sum() / sam3_wta_b.size
            if min(pix_a, pix_b) < 0.02:
                sam3_wta_a = (wa > 0.5).astype(np.float32)
                sam3_wta_b = (wb > 0.5).astype(np.float32)
                print(f"    SAM3 WTA degenerate — using independent thresholding")

            # Handle label swap
            iou_d = compute_iou(sam3_wta_a, gt_a) + compute_iou(sam3_wta_b, gt_b)
            iou_s = compute_iou(sam3_wta_a, gt_b) + compute_iou(sam3_wta_b, gt_a)
            if iou_s > iou_d:
                sam3_wta_a, sam3_wta_b = sam3_wta_b, sam3_wta_a

            sam3_iou_a = compute_iou(sam3_wta_a, gt_a)
            sam3_iou_b = compute_iou(sam3_wta_b, gt_b)
            sam3_dice_a = compute_dice(sam3_wta_a, gt_a)
            sam3_dice_b = compute_dice(sam3_wta_b, gt_b)
            sam3_ari = compute_ari(sam3_wta_a, sam3_wta_b, gt_a, gt_b)
            sam3_miou = (sam3_iou_a + sam3_iou_b) / 2.0
            sam3_mdice = (sam3_dice_a + sam3_dice_b) / 2.0

            print(f"    SAM3 Diverse: mIoU={sam3_miou:.4f} mDice={sam3_mdice:.4f} "
                  f"ARI={sam3_ari:.4f}")
        else:
            sam3_iou_a = sam3_iou_b = sam3_dice_a = sam3_dice_b = sam3_ari = 0.0
            sam3_miou = sam3_mdice = 0.0

        sam3_results_log.append({
            "crop_name": crop_name,
            "iou_a": sam3_iou_a, "iou_b": sam3_iou_b,
            "mean_iou": sam3_miou,
            "dice_a": sam3_dice_a, "dice_b": sam3_dice_b,
            "mean_dice": sam3_mdice,
            "ari": sam3_ari,
            "best_pair": sam3_best,
            "n_candidates_a": n_valid_sam3_a,
            "n_candidates_b": n_valid_sam3_b,
            "mean_std_a": avg_std_a,
            "mean_std_b": avg_std_b,
        })

        # ============================================================ #
        #  Pair Optimizer: diverse descriptions → CLIPSeg → best pair    #
        # ============================================================ #

        # --- CLIPSeg: heatmaps ---
        heatmaps_a = [clipseg_heatmap(clipseg_model, clipseg_proc, image_pil, d, device)
                      for d in descs_a]
        heatmaps_b = [clipseg_heatmap(clipseg_model, clipseg_proc, image_pil, d, device)
                      for d in descs_b]

        # --- Run optimizer ---
        best, sorted_pairs, flog_a, flog_b = visualize_pairing_test(
            image_rgb, heatmaps_a, heatmaps_b, descs_a, descs_b,
            gt_a=gt_a, gt_b=gt_b, gamma=args.gamma,
            save_path=None if args.no_vis else output_dir / f"{crop_name}_pairs.png",
            crop_name=crop_name,
            skip_vis=args.no_vis,
        )

        # --- Filter stats ---
        n_passed_a = sum(1 for f in flog_a if f["passed"])
        n_passed_b = sum(1 for f in flog_b if f["passed"])
        print(f"\n    Filter: A={n_passed_a}/{len(heatmaps_a)} passed, "
              f"B={n_passed_b}/{len(heatmaps_b)} passed")
        for fl in flog_a + flog_b:
            if not fl["passed"]:
                print(f"      REJECTED: \"{fl['description'][:40]}\" — {fl['reject_reason']}")

        if best is None:
            print(f"    No valid pairs!")
            results_log.append({"crop_name": crop_name, "status": "no_valid_pairs"})
            continue

        # --- Evaluate winning pair against GT ---
        win_a = heatmaps_a[flog_a.index(
            next(f for f in flog_a if f["passed"] and
                 f["description"] == best["desc_a"]))]
        win_b = heatmaps_b[flog_b.index(
            next(f for f in flog_b if f["passed"] and
                 f["description"] == best["desc_b"]))]

        # WTA mask from winning pair
        h, w = gt_a.shape
        wa = cv2.resize(win_a, (w, h), interpolation=cv2.INTER_LINEAR)
        wb = cv2.resize(win_b, (w, h), interpolation=cv2.INTER_LINEAR)
        wta_a = (wa > wb).astype(np.float32)
        wta_b = (wb > wa).astype(np.float32)

        # Degenerate WTA fallback: if one mask gets <2% of pixels
        pix_a = wta_a.sum() / wta_a.size
        pix_b = wta_b.sum() / wta_b.size
        if min(pix_a, pix_b) < 0.02:
            wta_a = (wa > 0.5).astype(np.float32)
            wta_b = (wb > 0.5).astype(np.float32)
            print(f"    CSeg WTA degenerate (A={pix_a:.3f}, B={pix_b:.3f}) "
                  f"— using independent thresholding")

        # Handle label swap via IoU
        iou_direct_a = compute_iou(wta_a, gt_a)
        iou_direct_b = compute_iou(wta_b, gt_b)
        iou_swap_a = compute_iou(wta_a, gt_b)
        iou_swap_b = compute_iou(wta_b, gt_a)
        if (iou_swap_a + iou_swap_b) > (iou_direct_a + iou_direct_b):
            wta_a, wta_b = wta_b, wta_a

        iou_a = compute_iou(wta_a, gt_a)
        iou_b = compute_iou(wta_b, gt_b)
        dice_a = compute_dice(wta_a, gt_a)
        dice_b = compute_dice(wta_b, gt_b)
        ari = compute_ari(wta_a, wta_b, gt_a, gt_b)

        mean_iou = (iou_a + iou_b) / 2.0
        mean_dice = (dice_a + dice_b) / 2.0

        print(f"\n    WINNING PAIR: A[{best['idx_a']}] × B[{best['idx_b']}]")
        print(f"      A: \"{best['desc_a'][:60]}\"")
        print(f"      B: \"{best['desc_b'][:60]}\"")
        print(f"      Quality: A={best['quality_a']:.3f}  B={best['quality_b']:.3f}")
        print(f"      Overlap: {best['overlap']:.3f}  Score: {best['score']:.3f}")
        print(f"      IoU:  A={iou_a:.3f}  B={iou_b:.3f}  Mean={mean_iou:.3f}")
        print(f"      Dice: A={dice_a:.3f}  B={dice_b:.3f}  Mean={mean_dice:.3f}")
        print(f"      ARI:  {ari:.3f}")

        # Top 3 pairs
        print(f"\n    Top 3 pairs:")
        for k, p in enumerate(sorted_pairs[:3]):
            print(f"      #{k+1}: A[{p['idx_a']}]×B[{p['idx_b']}] "
                  f"score={p['score']:.3f} (Q_A={p['quality_a']:.3f} "
                  f"Q_B={p['quality_b']:.3f} ovlp={p['overlap']:.3f})")

        results_log.append({
            "crop_name": crop_name,
            "status": "ok",
            "n_valid_a": n_passed_a,
            "n_valid_b": n_passed_b,
            "best_pair": best,
            "iou_a": iou_a, "iou_b": iou_b, "mean_iou": mean_iou,
            "dice_a": dice_a, "dice_b": dice_b, "mean_dice": mean_dice,
            "ari": ari,
            "top3": sorted_pairs[:3],
        })

    elapsed = time.time() - t0

    # Summary
    valid_results = [r for r in results_log if r.get("status") == "ok"]
    txt_by_name = {r["crop_name"]: r for r in txt_results_log}
    sam3_by_name = {r["crop_name"]: r for r in sam3_results_log}

    print(f"\n{'='*110}")
    print(f"  CSeg PairOpt vs SAM3+SemSeg vs QwenTxt(single) — "
          f"{len(samples)} samples ({elapsed:.1f}s)")
    print(f"{'='*110}")

    if valid_results:
        # Per-sample comparison table — 3 approaches
        print(f"\n  {'':>8s}  {'── CSeg PairOpt ──':^22s}  "
              f"{'── SAM3+SemSeg ──':^22s}  "
              f"{'── QwenTxt(single) ──':^22s}  {'Best':>8s}")
        print(f"  {'Sample':>8s}  {'mIoU':>7s} {'mDice':>7s} {'ARI':>7s}"
              f"  {'mIoU':>7s} {'mDice':>7s} {'ARI':>7s}"
              f"  {'mIoU':>7s} {'mDice':>7s} {'ARI':>7s}  {'':>8s}")
        print(f"  {'─'*100}")

        cseg_wins, sam3d_wins, txt_wins = 0, 0, 0
        for r in valid_results:
            tr = txt_by_name.get(r["crop_name"], {})
            sr = sam3_by_name.get(r["crop_name"], {})

            c_miou = r["mean_iou"]
            s_miou = sr.get("mean_iou", 0)
            t_miou = tr.get("mean_iou", 0)

            best_val = max(c_miou, s_miou, t_miou)
            if c_miou == best_val:
                winner = "CSeg"
                cseg_wins += 1
            elif s_miou == best_val:
                winner = "SAM3+Sem"
                sam3d_wins += 1
            else:
                winner = "QwenTxt"
                txt_wins += 1

            print(f"  {r['crop_name']:>8s}"
                  f"  {c_miou:7.4f} {r['mean_dice']:7.4f} {r['ari']:7.4f}"
                  f"  {s_miou:7.4f} {sr.get('mean_dice', 0):7.4f} {sr.get('ari', 0):7.4f}"
                  f"  {t_miou:7.4f} {tr.get('mean_dice', 0):7.4f} {tr.get('ari', 0):7.4f}"
                  f"  {winner:>8s}")
        print(f"  {'─'*100}")

        # Aggregated
        valid_names = {r["crop_name"] for r in valid_results}

        p_miou = float(np.mean([r["mean_iou"] for r in valid_results]))
        p_mdice = float(np.mean([r["mean_dice"] for r in valid_results]))
        p_ari = float(np.nanmean([r["ari"] for r in valid_results]))

        sam3_valid = [r for r in sam3_results_log if r["crop_name"] in valid_names]
        s_miou = float(np.mean([r["mean_iou"] for r in sam3_valid])) if sam3_valid else 0
        s_mdice = float(np.mean([r["mean_dice"] for r in sam3_valid])) if sam3_valid else 0
        s_ari = float(np.nanmean([r["ari"] for r in sam3_valid])) if sam3_valid else 0

        txt_valid = [r for r in txt_results_log if r["crop_name"] in valid_names]
        t_miou = float(np.mean([r["mean_iou"] for r in txt_valid])) if txt_valid else 0
        t_mdice = float(np.mean([r["mean_dice"] for r in txt_valid])) if txt_valid else 0
        t_ari = float(np.nanmean([r["ari"] for r in txt_valid])) if txt_valid else 0

        print(f"  {'MEAN':>8s}"
              f"  {p_miou:7.4f} {p_mdice:7.4f} {p_ari:7.4f}"
              f"  {s_miou:7.4f} {s_mdice:7.4f} {s_ari:7.4f}"
              f"  {t_miou:7.4f} {t_mdice:7.4f} {t_ari:7.4f}")

        print(f"\n  {'─'*68}")
        print(f"  {'':15s} {'CSeg PairOpt':>14s} {'SAM3+SemSeg':>14s} {'QwenTxt':>14s}")
        print(f"  {'─'*68}")
        print(f"  {'mIoU':15s} {p_miou:14.4f} {s_miou:14.4f} {t_miou:14.4f}")
        print(f"  {'mDice':15s} {p_mdice:14.4f} {s_mdice:14.4f} {t_mdice:14.4f}")
        print(f"  {'mARI':15s} {p_ari:14.4f} {s_ari:14.4f} {t_ari:14.4f}")
        print(f"  {'─'*68}")
        print(f"\n  Head-to-head (mIoU):")
        print(f"    CSeg PairOpt: {cseg_wins}  |  SAM3+SemSeg: {sam3d_wins}  |  "
              f"QwenTxt: {txt_wins}")

        # SAM3 DETR mask diversity (mean STD across 200 proposals)
        if sam3_valid:
            all_std_a = [r.get("mean_std_a", 0) for r in sam3_valid]
            all_std_b = [r.get("mean_std_b", 0) for r in sam3_valid]
            print(f"\n  {'─'*68}")
            print(f"  SAM3 DETR mask diversity (mean STD across 200 proposals):")
            print(f"    Per sample:")
            for r in sam3_valid:
                print(f"      {r['crop_name']:>8s}  "
                      f"STD_A={r.get('mean_std_a', 0):.4f}  "
                      f"STD_B={r.get('mean_std_b', 0):.4f}")
            print(f"    {'─'*40}")
            print(f"      {'MEAN':>8s}  "
                  f"STD_A={np.mean(all_std_a):.4f}  "
                  f"STD_B={np.mean(all_std_b):.4f}")
            overall_std = np.mean(all_std_a + all_std_b)
            if overall_std < 0.05:
                print(f"    >>> Very LOW diversity — DETR proposals are nearly identical")
            elif overall_std < 0.15:
                print(f"    >>> Moderate diversity among DETR proposals")
            else:
                print(f"    >>> Good diversity among DETR proposals")

    print(f"\n  Output: {output_dir}/")
    print(f"  SAM3 proposals: {sam3_proposals_dir}/")
    print(f"{'='*110}")

    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "pair_optimizer": results_log,
            "sam3_semseg_combined": sam3_results_log,
            "qwen_txt_baseline": txt_results_log,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
