"""
CLIPSeg heatmap diversity test — how much does the description matter?

Flow:
  1. Qwen3 generates 5 diverse descriptions per texture (forced different formulations)
  2. Each description → CLIPSeg heatmap
  3. Visualize all heatmaps side-by-side to see diversity across descriptions

This is a visualization-only test — no metrics, no SAM.
Just exploring how sensitive CLIPSeg is to description phrasing.

Usage:
  python qwen2sam/scripts/test_clipseg_description_diversity.py
  python qwen2sam/scripts/test_clipseg_description_diversity.py --samples 3,12,13
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

from qwen2sam.training.train_phase1 import load_config, set_seed


# ===================================================================== #
#  Prompts                                                                #
# ===================================================================== #

QWEN_SYSTEM_PROMPT = (
    "You analyze surface textures in images. Always respond in the exact "
    "format requested, with no extra text."
)

QWEN_DIVERSE_PROMPT = (
    "This image contains exactly TWO main visually distinct regions separated by a boundary.\n\n"
    "For each region, provide {n_desc} DIFFERENT descriptions. Each description must be a single "
    "phrase of 10-15 words that captures the region, but each one MUST use a COMPLETELY DIFFERENT "
    "angle or vocabulary:\n"
    "  - One focusing on COLOR and TONE\n"
    "  - One focusing on TEXTURE and PATTERN\n"
    "  - One focusing on MATERIAL and SURFACE type\n"
    "  - One focusing on SPATIAL CONTEXT and SHAPE\n"
    "  - One using EVERYDAY LANGUAGE (how a non-expert would describe it)\n\n"
    "IMPORTANT: Each description must be genuinely different — not just rephrasing the same words. "
    "Use diverse vocabulary. Describe the ENTIRE region as a surface/area.\n\n"
    "Format your response exactly like this:\n"
    "TEXTURE_A_1: Texture of <color/tone description>\n"
    "TEXTURE_A_2: Texture of <texture/pattern description>\n"
    "TEXTURE_A_3: Texture of <material/surface description>\n"
    "TEXTURE_A_4: Texture of <spatial/shape description>\n"
    "TEXTURE_A_5: Texture of <everyday language description>\n"
    "TEXTURE_B_1: Texture of <color/tone description>\n"
    "TEXTURE_B_2: Texture of <texture/pattern description>\n"
    "TEXTURE_B_3: Texture of <material/surface description>\n"
    "TEXTURE_B_4: Texture of <spatial/shape description>\n"
    "TEXTURE_B_5: Texture of <everyday language description>"
)

DEFAULT_SAMPLES = ["3", "12", "13", "15", "18", "22"]
N_DESCRIPTIONS = 5

DESC_LABELS = ["Color/Tone", "Texture/Pattern", "Material/Surface",
               "Spatial/Shape", "Everyday"]


# ===================================================================== #
#  Visualization helpers                                                #
# ===================================================================== #

def heatmap_to_bgr(heatmap, size=None):
    h = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    if size:
        colored = cv2.resize(colored, size, interpolation=cv2.INTER_LINEAR)
    return colored


def mask_overlay(image, mask, color=(0, 0, 220), alpha=0.45):
    vis = image.copy()
    overlay = image.copy()
    overlay[mask > 0.5] = color
    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)


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
#  Qwen3 diverse description generation                                   #
# ===================================================================== #

@torch.no_grad()
def qwen3_generate_diverse(model, processor, image_pil, device, n_desc=5):
    prompt = QWEN_DIVERSE_PROMPT.format(n_desc=n_desc)
    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image_pil], return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs, max_new_tokens=500, do_sample=False, temperature=1.0)
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, input_len:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True)


def parse_diverse_output(text, n_desc=5):
    """Parse N descriptions per texture from Qwen output."""
    descs_a = []
    descs_b = []
    for i in range(1, n_desc + 1):
        match_a = re.search(rf'TEXTURE_A_{i}:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        match_b = re.search(rf'TEXTURE_B_{i}:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        descs_a.append(match_a.group(1).strip() if match_a else "")
        descs_b.append(match_b.group(1).strip() if match_b else "")
    ok = all(descs_a) and all(descs_b)
    return descs_a, descs_b, ok


# ===================================================================== #
#  CLIPSeg heatmap extraction                                             #
# ===================================================================== #

@torch.no_grad()
def clipseg_heatmap(clipseg_model, clipseg_proc, image_pil, desc, device="cuda"):
    """Get CLIPSeg heatmap for a single description. Returns numpy [0,1]."""
    inputs = clipseg_proc(
        text=[desc], images=[image_pil],
        return_tensors="pt", padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.amp.autocast("cuda", enabled=False):
        outputs = clipseg_model.float()(**inputs)
    logits = outputs.logits.squeeze().float()
    heatmap = torch.sigmoid(logits).cpu().numpy()
    return heatmap


# ===================================================================== #
#  Visualization grid                                                     #
# ===================================================================== #

def draw_diversity_grid(image_bgr, gt_a, gt_b,
                        descs_a, descs_b,
                        heatmaps_a, heatmaps_b,
                        diff_maps,
                        wta_masks,
                        crop_name, cell_size=220):
    """
    Grid layout:
      Row 0: Image | GT | (header row)
      --- Texture A ---
      Row 1..N: heatmap_a_i | diff_a_i | WTA_a_i  (one per description)
      --- Texture B ---
      Row N+1..2N: heatmap_b_i | diff_b_i | WTA_b_i  (one per description)

    Columns: Description text | Raw Heatmap | Diff (A-B) | WTA Mask
    """
    h, w = image_bgr.shape[:2]
    s = cell_size / max(h, w)
    ch, cw = int(h * s), int(w * s)
    n_desc = len(descs_a)

    def ri(img):
        return cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)

    def rm(mask):
        return cv2.resize(mask.astype(np.float32), (cw, ch),
                          interpolation=cv2.INTER_NEAREST)

    img = ri(image_bgr)
    ga, gb = rm(gt_a), rm(gt_b)

    sep = 2
    text_col_w = 350  # width for description text column

    def make_text_cell(text, color=(255, 255, 255), bg=20):
        cell = np.zeros((ch, text_col_w, 3), dtype=np.uint8) + bg
        # Wrap text
        words = text.split()
        lines = []
        line = ""
        for word in words:
            test = f"{line} {word}".strip()
            if len(test) > 45:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)
        for li, l in enumerate(lines[:6]):
            cv2.putText(cell, l, (8, 20 + li * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        return cell

    # --- Build rows ---
    all_rows = []
    sep_h = np.zeros((sep, text_col_w + sep + cw * 3 + sep * 3, 3), dtype=np.uint8)

    # Header row: Image | GT overlay | GT binary | blank
    header_img = img.copy()
    gt_overlay = img.copy()
    overlay = img.copy()
    overlay[ga > 0.5] = (0, 0, 220)
    overlay[gb > 0.5] = (220, 80, 0)
    gt_overlay = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

    header_text = make_text_cell(f"Sample: {crop_name}", (0, 255, 255), 30)
    header_row = np.hstack([
        header_text,
        np.zeros((ch, sep, 3), dtype=np.uint8),
        header_img,
        np.zeros((ch, sep, 3), dtype=np.uint8),
        gt_overlay,
        np.zeros((ch, sep, 3), dtype=np.uint8),
        np.zeros((ch, cw, 3), dtype=np.uint8) + 30,
    ])
    all_rows.append(header_row)

    # Column headers
    col_header_h = 24
    total_w = header_row.shape[1]
    col_bar = np.zeros((col_header_h, total_w, 3), dtype=np.uint8) + 40
    labels = ["Description", "Raw Heatmap", "Diff Map", "WTA Mask"]
    x_positions = [8, text_col_w + sep + 8,
                   text_col_w + sep + cw + sep + 8,
                   text_col_w + sep + cw * 2 + sep * 2 + 8]
    for lbl, xp in zip(labels, x_positions):
        cv2.putText(col_bar, lbl, (xp, col_header_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    all_rows.append(col_bar)

    # Texture A section header
    sec_h = 22
    sec_a = np.zeros((sec_h, total_w, 3), dtype=np.uint8)
    sec_a[:, :] = (40, 0, 0)
    cv2.putText(sec_a, "TEXTURE A", (8, sec_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1, cv2.LINE_AA)
    all_rows.append(sec_a)

    # Texture A rows
    for i in range(n_desc):
        heat_vis = heatmap_to_bgr(heatmaps_a[i], (cw, ch))
        diff_vis = heatmap_to_bgr(diff_maps[i][0], (cw, ch))
        wta_vis = ri(np.stack([
            (wta_masks[i][0] * 220).astype(np.uint8),
            np.zeros_like(wta_masks[i][0], dtype=np.uint8),
            np.zeros_like(wta_masks[i][0], dtype=np.uint8),
        ], axis=-1))

        label = DESC_LABELS[i] if i < len(DESC_LABELS) else f"Desc {i+1}"
        text_cell = make_text_cell(f"[{label}] {descs_a[i]}", (150, 150, 255))

        row = np.hstack([
            text_cell,
            np.zeros((ch, sep, 3), dtype=np.uint8),
            heat_vis,
            np.zeros((ch, sep, 3), dtype=np.uint8),
            diff_vis,
            np.zeros((ch, sep, 3), dtype=np.uint8),
            wta_vis,
        ])
        all_rows.append(row)
        all_rows.append(np.zeros((1, total_w, 3), dtype=np.uint8))

    # Texture B section header
    sec_b = np.zeros((sec_h, total_w, 3), dtype=np.uint8)
    sec_b[:, :] = (0, 30, 40)
    cv2.putText(sec_b, "TEXTURE B", (8, sec_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1, cv2.LINE_AA)
    all_rows.append(sec_b)

    # Texture B rows
    for i in range(n_desc):
        heat_vis = heatmap_to_bgr(heatmaps_b[i], (cw, ch))
        diff_vis = heatmap_to_bgr(diff_maps[i][1], (cw, ch))
        wta_vis = ri(np.stack([
            np.zeros_like(wta_masks[i][1], dtype=np.uint8),
            (wta_masks[i][1] * 80).astype(np.uint8),
            (wta_masks[i][1] * 220).astype(np.uint8),
        ], axis=-1))

        label = DESC_LABELS[i] if i < len(DESC_LABELS) else f"Desc {i+1}"
        text_cell = make_text_cell(f"[{label}] {descs_b[i]}", (0, 180, 255))

        row = np.hstack([
            text_cell,
            np.zeros((ch, sep, 3), dtype=np.uint8),
            heat_vis,
            np.zeros((ch, sep, 3), dtype=np.uint8),
            diff_vis,
            np.zeros((ch, sep, 3), dtype=np.uint8),
            wta_vis,
        ])
        all_rows.append(row)
        all_rows.append(np.zeros((1, total_w, 3), dtype=np.uint8))

    grid = np.vstack(all_rows)
    return grid


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="CLIPSeg heatmap diversity across Qwen descriptions")
    parser.add_argument("--data_root", type=str, default="/home/aviad/RWTD")
    parser.add_argument("--samples", type=str, default=None,
                        help="Comma-separated crop names")
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/clipseg_diversity")
    parser.add_argument("--n_desc", type=int, default=N_DESCRIPTIONS,
                        help="Number of diverse descriptions per texture")
    args = parser.parse_args()

    sample_names = args.samples.split(",") if args.samples else DEFAULT_SAMPLES
    n_desc = args.n_desc

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata
    meta_path = Path(args.data_root) / "metadata_phase1.json"
    with open(meta_path) as f:
        all_meta = json.load(f)
    meta_by_name = {e["crop_name"]: e for e in all_meta}
    samples = [meta_by_name[n] for n in sample_names if n in meta_by_name]
    missing = [n for n in sample_names if n not in meta_by_name]
    if missing:
        print(f"Warning: samples not found: {missing}")
    print(f"Testing {len(samples)} samples, {n_desc} descriptions each")

    # Output dirs
    output_dir = _project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load models — only Qwen + CLIPSeg, no SAM
    qwen3, qwen3_proc = load_qwen3_model(device)
    clipseg_model, clipseg_proc = load_clipseg_model(device)

    outputs_log = []
    t0 = time.time()

    for i, entry in enumerate(samples):
        crop_name = entry["crop_name"]
        print(f"\n{'─'*60}")
        print(f"  [{crop_name}] ({i+1}/{len(samples)})")

        image_bgr = cv2.imread(entry["image_path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        gt_a = cv2.imread(entry["mask_a_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_b = cv2.imread(entry["mask_b_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        # --- Qwen3: generate diverse descriptions ---
        raw = qwen3_generate_diverse(qwen3, qwen3_proc, image_pil, device, n_desc)
        descs_a, descs_b, ok = parse_diverse_output(raw, n_desc)

        if not ok:
            print(f"    PARSE FAIL — raw output:")
            print(f"    {raw[:300]}")
            # Try to salvage what we can
            print(f"    Parsed A: {descs_a}")
            print(f"    Parsed B: {descs_b}")
            # Fill empty descriptions with placeholder
            descs_a = [d if d else f"texture region A variant {j+1}" for j, d in enumerate(descs_a)]
            descs_b = [d if d else f"texture region B variant {j+1}" for j, d in enumerate(descs_b)]

        print(f"  Texture A descriptions:")
        for j, d in enumerate(descs_a):
            label = DESC_LABELS[j] if j < len(DESC_LABELS) else f"Desc {j+1}"
            print(f"    [{label}] {d}")
        print(f"  Texture B descriptions:")
        for j, d in enumerate(descs_b):
            label = DESC_LABELS[j] if j < len(DESC_LABELS) else f"Desc {j+1}"
            print(f"    [{label}] {d}")

        # --- CLIPSeg: get heatmap for each description ---
        heatmaps_a = []
        heatmaps_b = []
        diff_maps = []   # list of (diff_a, diff_b) per description pair
        wta_masks = []   # list of (wta_a, wta_b) per description pair

        for j in range(n_desc):
            ha = clipseg_heatmap(clipseg_model, clipseg_proc, image_pil, descs_a[j], device)
            hb = clipseg_heatmap(clipseg_model, clipseg_proc, image_pil, descs_b[j], device)
            heatmaps_a.append(ha)
            heatmaps_b.append(hb)

            # Diff and WTA using this pair of descriptions
            da = np.clip(ha - hb, 0, 1)
            db = np.clip(hb - ha, 0, 1)
            d_max = max(da.max(), db.max(), 1e-8)
            diff_maps.append((da / d_max, db / d_max))

            wta_a = (ha > hb).astype(np.float32)
            wta_b = (hb > ha).astype(np.float32)
            wta_masks.append((wta_a, wta_b))

        # --- Compute heatmap variance across descriptions ---
        stack_a = np.stack(heatmaps_a, axis=0)  # (N, H, W)
        stack_b = np.stack(heatmaps_b, axis=0)
        var_a = stack_a.std(axis=0).mean()
        var_b = stack_b.std(axis=0).mean()
        print(f"  Heatmap std across descriptions:  A={var_a:.4f}  B={var_b:.4f}")

        outputs_log.append({
            "crop_name": crop_name,
            "raw_output": raw,
            "descs_a": descs_a,
            "descs_b": descs_b,
            "heatmap_std_a": float(var_a),
            "heatmap_std_b": float(var_b),
        })

        # --- Visualization ---
        grid = draw_diversity_grid(
            image_bgr, gt_a, gt_b,
            descs_a, descs_b,
            heatmaps_a, heatmaps_b,
            diff_maps, wta_masks,
            crop_name,
        )
        cv2.imwrite(str(vis_dir / f"{crop_name}_diversity.png"), grid)
        print(f"  Saved: {vis_dir / f'{crop_name}_diversity.png'}")

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Description Diversity Test — {len(samples)} samples ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Mean heatmap std across descriptions:")
    mean_std_a = np.mean([o["heatmap_std_a"] for o in outputs_log])
    mean_std_b = np.mean([o["heatmap_std_b"] for o in outputs_log])
    print(f"    Texture A: {mean_std_a:.4f}")
    print(f"    Texture B: {mean_std_b:.4f}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*60}")

    with open(output_dir / "diversity_log.json", "w") as f:
        json.dump(outputs_log, f, indent=2, default=str)


if __name__ == "__main__":
    main()
