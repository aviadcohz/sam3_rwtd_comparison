"""
SAM3 DETR proposal diversity test.

Why do SAM3 proposals always look the same? This test investigates by:
  1. Generic prompt "texture" → top 8 DETR proposals
  2. Qwen3 generates 3 diverse descriptions per texture →
     each description fed to SAM3 DETR → top 8 proposals

For each prompt we visualize the top-8 masks so we can see:
  - Are proposals different across different text prompts?
  - Or does SAM3 always return the same set of masks regardless of text?

Runs on a few samples only (visualization test).

Usage:
  python qwen2sam/scripts/test_sam3_proposal_diversity.py
  python qwen2sam/scripts/test_sam3_proposal_diversity.py --samples 3,12,13
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

from qwen2sam.models.qwen2sam_v3_tracker import Qwen2SAMv3Tracker
from qwen2sam.training.train_phase1 import load_config, set_seed
from qwen2sam.data.dataset_v2 import preprocess_image_for_sam3


# ===================================================================== #
#  Config                                                                 #
# ===================================================================== #

DEFAULT_SAMPLES = ["3", "12", "13", "15", "18", "22"]
TOP_K = 8

QWEN_SYSTEM_PROMPT = (
    "You analyze surface textures in images. Always respond in the exact "
    "format requested, with no extra text."
)

QWEN_DIVERSE_PROMPT = (
    "This image contains exactly TWO main visually distinct regions separated by a boundary.\n\n"
    "For each region, provide 3 DIFFERENT descriptions. Each description must be a single "
    "phrase of 10-15 words, but each MUST use COMPLETELY DIFFERENT vocabulary and angle:\n"
    "  - Description 1: Focus on COLOR, TONE, and BRIGHTNESS\n"
    "  - Description 2: Focus on TEXTURE, PATTERN, and SURFACE DETAIL\n"
    "  - Description 3: Focus on MATERIAL TYPE and EVERYDAY LANGUAGE\n\n"
    "IMPORTANT: Each description must be genuinely different — not rephrasing. "
    "Describe the ENTIRE region as a surface/area, not individual objects.\n\n"
    "Format exactly:\n"
    "TEXTURE_A_1: Texture of <color/tone>\n"
    "TEXTURE_A_2: Texture of <texture/pattern>\n"
    "TEXTURE_A_3: Texture of <material/everyday>\n"
    "TEXTURE_B_1: Texture of <color/tone>\n"
    "TEXTURE_B_2: Texture of <texture/pattern>\n"
    "TEXTURE_B_3: Texture of <material/everyday>"
)

N_DESCRIPTIONS = 3
DESC_LABELS = ["Color/Tone", "Texture/Pattern", "Material/Everyday"]


# ===================================================================== #
#  Helpers                                                                #
# ===================================================================== #

def compute_iou_np(pred, gt):
    pred_b, gt_b = pred > 0.5, gt > 0.5
    inter = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    return 0.0 if union == 0 else float(inter / union)


def heatmap_to_bgr(heatmap, size=None):
    h = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    if size:
        colored = cv2.resize(colored, size, interpolation=cv2.INTER_LINEAR)
    return colored


def postprocess_mask_to_np(mask_logit, gt_h, gt_w):
    if mask_logit.shape[-2:] != (gt_h, gt_w):
        mask_logit = F.interpolate(
            mask_logit[None, None].float() if mask_logit.ndim == 2
            else mask_logit.unsqueeze(0).float(),
            size=(gt_h, gt_w),
            mode="bilinear", align_corners=False,
        ).squeeze()
    return mask_logit.sigmoid().float().cpu().numpy()


@torch.no_grad()
def run_detr_full(model, backbone_out, text_feat, B, device):
    """Run DETR, return ALL proposals with scores."""
    prompt_bf = text_feat["prompt"].unsqueeze(0).expand(B, -1, -1)
    mask_bf = text_feat["mask"].unsqueeze(0).expand(B, -1)
    out = model.base._run_sam3_from_backbone(backbone_out, prompt_bf, mask_bf)
    scores = out["pred_logits"][0].squeeze(-1).sigmoid()
    all_masks = out["pred_masks"][0]
    return scores, all_masks


def compute_mask_similarity(masks_list):
    """Compute pairwise IoU between a list of binary masks to measure diversity."""
    n = len(masks_list)
    if n < 2:
        return 1.0
    ious = []
    for i in range(n):
        for j in range(i + 1, n):
            ious.append(compute_iou_np(masks_list[i], masks_list[j]))
    return float(np.mean(ious))


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


# ===================================================================== #
#  Qwen3 diverse description generation                                   #
# ===================================================================== #

@torch.no_grad()
def qwen3_generate_diverse(model, proc, image_pil, device):
    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": QWEN_DIVERSE_PROMPT},
        ]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image_pil], return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=400, do_sample=False, temperature=1.0)
    gen = out[0, inputs["input_ids"].shape[1]:]
    return proc.tokenizer.decode(gen, skip_special_tokens=True)


def parse_diverse_output(text):
    descs_a, descs_b = [], []
    for i in range(1, N_DESCRIPTIONS + 1):
        ma = re.search(rf'TEXTURE_A_{i}:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        mb = re.search(rf'TEXTURE_B_{i}:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        descs_a.append(ma.group(1).strip() if ma else "")
        descs_b.append(mb.group(1).strip() if mb else "")
    ok = all(descs_a) and all(descs_b)
    return descs_a, descs_b, ok


# ===================================================================== #
#  Visualization                                                          #
# ===================================================================== #

COLOR_A = (0, 0, 220)
COLOR_B = (220, 80, 0)


def draw_proposal_row(image_bgr, gt_mask, proposals, desc, label,
                      cell_size=180, gt_color=COLOR_A):
    """
    Single row: GT | Prop1 | Prop2 | ... | Prop8
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
    cell_labels = ["GT"]

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
        cell_labels.append(f"P{rank}")

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


def draw_sample_grid(image_bgr, gt_a, gt_b, sections, crop_name, cell_size=180):
    """
    sections: list of (label, desc, gt_mask, proposals, gt_color)
    Each section becomes a row.
    """
    h, w = image_bgr.shape[:2]
    s = cell_size / max(h, w)
    ch, cw = int(h * s), int(w * s)

    rows = []

    # Title
    # Estimate width from first section
    n_cols = 1 + TOP_K  # GT + proposals
    est_w = n_cols * (cw + 2)
    title_bar = np.zeros((26, est_w, 3), dtype=np.uint8) + 40
    cv2.putText(title_bar, f"Sample: {crop_name}  |  Top-{TOP_K} SAM3 DETR proposals per prompt",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    rows.append(title_bar)

    sep_h = 4
    for label, desc, gt_mask, proposals, gt_color in sections:
        row = draw_proposal_row(image_bgr, gt_mask, proposals, desc, label,
                                cell_size=cell_size, gt_color=gt_color)
        # Pad or crop to match title width
        if row.shape[1] < est_w:
            pad = np.zeros((row.shape[0], est_w - row.shape[1], 3), dtype=np.uint8)
            row = np.hstack([row, pad])
        elif row.shape[1] > est_w:
            est_w = row.shape[1]
            # Repad title
            if title_bar.shape[1] < est_w:
                pad = np.zeros((title_bar.shape[0], est_w - title_bar.shape[1], 3), dtype=np.uint8) + 40
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


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="SAM3 DETR proposal diversity test")
    parser.add_argument("--config", type=str,
                        default="qwen2sam/configs/v3_tracker_detexure.yaml")
    parser.add_argument("--data_root", type=str, default="/home/aviad/RWTD")
    parser.add_argument("--samples", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/sam3_proposal_diversity")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()

    sample_names = args.samples.split(",") if args.samples else DEFAULT_SAMPLES
    top_k = args.top_k

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _project_root / config_path
    cfg = load_config(str(config_path))
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg["model"].get("image_size", 1008)

    meta = json.load(open(Path(args.data_root) / "metadata_phase1.json"))
    meta_by_name = {e["crop_name"]: e for e in meta}
    samples = [meta_by_name[n] for n in sample_names if n in meta_by_name]
    print(f"Testing {len(samples)} samples, top-{top_k} proposals")

    output_dir = _project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    if "tracker" in cfg:
        cfg["tracker"].pop("v3_checkpoint", None)
    print("Loading SAM3...")
    model = Qwen2SAMv3Tracker(cfg, device=str(device))
    model.base.sam3.eval()

    qwen3, qwen3_proc = load_qwen3(device)

    outputs_log = []
    t0 = time.time()

    for i, entry in enumerate(samples):
        crop_name = entry["crop_name"]
        print(f"\n{'='*80}")
        print(f"  [{crop_name}] ({i+1}/{len(samples)})")
        print(f"{'='*80}")

        image_bgr = cv2.imread(entry["image_path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        gt_a = cv2.imread(entry["mask_a_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_b = cv2.imread(entry["mask_b_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_h, gt_w = gt_a.shape

        sam_img = preprocess_image_for_sam3(image_rgb, image_size).unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = model.base.sam3.backbone.forward_image(sam_img)
            backbone_out["img_batch_all_stages"] = sam_img

            # ---------------------------------------------------------- #
            #  Helper: run a prompt through SAM3 DETR, get top-K          #
            # ---------------------------------------------------------- #
            def get_top_proposals(desc, gt_mask):
                text_out = model.base.sam3.backbone.forward_text([desc], device=device)
                text_feat = {
                    "prompt": text_out["language_features"].squeeze(1),
                    "mask": text_out["language_mask"].squeeze(0),
                }
                scores, all_masks = run_detr_full(model, backbone_out, text_feat, 1, device)
                top_indices = scores.topk(min(top_k, len(scores))).indices

                proposals = []
                for rank, idx in enumerate(top_indices):
                    idx_val = idx.item()
                    conf = scores[idx_val].item()
                    mask_np = postprocess_mask_to_np(all_masks[idx_val], gt_h, gt_w)
                    mask_bin = (mask_np > 0.5).astype(np.float32)
                    gt_iou = compute_iou_np(mask_bin, gt_mask)
                    proposals.append((mask_bin, conf, gt_iou, rank + 1))
                return proposals

            # ---------------------------------------------------------- #
            #  1. Generic "texture" prompt                                 #
            # ---------------------------------------------------------- #
            print(f"\n  --- Generic prompt: \"texture\" ---")
            gen_props_a = get_top_proposals("texture", gt_a)
            gen_props_b = get_top_proposals("texture", gt_b)

            # Top-1 masks from generic prompt
            gen_top1_a = gen_props_a[0][0] if gen_props_a else np.zeros_like(gt_a)
            gen_top1_b = gen_props_b[0][0] if gen_props_b else np.zeros_like(gt_b)

            print(f"    Top-1 conf: {gen_props_a[0][1]:.3f} | GT-IoU(A): {gen_props_a[0][2]:.3f}")
            print(f"    Top-1 conf: {gen_props_b[0][1]:.3f} | GT-IoU(B): {gen_props_b[0][2]:.3f}")

            # Check diversity within generic proposals
            gen_masks_a = [p[0] for p in gen_props_a]
            gen_pairwise = compute_mask_similarity(gen_masks_a)
            print(f"    Pairwise IoU among top-{top_k} (A): {gen_pairwise:.3f} "
                  f"({'SIMILAR' if gen_pairwise > 0.7 else 'diverse'})")

            # ---------------------------------------------------------- #
            #  2. Qwen3 diverse descriptions → SAM3 proposals              #
            # ---------------------------------------------------------- #
            raw = qwen3_generate_diverse(qwen3, qwen3_proc, image_pil, device)
            descs_a, descs_b, ok = parse_diverse_output(raw)

            if not ok:
                print(f"    PARSE FAIL: {raw[:200]}")
                descs_a = [d if d else f"texture region A" for d in descs_a]
                descs_b = [d if d else f"texture region B" for d in descs_b]

            # Collect sections for visualization
            sections = []

            # Generic "texture" rows
            sections.append(("Generic→A", "texture", gt_a, gen_props_a, COLOR_A))
            sections.append(("Generic→B", "texture", gt_b, gen_props_b, COLOR_B))

            # Per-description rows
            all_top1_a = [gen_top1_a]  # collect top-1 from each prompt for cross-prompt diversity
            all_top1_b = [gen_top1_b]

            for j in range(N_DESCRIPTIONS):
                label = DESC_LABELS[j] if j < len(DESC_LABELS) else f"Desc {j+1}"

                # Texture A description → proposals (evaluated against GT-A)
                props_a = get_top_proposals(descs_a[j], gt_a)
                sections.append((f"A-{label}", descs_a[j], gt_a, props_a, COLOR_A))
                all_top1_a.append(props_a[0][0] if props_a else np.zeros_like(gt_a))

                print(f"\n  A [{label}]: \"{descs_a[j]}\"")
                print(f"    Top-1 conf: {props_a[0][1]:.3f} | GT-IoU: {props_a[0][2]:.3f}")

                # Texture B description → proposals (evaluated against GT-B)
                props_b = get_top_proposals(descs_b[j], gt_b)
                sections.append((f"B-{label}", descs_b[j], gt_b, props_b, COLOR_B))
                all_top1_b.append(props_b[0][0] if props_b else np.zeros_like(gt_b))

                print(f"  B [{label}]: \"{descs_b[j]}\"")
                print(f"    Top-1 conf: {props_b[0][1]:.3f} | GT-IoU: {props_b[0][2]:.3f}")

            # Cross-prompt diversity: IoU between top-1 masks from different prompts
            cross_div_a = compute_mask_similarity(all_top1_a)
            cross_div_b = compute_mask_similarity(all_top1_b)
            print(f"\n  Cross-prompt top-1 pairwise IoU:")
            print(f"    Texture A: {cross_div_a:.3f} ({'SAME MASK' if cross_div_a > 0.85 else 'some diversity' if cross_div_a > 0.5 else 'DIVERSE'})")
            print(f"    Texture B: {cross_div_b:.3f} ({'SAME MASK' if cross_div_b > 0.85 else 'some diversity' if cross_div_b > 0.5 else 'DIVERSE'})")

        outputs_log.append({
            "crop_name": crop_name,
            "descs_a": descs_a,
            "descs_b": descs_b,
            "generic_top1_iou_a": gen_props_a[0][2],
            "generic_top1_iou_b": gen_props_b[0][2],
            "cross_prompt_similarity_a": cross_div_a,
            "cross_prompt_similarity_b": cross_div_b,
        })

        # Visualization
        grid = draw_sample_grid(image_bgr, gt_a, gt_b, sections, crop_name)
        cv2.imwrite(str(output_dir / f"{crop_name}_proposals.png"), grid)
        print(f"  Saved: {output_dir / f'{crop_name}_proposals.png'}")

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*80}")
    print(f"  SAM3 Proposal Diversity — {len(samples)} samples ({elapsed:.1f}s)")
    print(f"{'='*80}")
    if outputs_log:
        mean_cross_a = np.mean([o["cross_prompt_similarity_a"] for o in outputs_log])
        mean_cross_b = np.mean([o["cross_prompt_similarity_b"] for o in outputs_log])
        print(f"  Mean cross-prompt top-1 similarity:")
        print(f"    Texture A: {mean_cross_a:.3f}")
        print(f"    Texture B: {mean_cross_b:.3f}")
        if mean_cross_a > 0.85 and mean_cross_b > 0.85:
            print(f"  >>> SAM3 DETR produces nearly IDENTICAL masks regardless of text prompt!")
        elif mean_cross_a > 0.5 and mean_cross_b > 0.5:
            print(f"  >>> SAM3 DETR shows SOME sensitivity to text prompts")
        else:
            print(f"  >>> SAM3 DETR shows GOOD diversity across text prompts")
    print(f"  Output: {output_dir}/")
    print(f"{'='*80}")

    with open(output_dir / "diversity_log.json", "w") as f:
        json.dump(outputs_log, f, indent=2, default=str)


if __name__ == "__main__":
    main()
