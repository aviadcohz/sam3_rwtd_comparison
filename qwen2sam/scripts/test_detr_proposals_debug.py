"""
DETR Proposals Deep Debug.

For problematic samples, visualize ALL top-10 DETR proposals to understand:
- Is the right mask among top-10 but not top-1?
- Does CLIPSeg-guided selection find a better proposal?
- What's the oracle-best proposal (highest IoU with GT)?

Visualization per sample:
  Row 1: Image | CLIPSeg diff A | GT | Top-1..Top-10 proposals for texture A
  Row 2: Image | CLIPSeg diff B | GT | Top-1..Top-10 proposals for texture B
  Each proposal shows: confidence score, CLIPSeg IoU, GT IoU

Usage (VS Code "Run Python File"):
  python qwen2sam/scripts/test_detr_proposals_debug.py
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
    "TEXTURE_A: Texture of <description>\n"
    "TEXTURE_B: Texture of <description>"
)

DEFAULT_SAMPLES = None  # ["13", "22", "44", "47", "48", "59", "107"]


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
    """Convert mask logit tensor to numpy binary mask at GT resolution."""
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
    """Run DETR, return ALL 200 proposals with scores + semantic mask."""
    prompt_bf = text_feat["prompt"].unsqueeze(0).expand(B, -1, -1)
    mask_bf = text_feat["mask"].unsqueeze(0).expand(B, -1)
    out = model.base._run_sam3_from_backbone(backbone_out, prompt_bf, mask_bf)
    scores = out["pred_logits"][0].squeeze(-1).sigmoid()  # (200,)
    all_masks = out["pred_masks"][0]  # (200, H, W)
    semantic_mask = out.get("semantic_mask", None)
    return scores, all_masks, semantic_mask


# ===================================================================== #
#  CLIPSeg                                                                #
# ===================================================================== #

@torch.no_grad()
def clipseg_get_diff(clipseg_model, clipseg_proc, image_pil,
                      desc_a, desc_b, img_size=256, device="cuda"):
    raw = []
    for desc in [desc_a, desc_b]:
        inputs = clipseg_proc(text=[desc], images=[image_pil],
                               return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.amp.autocast("cuda", enabled=False):
            outputs = clipseg_model.float()(**inputs)
        hm = torch.sigmoid(outputs.logits.squeeze().float())
        hm = F.interpolate(hm[None, None], size=(img_size, img_size),
                           mode="bilinear", align_corners=False
                           ).squeeze().cpu().numpy()
        raw.append(hm)
    diff_a = np.clip(raw[0] - raw[1], 0, 1)
    diff_b = np.clip(raw[1] - raw[0], 0, 1)
    return diff_a, diff_b


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
#  Visualization: Top-10 proposals grid                                   #
# ===================================================================== #

def draw_proposals_grid(image_bgr, gt_mask, proposals_info,
                        desc, texture_label, sem_mask_np=None, cell_size=240):
    """
    Draw a 3-column grid of proposals:
      Row 0: Image | GT overlay | SemSeg overlay
      Row 1: Prop1 | Prop2 | Prop3
      Row 2: Prop4 | Prop5 | Prop6
      ...

    proposals_info: list of (mask_np, conf, cseg_iou, gt_iou, rank)
    sem_mask_np: optional semantic segmentation mask (probability, 0-1)
    """
    COLS = 3
    h, w = image_bgr.shape[:2]
    s = cell_size / max(h, w)
    ch, cw = int(h * s), int(w * s)
    sep = 3
    lbl_h = 26
    font = 0.4

    def ri(img):
        return cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)

    def rm(m):
        return cv2.resize(m, (cw, ch), interpolation=cv2.INTER_NEAREST)

    def make_overlay(base, mask_r, color=(0, 0, 220), alpha=0.45):
        overlay = base.copy()
        overlay[mask_r > 0.5] = color
        return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

    def label_cell(text, w=cw):
        cell = np.zeros((lbl_h, w, 3), dtype=np.uint8) + 30
        cv2.putText(cell, text, (4, lbl_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (230, 230, 230), 1, cv2.LINE_AA)
        return cell

    img = ri(image_bgr)
    gt_r = rm(gt_mask)

    # -- Row 0: Image | GT overlay | SemSeg overlay --
    gt_overlay = make_overlay(img, gt_r)
    if sem_mask_np is not None:
        sem_r = rm(sem_mask_np)
        sem_overlay = make_overlay(img, sem_r, color=(0, 180, 0))
        sem_iou = compute_iou_np(sem_mask_np, gt_mask)
        sem_label = label_cell(f"SemSeg  IoU:{sem_iou:.3f}")
    else:
        sem_overlay = np.zeros_like(img)
        sem_label = label_cell("SemSeg (n/a)")

    row0_labels = np.hstack([
        label_cell(f"{texture_label}"),
        np.zeros((lbl_h, sep, 3), dtype=np.uint8),
        label_cell("GT Overlay"),
        np.zeros((lbl_h, sep, 3), dtype=np.uint8),
        sem_label,
    ])
    row0_imgs = np.hstack([
        img,
        np.zeros((ch, sep, 3), dtype=np.uint8),
        gt_overlay,
        np.zeros((ch, sep, 3), dtype=np.uint8),
        sem_overlay,
    ])

    target_w = row0_imgs.shape[1]

    # -- Proposal cells --
    prop_cells = []
    for mask_np, conf, cseg_iou, gt_iou, rank in proposals_info:
        mask_r = rm(mask_np)
        cell = make_overlay(img, mask_r)

        # Color-code border by GT IoU
        g = int(min(gt_iou * 255, 255))
        r = int(max(0, (1 - gt_iou) * 255))
        cv2.rectangle(cell, (0, 0), (cw - 1, ch - 1), (0, g, r), 3)

        # Score annotations
        cv2.putText(cell, f"#{rank} conf:{conf:.2f}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(cell, f"GT IoU:{gt_iou:.3f}", (4, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

        lbl = label_cell(f"Proposal #{rank}  conf:{conf:.2f}")
        prop_cells.append(np.vstack([lbl, cell]))

    # -- Arrange proposals in 3-column grid rows --
    prop_rows = []
    for i in range(0, len(prop_cells), COLS):
        chunk = prop_cells[i:i + COLS]
        parts = []
        for c in chunk:
            parts.append(c)
            parts.append(np.zeros((c.shape[0], sep, 3), dtype=np.uint8))
        row = np.hstack(parts[:-1])
        # Pad to target width
        if row.shape[1] < target_w:
            pad = np.zeros((row.shape[0], target_w - row.shape[1], 3), dtype=np.uint8)
            row = np.hstack([row, pad])
        prop_rows.append(row)

    # -- Description header --
    desc_bar = np.zeros((22, target_w, 3), dtype=np.uint8) + 20
    max_chars = target_w // 6
    cv2.putText(desc_bar, f'"{desc}"'[:max_chars], (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 255), 1, cv2.LINE_AA)

    row_sep = np.zeros((3, target_w, 3), dtype=np.uint8)

    all_parts = [desc_bar, row0_labels, row0_imgs, row_sep]
    for pr in prop_rows:
        all_parts.extend([pr, row_sep])

    return np.vstack(all_parts[:-1])


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="DETR proposals debug")
    parser.add_argument("--config", type=str,
                        default="qwen2sam/configs/v3_tracker_detexure.yaml")
    parser.add_argument("--data_root", type=str, default="/home/aviad/RWTD")
    parser.add_argument("--samples", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/detr_proposals_debug")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--fixed_prompt", type=str, default=None,
                        help="Use a fixed text prompt (e.g. 'texture') instead of Qwen3. "
                             "Skips Qwen3 and CLIPSeg. Runs on ALL samples by default.")
    args = parser.parse_args()

    if args.samples:
        sample_names = args.samples.split(",")
    elif args.fixed_prompt:
        sample_names = None  # use all samples
    else:
        sample_names = DEFAULT_SAMPLES

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
        samples = meta
    else:
        samples = [meta_by_name[n] for n in sample_names if n in meta_by_name]
    print(f"Debugging {len(samples)} samples"
          f"{': ' + str([s['crop_name'] for s in samples]) if len(samples) <= 20 else ''}")

    output_dir = _project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    if "tracker" in cfg:
        cfg["tracker"].pop("v3_checkpoint", None)
    print("Loading SAM3...")
    model = Qwen2SAMv3Tracker(cfg, device=str(device))
    model.base.sam3.eval()

    if args.fixed_prompt:
        print(f"Fixed prompt mode: \"{args.fixed_prompt}\" — skipping Qwen3 & CLIPSeg")
        qwen3 = qwen3_proc = cseg_model = cseg_proc = None
    else:
        qwen3, qwen3_proc = load_qwen3(device)
        cseg_model, cseg_proc = load_clipseg(device)

    for entry in samples:
        crop_name = entry["crop_name"]
        print(f"\n{'='*80}")
        print(f"  Sample: {crop_name}")
        print(f"{'='*80}")

        image_bgr = cv2.imread(entry["image_path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        gt_a = cv2.imread(entry["mask_a_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_b = cv2.imread(entry["mask_b_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_h, gt_w = gt_a.shape

        image_pil = Image.fromarray(image_rgb)
        sam_img = preprocess_image_for_sam3(image_rgb, image_size).unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = model.base.sam3.backbone.forward_image(sam_img)
            backbone_out["img_batch_all_stages"] = sam_img

            if args.fixed_prompt:
                # ── Fixed-prompt mode: single prompt, no Qwen3/CLIPSeg ──
                desc = args.fixed_prompt
                print(f"  Prompt: \"{desc}\"")

                text_out = model.base.sam3.backbone.forward_text([desc], device=device)
                text_feat = {
                    "prompt": text_out["language_features"].squeeze(1),
                    "mask": text_out["language_mask"].squeeze(0),
                }
                scores, all_masks, sem_mask = run_detr_full(model, backbone_out, text_feat, 1, device)

                # Semantic seg mask
                sem_np = None
                if sem_mask is not None:
                    sem_np = postprocess_mask_to_np(sem_mask[0, 0], gt_h, gt_w)
                    sem_np = (sem_np > 0.5).astype(np.float32)

                top_k = min(args.top_k, len(scores))
                top_indices = scores.topk(top_k).indices

                proposals_info = []
                print(f"  {'Rank':>4s} {'Idx':>4s} {'Conf':>6s} {'GT_A':>6s} {'GT_B':>6s} {'Area%':>6s}")
                print(f"  {'─'*40}")

                for rank, idx in enumerate(top_indices):
                    idx_val = idx.item()
                    conf = scores[idx_val].item()
                    mask_np = postprocess_mask_to_np(all_masks[idx_val], gt_h, gt_w)
                    gt_iou_a = compute_iou_np(mask_np, gt_a)
                    gt_iou_b = compute_iou_np(mask_np, gt_b)
                    area_pct = (mask_np > 0.5).sum() / (gt_h * gt_w) * 100

                    proposals_info.append((
                        (mask_np > 0.5).astype(np.float32),
                        conf, 0.0, max(gt_iou_a, gt_iou_b), rank + 1
                    ))
                    print(f"  {rank+1:4d} {idx_val:4d} {conf:6.3f} {gt_iou_a:6.3f} {gt_iou_b:6.3f} {area_pct:5.1f}%")

                grid = draw_proposals_grid(
                    image_bgr, gt_a, proposals_info,
                    desc, f"{crop_name}",
                    sem_mask_np=sem_np,
                )
                cv2.imwrite(str(output_dir / f"{crop_name}_fixed_proposals.png"), grid)

            else:
                # ── Original mode: Qwen3 + CLIPSeg ──
                raw = qwen3_generate(qwen3, qwen3_proc, image_pil, device)
                desc_a, desc_b, ok = parse_text(raw)
                print(f"  Oracle A: {entry['texture_a']}")
                print(f"  Oracle B: {entry['texture_b']}")
                print(f"  Qwen A:   {desc_a}")
                print(f"  Qwen B:   {desc_b}")

                if not ok:
                    print("  PARSE FAIL — skipping")
                    continue

                # CLIPSeg diff
                diff_a, diff_b = clipseg_get_diff(
                    cseg_model, cseg_proc, image_pil,
                    desc_a, desc_b, img_size=orig_w, device=device)
                cseg_binary_a = (diff_a > 0.1).astype(np.float32)
                cseg_binary_b = (diff_b > 0.1).astype(np.float32)

                # DETR proposals for each texture
                for tex_label, desc, gt_mask, cseg_binary in [
                    ("A", desc_a, gt_a, cseg_binary_a),
                    ("B", desc_b, gt_b, cseg_binary_b),
                ]:
                    text_out = model.base.sam3.backbone.forward_text([desc], device=device)
                    text_feat = {
                        "prompt": text_out["language_features"].squeeze(1),
                        "mask": text_out["language_mask"].squeeze(0),
                    }
                    scores, all_masks, sem_mask = run_detr_full(model, backbone_out, text_feat, 1, device)

                    # Semantic seg mask
                    sem_np = None
                    if sem_mask is not None:
                        sem_np = postprocess_mask_to_np(sem_mask[0, 0], gt_h, gt_w)
                        sem_np = (sem_np > 0.5).astype(np.float32)

                    # Get top-K
                    top_k = min(args.top_k, len(scores))
                    top_indices = scores.topk(top_k).indices

                    # Analyze each proposal
                    proposals_info = []
                    print(f"\n  Texture {tex_label}: \"{desc}\"")
                    print(f"  {'Rank':>4s} {'Idx':>4s} {'Conf':>6s} {'CSeg':>6s} {'GT':>6s} {'Area%':>6s}")
                    print(f"  {'─'*35}")

                    for rank, idx in enumerate(top_indices):
                        idx_val = idx.item()
                        conf = scores[idx_val].item()

                        # Convert to numpy at GT resolution
                        mask_np = postprocess_mask_to_np(all_masks[idx_val], gt_h, gt_w)

                        # CLIPSeg IoU
                        cseg_iou = compute_iou_np(mask_np, cseg_binary)

                        # GT IoU
                        gt_iou = compute_iou_np(mask_np, gt_mask)

                        # Area
                        area_pct = (mask_np > 0.5).sum() / (gt_h * gt_w) * 100

                        proposals_info.append((
                            (mask_np > 0.5).astype(np.float32),
                            conf, cseg_iou, gt_iou, rank + 1
                        ))

                        marker = " <<<" if rank == 0 else ""
                        best_cseg = " [best-cseg]" if cseg_iou == max(p[2] for p in proposals_info) else ""
                        best_gt = " [best-gt]" if gt_iou == max(p[3] for p in proposals_info) else ""
                        print(f"  {rank+1:4d} {idx_val:4d} {conf:6.3f} {cseg_iou:6.3f} {gt_iou:6.3f} {area_pct:5.1f}%{marker}{best_cseg}{best_gt}")

                    # Summary
                    best_gt_rank = max(range(len(proposals_info)), key=lambda i: proposals_info[i][3])
                    best_cseg_rank = max(range(len(proposals_info)), key=lambda i: proposals_info[i][2])
                    print(f"\n  Best GT IoU: rank {best_gt_rank+1} ({proposals_info[best_gt_rank][3]:.3f})")
                    print(f"  Best CSeg IoU: rank {best_cseg_rank+1} ({proposals_info[best_cseg_rank][2]:.3f})")
                    if best_gt_rank != 0:
                        print(f"  *** Top-1 is NOT the best! Rank {best_gt_rank+1} has {proposals_info[best_gt_rank][3]:.3f} vs top-1 {proposals_info[0][3]:.3f}")

                    # Visualization
                    grid = draw_proposals_grid(
                        image_bgr, gt_mask, proposals_info,
                        desc, f"{crop_name}_{tex_label}",
                        sem_mask_np=sem_np,
                    )
                    cv2.imwrite(str(output_dir / f"{crop_name}_{tex_label}_proposals.png"), grid)

    print(f"\n{'='*80}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
