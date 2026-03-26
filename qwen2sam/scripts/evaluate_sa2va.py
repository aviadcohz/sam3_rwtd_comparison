"""
Sa2VA Evaluation on RWTD — Single-Pass Direct Segmentation.

Tests whether Sa2VA (SAM2 + VLM) can autonomously identify and segment
both textures in one forward pass, without being given descriptions.

The model receives a single prompt asking it to segment both textures.
It generates text with [SEG] tokens, and SAM2 produces a mask for each.

This is a fair comparison to the Qwen3+SAM3 pipeline where Qwen describes
textures on its own and SAM3 segments them.

Usage:
  python -m qwen2sam.scripts.evaluate_sa2va \
      --data_root /home/aviad/RWTD \
      --output_dir eval_results/sa2va \
      --model_name ByteDance/Sa2VA-Qwen3-VL-4B
"""

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ===================================================================== #
#  Path setup                                                             #
# ===================================================================== #
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT.parent / "real_world_texture_boundary"))

from transformers import AutoModel, AutoProcessor, AutoTokenizer

from qwen2sam.scripts.evaluate_v2 import (
    compute_sample_metrics, aggregate_metrics, save_metrics_csv,
    mask_overlay, binary_mask_image,
)


# ===================================================================== #
#  Configurable Sa2VA model                                               #
# ===================================================================== #

class ConfigurableSa2VA:
    """Sa2VA model with configurable HuggingFace model ID."""

    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load_model(self):
        if self.model is not None:
            return
        print(f"\n{'='*70}")
        print(f"Loading Sa2VA: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"{'='*70}")

        # Patch PreTrainedModel for transformers 5.x compat:
        # Sa2VA model class lacks `all_tied_weights_keys` attribute
        from transformers.modeling_utils import PreTrainedModel
        _orig_init_weights = PreTrainedModel.mark_tied_weights_as_initialized
        def _safe_mark_tied(self_model):
            if not hasattr(self_model, 'all_tied_weights_keys'):
                self_model.all_tied_weights_keys = getattr(
                    self_model, '_tied_weights_keys', None) or {}
            return _orig_init_weights(self_model)
        PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark_tied

        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        ).to(self.device)

        # Restore original method
        PreTrainedModel.mark_tied_weights_as_initialized = _orig_init_weights
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True,
            )
        except Exception:
            self.tokenizer = getattr(self.processor, "tokenizer", None)

        print(f"Sa2VA loaded successfully\n{'='*70}\n")

    def unload(self):
        del self.model, self.processor, self.tokenizer
        self.model = self.processor = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def segment_textures_direct(self, image: Image.Image, prompt: str):
        """
        Single-pass segmentation: ask Sa2VA to identify and segment textures.

        Returns:
            text_output: str — the model's text response
            masks: list of np.ndarray — one mask per [SEG] token
        """
        self.load_model()

        result = self.model.predict_forward(
            image=image,
            text=prompt,
            past_text="",
            mask_prompts=None,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        text_output = result.get("prediction", "")
        raw_masks = result.get("prediction_masks", [])

        # Convert masks to numpy uint8
        masks = []
        orig_w, orig_h = image.size
        for m in raw_masks:
            m_np = m if isinstance(m, np.ndarray) else np.array(m)
            if m_np.ndim == 3:
                m_np = m_np[0]
            if m_np.dtype == bool or m_np.dtype == np.bool_:
                m_np = m_np.astype(np.uint8) * 255
            elif m_np.max() <= 1.0:
                m_np = (m_np * 255).astype(np.uint8)
            else:
                m_np = m_np.astype(np.uint8)
            if m_np.shape != (orig_h, orig_w):
                m_np = cv2.resize(m_np, (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST)
            masks.append(m_np)

        return text_output, masks


# ===================================================================== #
#  Prompt variants                                                        #
# ===================================================================== #

PROMPTS = {
    "v1": (
        "<image>This image shows two different textures meeting at a boundary. "
        "Identify and segment each texture."
    ),
    "v2": "<image>Segment the two different textures in this image.",
    "v3": (
        "<image>This image contains exactly two texture regions. "
        "Please segment each texture region separately."
    ),
    "v4": (
        "<image>This image shows two textures separated by a boundary. "
        "First, segment the first texture region. "
        "Then, segment the second texture region."
    ),
    "v5": (
        "<image>There are two distinct texture regions in this image. "
        "Segment the first texture and then segment the second texture. "
        "Provide two separate segmentation masks."
    ),
    "v6": (
        "<image>Can you segment both texture regions in this image? "
        "The image contains exactly two textures meeting at a boundary. "
        "Please provide a segmentation mask for each texture."
    ),
}

DEFAULT_PROMPT_KEY = "v1"


# ===================================================================== #
#  Visualization                                                          #
# ===================================================================== #

def save_sa2va_vis(image_bgr, gt_a, gt_b, pred_a, pred_b, met,
                   crop_name, vis_dir, cell_size=256):
    h, w = image_bgr.shape[:2]
    s = cell_size / max(h, w)
    ch, cw = int(h * s), int(w * s)
    sep = 4
    lbl_h = 30

    def ri(img):
        return cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)

    def rm(mask):
        return cv2.resize(mask.astype(np.float32), (cw, ch),
                          interpolation=cv2.INTER_NEAREST)

    def label_bar(text, width):
        bar = np.zeros((lbl_h, width, 3), dtype=np.uint8) + 35
        cv2.putText(bar, text, (8, lbl_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1,
                    cv2.LINE_AA)
        return bar

    img = ri(image_bgr)
    ga, gb = rm(gt_a), rm(gt_b)
    pa, pb = rm(pred_a), rm(pred_b)

    iou = met.get("mean_iou", 0)
    ari = met.get("ari", 0)

    # GT column
    gt_overlay = mask_overlay(img, ga, gb)
    gt_masks = binary_mask_image(ga, gb, ch, cw)
    gt_lbl = label_bar("Ground Truth", cw)
    gt_col = np.vstack([gt_lbl, gt_overlay,
                        np.zeros((sep, cw, 3), dtype=np.uint8), gt_masks])

    # Sa2VA column
    sa_overlay = mask_overlay(img, pa, pb)
    sa_masks = binary_mask_image(pa, pb, ch, cw)
    sa_lbl = label_bar(f"Sa2VA  mIoU:{iou:.3f}  ARI:{ari:.3f}", cw)
    sa_col = np.vstack([sa_lbl, sa_overlay,
                        np.zeros((sep, cw, 3), dtype=np.uint8), sa_masks])

    spacer = np.zeros((gt_col.shape[0], sep * 2, 3), dtype=np.uint8)
    fig = np.hstack([gt_col, spacer, sa_col])
    cv2.imwrite(str(vis_dir / f"{crop_name}_06_sa2va.png"), fig)


# ===================================================================== #
#  Main evaluation                                                        #
# ===================================================================== #

def evaluate(args):
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_path = data_root / args.metadata_file
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} samples from {metadata_path}")

    # Load model
    sa2va = ConfigurableSa2VA(args.model_name, device=args.device)
    sa2va.load_model()

    prompt = PROMPTS.get(args.prompt_key, PROMPTS[DEFAULT_PROMPT_KEY])
    print(f"Prompt ({args.prompt_key}): {prompt[:80]}...")

    all_metrics = []
    sa2va_outputs = []
    t0 = time.time()

    for i, entry in enumerate(metadata):
        crop_name = entry["crop_name"]
        image_path = entry.get("image_path", entry.get("image"))

        # Load image
        pil_image = Image.open(image_path).convert("RGB")

        # Load GT masks
        gt_a = np.array(Image.open(entry["mask_a_path"]).convert("L"))
        gt_b = np.array(Image.open(entry["mask_b_path"]).convert("L"))
        gt_a = (gt_a > 127).astype(np.float32)
        gt_b = (gt_b > 127).astype(np.float32)

        # Single-pass Sa2VA inference
        with torch.no_grad():
            text_output, masks = sa2va.segment_textures_direct(
                pil_image, prompt)

        num_masks = len(masks)

        # Handle edge cases
        h, w = gt_a.shape
        if num_masks == 0:
            pred_a = np.zeros((h, w), dtype=np.float32)
            pred_b = np.zeros((h, w), dtype=np.float32)
        elif num_masks == 1:
            pred_a = (masks[0] > 127).astype(np.float32)
            pred_b = 1.0 - pred_a
        else:
            # Take first 2 masks
            pred_a = (masks[0] > 127).astype(np.float32)
            pred_b = (masks[1] > 127).astype(np.float32)

        # Metrics (Hungarian matching inside)
        met = compute_sample_metrics(pred_a, pred_b, gt_a, gt_b, crop_name)
        all_metrics.append(met)

        # Record output
        sa2va_outputs.append({
            "crop_name": crop_name,
            "text_output": text_output,
            "num_masks": num_masks,
            "mean_iou": met["mean_iou"],
        })

        # Visualization
        if not args.no_vis:
            image_bgr = cv2.imread(str(image_path))
            save_sa2va_vis(image_bgr, gt_a, gt_b, pred_a, pred_b, met,
                           crop_name, vis_dir, args.cell_size)

        if (i + 1) % 10 == 0 or (i + 1) == len(metadata):
            avg_iou = np.mean([m["mean_iou"] for m in all_metrics])
            print(f"  [{i+1}/{len(metadata)}] masks={num_masks} "
                  f"mIoU={met['mean_iou']:.3f} "
                  f"avg_mIoU={avg_iou:.4f}")

    elapsed = time.time() - t0

    # Save metrics CSV
    save_metrics_csv(all_metrics, output_dir / "metrics_sa2va.csv")

    # Save text outputs
    with open(output_dir / "sa2va_outputs.json", "w") as f:
        json.dump(sa2va_outputs, f, indent=2)

    # Summary
    tag = f"sa2va_{args.prompt_key}"
    summary_data = aggregate_metrics(all_metrics, tag)

    # Add mask stats
    mask_counts = [o["num_masks"] for o in sa2va_outputs]
    summary_data["mask_count_stats"] = {
        "0_masks": mask_counts.count(0),
        "1_mask": mask_counts.count(1),
        "2_masks": mask_counts.count(2),
        "3+_masks": sum(1 for c in mask_counts if c >= 3),
    }

    summary = {tag: summary_data}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print(f"\n{'='*70}")
    print(f"  Sa2VA Evaluation — {len(metadata)} samples ({elapsed:.1f}s)")
    print(f"  Model: {args.model_name}")
    print(f"  Prompt: {args.prompt_key}")
    print(f"{'='*70}")
    print(f"  Mean IoU:  {summary_data['mean_iou']:.4f}")
    print(f"  Mean Dice: {summary_data['mean_dice']:.4f}")
    print(f"  Mean ARI:  {summary_data['mean_ari']:.4f}")
    print(f"  Mask counts: {summary_data['mask_count_stats']}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*70}")

    sa2va.unload()
    return all_metrics


# ===================================================================== #
#  CLI                                                                     #
# ===================================================================== #

def parse_args():
    p = argparse.ArgumentParser(description="Sa2VA evaluation on RWTD")
    p.add_argument("--data_root", type=str, default="/home/aviad/RWTD")
    p.add_argument("--metadata_file", type=str, default="metadata_phase1.json")
    p.add_argument("--output_dir", type=str, default="eval_results/sa2va")
    p.add_argument("--model_name", type=str,
                   default="ByteDance/Sa2VA-Qwen3-VL-4B")
    p.add_argument("--prompt_key", type=str, default=DEFAULT_PROMPT_KEY,
                   choices=list(PROMPTS.keys()),
                   help="Prompt variant to use")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no_vis", action="store_true",
                   help="Skip visualization generation")
    p.add_argument("--cell_size", type=int, default=256)
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of samples (for quick testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Apply sample limit if set
    if args.limit:
        data_root = Path(args.data_root)
        with open(data_root / args.metadata_file) as f:
            full_meta = json.load(f)
        limited_meta = full_meta[:args.limit]
        # Write temp limited metadata
        tmp_meta = Path(args.output_dir) / "_tmp_metadata.json"
        tmp_meta.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_meta, "w") as f:
            json.dump(limited_meta, f)
        args.data_root = str(tmp_meta.parent)
        args.metadata_file = tmp_meta.name

    evaluate(args)
