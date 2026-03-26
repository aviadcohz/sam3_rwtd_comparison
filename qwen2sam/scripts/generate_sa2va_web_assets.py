"""
Generate web assets for Sa2VA comparison:
  1. Comparison figures: GT | Qw3 SemSeg | Sa2VA (per sample)
  2. Update results.json with Sa2VA + AutoSAM per-sample metrics
  3. Copy thumbnails to docs/assets/thumbnails/

Usage:
  python -m qwen2sam.scripts.generate_sa2va_web_assets
"""

import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_DIR = DOCS_DIR / "data"
THUMB_DIR = DOCS_DIR / "assets" / "thumbnails"
EVAL_DIR = PROJECT_ROOT / "eval_results"
RWTD_DIR = Path("/home/aviad/RWTD")

# Visualization colors (BGR)
COLOR_A = (0, 0, 220)
COLOR_B = (220, 80, 0)


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


def load_csv_metrics(csv_path):
    """Load per-sample metrics from CSV into dict keyed by crop_name."""
    metrics = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["crop_name"]] = {
                k: float(row[k]) for k in row if k != "crop_name"
            }
    return metrics


def generate_comparison_figures():
    """Generate GT | Qw3 SemSeg | Sa2VA comparison images."""
    print("Generating comparison figures...")

    metadata = json.load(open(RWTD_DIR / "metadata_phase1.json"))

    # Load Sa2VA per-sample metrics
    sa2va_metrics = load_csv_metrics(EVAL_DIR / "sa2va" / "metrics_sa2va.csv")

    # Load Qwen3 SemSeg per-sample metrics
    semseg_metrics = load_csv_metrics(
        EVAL_DIR / "sam3_oracle_points" / "metrics_qwen3_semseg.csv")

    # Load SAM3 oracle points visualizations dir
    oracle_vis_dir = EVAL_DIR / "sam3_oracle_points" / "visualizations"
    sa2va_vis_dir = EVAL_DIR / "sa2va" / "visualizations"

    cell_size = 256
    sep = 4
    lbl_h = 30

    def ri(img):
        h, w = img.shape[:2]
        s = cell_size / max(h, w)
        ch, cw = int(h * s), int(w * s)
        return cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)

    def label_bar(text, width):
        bar = np.zeros((lbl_h, width, 3), dtype=np.uint8) + 35
        cv2.putText(bar, text, (8, lbl_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1,
                    cv2.LINE_AA)
        return bar

    count = 0
    for entry in metadata:
        crop_name = entry["crop_name"]

        # Load original image
        image_path = entry.get("image_path", entry.get("image"))
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        # Load GT masks
        gt_a = np.array(Image.open(entry["mask_a_path"]).convert("L"))
        gt_b = np.array(Image.open(entry["mask_b_path"]).convert("L"))
        gt_a_f = (gt_a > 127).astype(np.float32)
        gt_b_f = (gt_b > 127).astype(np.float32)

        img = ri(image_bgr)
        ch, cw = img.shape[:2]

        def rm(mask):
            return cv2.resize(mask, (cw, ch),
                              interpolation=cv2.INTER_NEAREST)

        ga, gb = rm(gt_a_f), rm(gt_b_f)

        # --- GT column ---
        gt_overlay = mask_overlay(img, ga, gb)
        gt_masks = binary_mask_image(ga, gb, ch, cw)
        gt_lbl = label_bar("Ground Truth", cw)
        gt_col = np.vstack([gt_lbl, gt_overlay,
                            np.zeros((sep, cw, 3), dtype=np.uint8),
                            gt_masks])

        # --- Qw3 SemSeg column ---
        # Load from existing _04_qwen visualization (contains semseg)
        qwen_vis_path = oracle_vis_dir / f"{crop_name}_04_qwen.png"
        semseg_iou = semseg_metrics.get(crop_name, {}).get("mean_iou", 0)
        semseg_ari = semseg_metrics.get(crop_name, {}).get("ari", 0)

        if qwen_vis_path.exists():
            # Extract the SemSeg column from the existing 2-column image
            qwen_vis = cv2.imread(str(qwen_vis_path))
            if qwen_vis is not None:
                # The _04_qwen image has 2 columns: Proposal | SemSeg
                # Each column is cw wide + separators
                mid = qwen_vis.shape[1] // 2
                semseg_col_raw = qwen_vis[:, mid:, :]
                semseg_col = cv2.resize(semseg_col_raw,
                                        (cw, gt_col.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
            else:
                semseg_col = np.zeros_like(gt_col)
        else:
            semseg_col = np.zeros_like(gt_col)

        # Rebuild SemSeg column with proper label
        sem_lbl = label_bar(
            f"Qw3 SemSeg  mIoU:{semseg_iou:.3f}  ARI:{semseg_ari:.3f}", cw)
        # Try to use raw masks if we can rebuild
        # For now use the extracted column but replace label
        semseg_col[:lbl_h, :, :] = sem_lbl

        # --- Sa2VA column ---
        sa2va_iou = sa2va_metrics.get(crop_name, {}).get("mean_iou", 0)
        sa2va_ari = sa2va_metrics.get(crop_name, {}).get("ari", 0)
        sa2va_vis_path = sa2va_vis_dir / f"{crop_name}_06_sa2va.png"

        if sa2va_vis_path.exists():
            sa2va_raw = cv2.imread(str(sa2va_vis_path))
            if sa2va_raw is not None:
                # Extract the Sa2VA column (right half of the 2-column image)
                mid = sa2va_raw.shape[1] // 2
                sa2va_col = sa2va_raw[:, mid:, :]
                sa2va_col = cv2.resize(sa2va_col,
                                       (cw, gt_col.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
            else:
                sa2va_col = np.zeros_like(gt_col)
        else:
            sa2va_col = np.zeros_like(gt_col)

        # Replace label
        sa_lbl = label_bar(
            f"Sa2VA  mIoU:{sa2va_iou:.3f}  ARI:{sa2va_ari:.3f}", cw)
        sa2va_col[:lbl_h, :, :] = sa_lbl

        # --- Combine: GT | Qw3 SemSeg | Sa2VA ---
        spacer = np.zeros((gt_col.shape[0], sep * 2, 3), dtype=np.uint8)
        fig = np.hstack([gt_col, spacer, semseg_col, spacer, sa2va_col])

        # Save comparison figure
        out_path = THUMB_DIR / f"{crop_name}_07_sa2va_comparison.jpg"
        cv2.imwrite(str(out_path), fig, [cv2.IMWRITE_JPEG_QUALITY, 90])
        count += 1

    print(f"  Generated {count} comparison figures in {THUMB_DIR}")


def update_results_json():
    """Add Sa2VA and AutoSAM per-sample metrics to results.json."""
    print("Updating results.json...")

    results_path = DATA_DIR / "results.json"
    data = json.load(open(results_path))

    # Load Sa2VA metrics
    sa2va_metrics = load_csv_metrics(EVAL_DIR / "sa2va" / "metrics_sa2va.csv")
    sa2va_summary = json.load(
        open(EVAL_DIR / "sa2va" / "summary.json"))
    # Get the first (only) key
    sa2va_key = list(sa2va_summary.keys())[0]
    sa2va_sum = sa2va_summary[sa2va_key]

    # Load AutoSAM metrics
    autosam_metrics = load_csv_metrics(
        EVAL_DIR / "autosam" / "metrics_autosam.csv")
    autosam_summary = json.load(
        open(EVAL_DIR / "autosam" / "summary.json"))
    autosam_sum = autosam_summary.get("autosam", {})

    # Load Sa2VA text outputs for descriptions
    sa2va_outputs = json.load(
        open(EVAL_DIR / "sa2va" / "sa2va_outputs.json"))
    sa2va_text_map = {o["crop_name"]: o["text_output"] for o in sa2va_outputs}

    # Add to summary
    data["summary"]["sa2va"] = {
        "tag": "sa2va",
        "num_samples": sa2va_sum.get("num_samples", 253),
        "mean_iou": sa2va_sum["mean_iou"],
        "mean_iou_a": sa2va_sum.get("mean_iou_a", 0),
        "mean_iou_b": sa2va_sum.get("mean_iou_b", 0),
        "mean_dice": sa2va_sum["mean_dice"],
        "mean_ari": sa2va_sum["mean_ari"],
    }

    if "autosam" not in data["summary"]:
        data["summary"]["autosam"] = {
            "tag": "autosam",
            "num_samples": autosam_sum.get("num_samples", 253),
            "mean_iou": autosam_sum["mean_iou"],
            "mean_iou_a": autosam_sum.get("mean_iou_a", 0),
            "mean_iou_b": autosam_sum.get("mean_iou_b", 0),
            "mean_dice": autosam_sum["mean_dice"],
            "mean_ari": autosam_sum["mean_ari"],
        }

    # Add per-sample metrics
    for sample in data["samples"]:
        crop = sample["crop_name"]

        # Sa2VA
        if crop in sa2va_metrics:
            sample["metrics"]["sa2va"] = {
                "mean_iou": sa2va_metrics[crop]["mean_iou"]
            }

        # AutoSAM
        if crop in autosam_metrics:
            sample["metrics"]["autosam"] = {
                "mean_iou": autosam_metrics[crop]["mean_iou"]
            }

    # Save
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Updated {results_path}")
    print(f"  Sa2VA summary: mIoU={sa2va_sum['mean_iou']:.4f}")
    print(f"  AutoSAM summary: mIoU={autosam_sum['mean_iou']:.4f}")


if __name__ == "__main__":
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    generate_comparison_figures()
    update_results_json()
    print("Done!")
