"""
Generate web assets for Sa2VA comparison:
  1. Comparison figures: GT | Qw3 SemSeg | Sa2VA (per sample)
     All 3 columns built uniformly: label bar + overlay (top) + binary mask (bottom)
  2. Update results.json with Sa2VA + AutoSAM per-sample metrics

Usage:
  python -m qwen2sam.scripts.generate_sa2va_web_assets
"""

import csv
import json
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


def _find_separator_groups(image_or_slice, axis):
    """Find groups of black separator pixels along an axis.
    axis=1 → vertical separators (sum over rows), axis=0 → horizontal (sum over cols)."""
    if axis == 1:
        sums = image_or_slice.sum(axis=(0, 2))
    else:
        sums = image_or_slice.sum(axis=(1, 2))
    black = np.where(sums == 0)[0]
    if len(black) == 0:
        return []
    diffs = np.diff(black)
    breaks = np.where(diffs > 1)[0]
    return np.split(black, breaks + 1) if len(breaks) > 0 else [black]


def extract_pred_masks_from_qwen_vis(vis_path, target_size=256):
    """
    Extract SemSeg prediction masks from _04_qwen.png.
    Layout: [Qw3Prop block] 12px-sep [Qw3Sem block]
    Each block: label(30px) on top, then [overlay 4px-sep mask] side by side.
    """
    vis = cv2.imread(str(vis_path))
    if vis is None:
        return None, None

    # Find the main separator (widest, ~12px) between Proposal and SemSeg blocks
    groups = _find_separator_groups(vis, axis=1)
    if not groups:
        return None, None
    main_sep = max(groups, key=len)
    semseg_block = vis[:, main_sep[-1] + 1:]

    # Within SemSeg block: skip 30px label, find internal 4px separator
    content = semseg_block[30:]
    int_groups = _find_separator_groups(content, axis=1)
    if not int_groups:
        return None, None
    int_sep = int_groups[0]

    # Mask region is to the right of internal separator
    mask_region = content[:, int_sep[-1] + 1:]
    return _decode_color_masks(mask_region, target_size)


def extract_pred_masks_from_sa2va_vis(vis_path, target_size=256):
    """
    Extract Sa2VA prediction masks from _06_sa2va.png.
    Layout: [GT col] sep [Sa2VA col]
    Each col: label(30px) + overlay(H) + sep(4px) + binary_mask(H) stacked vertically.
    """
    vis = cv2.imread(str(vis_path))
    if vis is None:
        return None, None

    # Find main vertical separator
    groups = _find_separator_groups(vis, axis=1)
    if not groups:
        return None, None
    main_sep = max(groups, key=len)
    sa2va_col = vis[:, main_sep[-1] + 1:]

    # Layout is vertical: label(30) + overlay + sep + mask
    # Find horizontal separator (black rows) after skipping label
    h = sa2va_col.shape[0]
    lbl_h = 30
    content = sa2va_col[lbl_h:]
    row_means = content.mean(axis=(1, 2))

    # Find first dark row (separator between overlay and mask, mean < 5)
    sep_start = None
    for r in range(content.shape[0] // 3, content.shape[0]):
        if row_means[r] < 5:
            sep_start = r
            break

    if sep_start is None:
        # No clear separator — take bottom half of content as mask
        mask_start = content.shape[0] // 2
    else:
        mask_start = sep_start
        while mask_start < content.shape[0] and row_means[mask_start] < 5:
            mask_start += 1

    mask_region = content[mask_start:]
    return _decode_color_masks(mask_region, target_size)


def _decode_color_masks(mask_region, target_size):
    """Decode COLOR_A (red) and COLOR_B (blue) from a BGR mask image."""
    # COLOR_A = BGR(0, 0, 220) → high red channel (2), low blue (0)
    # COLOR_B = BGR(220, 80, 0) → high blue channel (0), low red (2)
    mask_a = ((mask_region[:, :, 2] > 150) & (mask_region[:, :, 0] < 100)).astype(np.float32)
    mask_b = ((mask_region[:, :, 0] > 150) & (mask_region[:, :, 2] < 100)).astype(np.float32)

    if mask_a.shape[0] != target_size or mask_a.shape[1] != target_size:
        mask_a = cv2.resize(mask_a, (target_size, target_size),
                            interpolation=cv2.INTER_NEAREST)
        mask_b = cv2.resize(mask_b, (target_size, target_size),
                            interpolation=cv2.INTER_NEAREST)
    return mask_a, mask_b


def generate_comparison_figures():
    """Generate GT | Qw3 SemSeg | Sa2VA comparison images.
    All 3 columns built uniformly at the same size."""
    print("Generating comparison figures...")

    metadata = json.load(open(RWTD_DIR / "metadata_phase1.json"))

    sa2va_metrics = load_csv_metrics(EVAL_DIR / "sa2va" / "metrics_sa2va.csv")
    semseg_metrics = load_csv_metrics(
        EVAL_DIR / "sam3_oracle_points" / "metrics_qwen3_semseg.csv")

    oracle_vis_dir = EVAL_DIR / "sam3_oracle_points" / "visualizations"
    sa2va_vis_dir = EVAL_DIR / "sa2va" / "visualizations"

    cell_size = 256
    sep = 4
    lbl_h = 28

    count = 0
    for entry in metadata:
        crop_name = entry["crop_name"]

        # Load original image
        image_path = entry.get("image_path", entry.get("image"))
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        # Resize image to cell_size x cell_size
        img = cv2.resize(image_bgr, (cell_size, cell_size),
                         interpolation=cv2.INTER_LINEAR)

        # Load GT masks and resize
        gt_a = np.array(Image.open(entry["mask_a_path"]).convert("L"))
        gt_b = np.array(Image.open(entry["mask_b_path"]).convert("L"))
        gt_a = cv2.resize((gt_a > 127).astype(np.float32),
                          (cell_size, cell_size),
                          interpolation=cv2.INTER_NEAREST)
        gt_b = cv2.resize((gt_b > 127).astype(np.float32),
                          (cell_size, cell_size),
                          interpolation=cv2.INTER_NEAREST)

        # Extract SemSeg prediction masks from _04_qwen.png
        qwen_vis_path = oracle_vis_dir / f"{crop_name}_04_qwen.png"
        sem_a, sem_b = extract_pred_masks_from_qwen_vis(
            qwen_vis_path, target_size=cell_size)
        if sem_a is None:
            sem_a = np.zeros((cell_size, cell_size), dtype=np.float32)
            sem_b = np.zeros_like(sem_a)

        # Extract Sa2VA prediction masks from _06_sa2va.png
        sa2va_vis_path = sa2va_vis_dir / f"{crop_name}_06_sa2va.png"
        sav_a, sav_b = extract_pred_masks_from_sa2va_vis(
            sa2va_vis_path, target_size=cell_size)
        if sav_a is None:
            sav_a = np.zeros((cell_size, cell_size), dtype=np.float32)
            sav_b = np.zeros_like(sav_a)

        # --- Build label bars ---
        def label_bar(text):
            bar = np.zeros((lbl_h, cell_size, 3), dtype=np.uint8) + 35
            cv2.putText(bar, text, (6, lbl_h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1,
                        cv2.LINE_AA)
            return bar

        semseg_iou = semseg_metrics.get(crop_name, {}).get("mean_iou", 0)
        semseg_ari = semseg_metrics.get(crop_name, {}).get("ari", 0)
        sa2va_iou = sa2va_metrics.get(crop_name, {}).get("mean_iou", 0)
        sa2va_ari = sa2va_metrics.get(crop_name, {}).get("ari", 0)

        separator = np.zeros((sep, cell_size, 3), dtype=np.uint8)

        # GT column
        gt_col = np.vstack([
            label_bar("Ground Truth"),
            mask_overlay(img, gt_a, gt_b),
            separator,
            binary_mask_image(gt_a, gt_b, cell_size, cell_size),
        ])

        # SemSeg column
        sem_col = np.vstack([
            label_bar(f"Qw3 SemSeg  IoU:{semseg_iou:.3f}"),
            mask_overlay(img, sem_a, sem_b),
            separator,
            binary_mask_image(sem_a, sem_b, cell_size, cell_size),
        ])

        # Sa2VA column
        sa_col = np.vstack([
            label_bar(f"Sa2VA  IoU:{sa2va_iou:.3f}"),
            mask_overlay(img, sav_a, sav_b),
            separator,
            binary_mask_image(sav_a, sav_b, cell_size, cell_size),
        ])

        # Combine: GT | SemSeg | Sa2VA
        col_sep = np.zeros((gt_col.shape[0], sep, 3), dtype=np.uint8)
        fig = np.hstack([gt_col, col_sep, sem_col, col_sep, sa_col])

        out_path = THUMB_DIR / f"{crop_name}_07_sa2va_comparison.jpg"
        cv2.imwrite(str(out_path), fig, [cv2.IMWRITE_JPEG_QUALITY, 92])
        count += 1

    print(f"  Generated {count} comparison figures in {THUMB_DIR}")


def update_results_json():
    """Add Sa2VA and AutoSAM per-sample metrics to results.json."""
    print("Updating results.json...")

    results_path = DATA_DIR / "results.json"
    data = json.load(open(results_path))

    sa2va_metrics = load_csv_metrics(EVAL_DIR / "sa2va" / "metrics_sa2va.csv")
    sa2va_summary = json.load(open(EVAL_DIR / "sa2va" / "summary.json"))
    sa2va_key = list(sa2va_summary.keys())[0]
    sa2va_sum = sa2va_summary[sa2va_key]

    autosam_metrics = load_csv_metrics(
        EVAL_DIR / "autosam" / "metrics_autosam.csv")
    autosam_summary = json.load(open(EVAL_DIR / "autosam" / "summary.json"))
    autosam_sum = autosam_summary.get("autosam", {})

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

    if "autosam" not in data["summary"] or not isinstance(data["summary"]["autosam"], dict):
        data["summary"]["autosam"] = {}
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
        if crop in sa2va_metrics:
            sample["metrics"]["sa2va"] = {
                "mean_iou": sa2va_metrics[crop]["mean_iou"]
            }
        if crop in autosam_metrics:
            sample["metrics"]["autosam"] = {
                "mean_iou": autosam_metrics[crop]["mean_iou"]
            }

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
