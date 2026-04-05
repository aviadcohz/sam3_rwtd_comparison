"""
Generate web assets for GitHub Pages from evaluation results.

Reads eval_results/sam3_oracle_points/ and eval_results/autosam/ and produces:
  - docs/data/results.json  (aggregated metrics for charts)
  - docs/assets/thumbnails/  (resized visualization PNGs)

Usage:
  python docs/generate_web_assets.py
"""

import json
import csv
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval_results" / "sam3_oracle_points"
AUTOSAM_EVAL_DIR = PROJECT_ROOT / "eval_results" / "autosam"
DOCS_DIR = PROJECT_ROOT / "docs"
THUMB_DIR = DOCS_DIR / "assets" / "thumbnails"
DATA_DIR = DOCS_DIR / "data"

THUMB_WIDTH = 800  # px


def load_csv_metrics(csv_path):
    """Load per-sample metrics from a CSV file."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k != "crop_name" else v
                         for k, v in row.items()})
    return rows


def generate_thumbnails():
    """Resize visualization PNGs for web display (separate per-row figures)."""
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    # Main evaluation visualizations: {crop}_{01_gt,02_baseline,03_oracle,04_qwen}.png
    vis_dir = EVAL_DIR / "visualizations"
    for img_path in sorted(vis_dir.glob("*_0[1-4]_*.png")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = THUMB_WIDTH / w
        new_h = int(h * scale)
        thumb = cv2.resize(img, (THUMB_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        out_path = THUMB_DIR / img_path.name
        cv2.imwrite(str(out_path), thumb, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        count += 1

    # AutoSAM visualizations: {crop}_05_autosam.png and {crop}_05_comparison.png
    autosam_vis_dir = AUTOSAM_EVAL_DIR / "visualizations"
    if autosam_vis_dir.exists():
        for img_path in sorted(autosam_vis_dir.glob("*_05_*.png")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = THUMB_WIDTH / w
            new_h = int(h * scale)
            thumb = cv2.resize(img, (THUMB_WIDTH, new_h),
                               interpolation=cv2.INTER_AREA)
            out_path = THUMB_DIR / img_path.name
            cv2.imwrite(str(out_path), thumb, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            count += 1

    print(f"Generated {count} thumbnails in {THUMB_DIR}")


def generate_results_json():
    """Build results.json from summary and per-sample CSVs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load summary
    summary_path = EVAL_DIR / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    # Merge AutoSAM summary if available
    autosam_summary_path = AUTOSAM_EVAL_DIR / "summary.json"
    if autosam_summary_path.exists():
        with open(autosam_summary_path) as f:
            autosam_summary = json.load(f)
        summary.update(autosam_summary)

    # Load per-sample metrics for each approach
    approach_csvs = {
        "generic": ("metrics_generic.csv", EVAL_DIR),
        "oracle_text": ("metrics_oracle_text.csv", EVAL_DIR),
        "points_only": ("metrics_points_only.csv", EVAL_DIR),
        "oracle_text_points": ("metrics_oracle_text_points.csv", EVAL_DIR),
        "qwen3_text_proposal": ("metrics_qwen3_text.csv", EVAL_DIR),
        "qwen3_text_semseg": ("metrics_qwen3_semseg.csv", EVAL_DIR),
        "qwen3_clipseg": ("metrics_qwen3_clipseg.csv", EVAL_DIR),
        "autosam": ("metrics_autosam.csv", AUTOSAM_EVAL_DIR),
    }

    per_sample = {}
    for key, (csv_name, eval_dir) in approach_csvs.items():
        csv_path = eval_dir / csv_name
        if csv_path.exists():
            per_sample[key] = load_csv_metrics(csv_path)
        else:
            # Try alternate name for qwen3
            alt = csv_name.replace("metrics_qwen3_text.", "metrics_qwen3_text_proposal.")
            alt_path = eval_dir / alt
            if alt_path.exists():
                per_sample[key] = load_csv_metrics(alt_path)

    # Load Qwen3 outputs for descriptions
    qwen_path = EVAL_DIR / "qwen3_outputs.json"
    descriptions = {}
    if qwen_path.exists():
        with open(qwen_path) as f:
            qwen_outputs = json.load(f)
        for entry in qwen_outputs:
            cn = entry.get("crop_name", "")
            descriptions[cn] = {
                "desc_a": entry.get("parsed", {}).get("desc_a", ""),
                "desc_b": entry.get("parsed", {}).get("desc_b", ""),
            }

    # Build per-sample combined data
    samples = []
    # Use generic as the base list
    if "generic" in per_sample:
        for row in per_sample["generic"]:
            cn = row["crop_name"]
            sample = {
                "crop_name": cn,
                "descriptions": descriptions.get(cn, {}),
                "metrics": {},
            }
            for approach, rows in per_sample.items():
                match = next((r for r in rows if r["crop_name"] == cn), None)
                if match:
                    sample["metrics"][approach] = {
                        "mean_iou": round(match.get("mean_iou", 0), 4),
                        "mean_dice": round(match.get("mean_dice", 0), 4),
                        "ari": round(match.get("ari", 0), 4),
                    }
            samples.append(sample)

    results = {
        "summary": summary,
        "samples": samples,
    }

    out_path = DATA_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path} ({len(samples)} samples)")


if __name__ == "__main__":
    generate_results_json()
    generate_thumbnails()
