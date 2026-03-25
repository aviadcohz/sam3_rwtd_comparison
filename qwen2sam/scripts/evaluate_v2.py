"""
Evaluation script for Qwen2SAM v2 (SAM3 DETR-based).

Compares zero-shot (untrained) vs trained model, producing:
  - Per-sample CSV with IoU, Dice, ARI (for both zero-shot and trained)
  - summary.json with aggregated metrics + improvement stats
  - Per-sample visualization grids: Image | GT | Zero-shot | Trained
  - training_curves.png (copied from checkpoint dir)

Usage:
  conda activate texture_boundary
  python -m qwen2sam.scripts.evaluate_v2 \
      --config qwen2sam/configs/v2.yaml \
      --checkpoint checkpoints/v2/best.pt \
      --split test --output_dir eval_results/v2_test
"""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from qwen2sam.models.qwen2sam_v2 import Qwen2SAMv2
from qwen2sam.data.dataset_v2 import V2Dataset, V2Collator
from qwen2sam.training.train_phase1 import load_config, set_seed


# ===================================================================== #
#  Metrics                                                                #
# ===================================================================== #

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0.5
    gt_b = gt > 0.5
    intersection = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0.5
    gt_b = gt > 0.5
    intersection = (pred_b & gt_b).sum()
    total = pred_b.sum() + gt_b.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / total)


def compute_ari(pred_a: np.ndarray, pred_b: np.ndarray,
                gt_a: np.ndarray, gt_b: np.ndarray) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return float("nan")
    pred_labels = np.zeros(pred_a.shape, dtype=np.int32)
    pred_labels[pred_a > 0.5] = 1
    pred_labels[pred_b > 0.5] = 2
    gt_labels = np.zeros(gt_a.shape, dtype=np.int32)
    gt_labels[gt_a > 0.5] = 1
    gt_labels[gt_b > 0.5] = 2
    return float(adjusted_rand_score(gt_labels.ravel(), pred_labels.ravel()))


def compute_sample_metrics(pred_a, pred_b, gt_a, gt_b, crop_name):
    # Hungarian matching: try both A/B assignments, pick the better one.
    # Fixes ARI collapse caused by consistent A↔B label swap after training.
    iou_direct = (compute_iou(pred_a, gt_a) + compute_iou(pred_b, gt_b)) / 2.0
    iou_swapped = (compute_iou(pred_a, gt_b) + compute_iou(pred_b, gt_a)) / 2.0
    if iou_swapped > iou_direct:
        pred_a, pred_b = pred_b, pred_a  # swap to best assignment

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


def save_metrics_csv(all_metrics, path):
    fieldnames = ["crop_name", "iou_a", "iou_b", "mean_iou",
                   "dice_a", "dice_b", "mean_dice", "ari"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: m[k] for k in fieldnames})


# ===================================================================== #
#  Visualization helpers                                                  #
# ===================================================================== #

COLOR_A = (0, 0, 220)       # red in BGR
COLOR_B = (220, 80, 0)      # blue in BGR
COLOR_BOUNDARY = (0, 255, 255)  # yellow


def mask_overlay(image: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray,
                 alpha: float = 0.45) -> np.ndarray:
    vis = image.copy()
    overlay = image.copy()
    overlay[mask_a > 0.5] = COLOR_A
    overlay[mask_b > 0.5] = COLOR_B
    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)


def binary_mask_image(mask_a: np.ndarray, mask_b: np.ndarray,
                      h: int, w: int) -> np.ndarray:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[mask_a > 0.5] = COLOR_A
    canvas[mask_b > 0.5] = COLOR_B
    return canvas


def boundary_image(mask_a: np.ndarray, mask_b: np.ndarray,
                   h: int, w: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ma = (mask_a > 0.5).astype(np.uint8) * 255
    mb = (mask_b > 0.5).astype(np.uint8) * 255
    bd_a = ma - cv2.erode(ma, kernel, iterations=1)
    bd_b = mb - cv2.erode(mb, kernel, iterations=1)
    da = cv2.dilate(bd_a, kernel, iterations=2)
    db = cv2.dilate(bd_b, kernel, iterations=2)
    interface = ((da > 0) & (db > 0)).astype(np.uint8) * 255
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[bd_a > 0] = (0, 0, 150)
    canvas[bd_b > 0] = (150, 60, 0)
    canvas[interface > 0] = COLOR_BOUNDARY
    return canvas


def create_grid_figure(
    image_bgr: np.ndarray,
    gt_a: np.ndarray, gt_b: np.ndarray,
    zs_a: np.ndarray, zs_b: np.ndarray,
    pred_a: np.ndarray, pred_b: np.ndarray,
    metrics_trained: dict,
    metrics_zs: dict,
    title: str = "",
    cell_size: int = 256,
) -> np.ndarray:
    """
    Create a 3-row x 4-col grid for one sample.

    Columns: Image | Ground Truth | Zero-shot | Trained
    Row 1: Mask overlay (red=A, blue=B on image)
    Row 2: Binary masks (colored, no background)
    Row 3: Boundary visualization
    """
    h, w = image_bgr.shape[:2]
    scale = cell_size / max(h, w)
    ch, cw = int(h * scale), int(w * scale)

    def ri(img):
        return cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)

    def rm(mask):
        return cv2.resize(mask.astype(np.float32), (cw, ch),
                          interpolation=cv2.INTER_NEAREST)

    def make_col(ma, mb):
        return (
            mask_overlay(img, ma, mb),
            binary_mask_image(ma, mb, ch, cw),
            boundary_image(ma, mb, ch, cw),
        )

    img = ri(image_bgr)
    ga, gb = rm(gt_a), rm(gt_b)
    za, zb = rm(zs_a), rm(zs_b)
    pa, pb = rm(pred_a), rm(pred_b)

    # Col 0: Original image
    col_image = (img.copy(),
                 np.zeros((ch, cw, 3), dtype=np.uint8),
                 np.zeros((ch, cw, 3), dtype=np.uint8))
    col_gt = make_col(ga, gb)
    col_zs = make_col(za, zb)
    col_pred = make_col(pa, pb)

    cols = [col_image, col_gt, col_zs, col_pred]
    col_labels = [
        title,
        "Ground Truth",
        f"Zero-shot mIoU={metrics_zs.get('mean_iou', 0):.3f}",
        f"Trained mIoU={metrics_trained.get('mean_iou', 0):.3f}",
    ]
    row_labels = ["Overlay", "Masks", "Boundary"]

    # --- Assemble grid ---
    sep = 2
    header_h = 36
    row_label_w = 70

    bar_w = row_label_w + len(cols) * (cw + sep)
    bar = np.zeros((header_h, bar_w, 3), dtype=np.uint8) + 30
    x = row_label_w
    for lbl in col_labels:
        cv2.putText(bar, lbl, (x + 4, header_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        x += cw + sep

    def make_row(row_idx, row_label):
        rl = np.zeros((ch, row_label_w, 3), dtype=np.uint8) + 20
        cv2.putText(rl, row_label, (4, ch // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        sep_col = np.ones((ch, sep, 3), dtype=np.uint8) * 80
        row = rl
        for col in cols:
            row = np.concatenate([row, sep_col, col[row_idx]], axis=1)
        return row

    sep_row = np.ones((sep, bar.shape[1], 3), dtype=np.uint8) * 80
    rows = [bar]
    for ri_idx, rl in enumerate(row_labels):
        rows.append(sep_row)
        rows.append(make_row(ri_idx, rl))

    return np.concatenate(rows, axis=0)


# ===================================================================== #
#  Prediction                                                             #
# ===================================================================== #

@torch.no_grad()
def predict_single(model, batch, device):
    """Run model on a single batch, return binarized masks."""
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)

    qwen_inputs = {
        k: batch[k] for k in [
            "input_ids", "attention_mask", "pixel_values",
            "image_grid_thw", "labels",
        ]
        if k in batch and isinstance(batch.get(k), torch.Tensor)
    }

    seg_a_pos = batch["seg_a_positions"]
    seg_b_pos = batch["seg_b_positions"]
    valid = (seg_a_pos >= 0) & (seg_b_pos >= 0)
    if not valid.any():
        return None, None

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            qwen_inputs, batch["sam_images"],
            seg_a_pos, seg_b_pos,
        )

    # Select best query by confidence
    scores_a = outputs["pred_logits_a"][0].squeeze(-1).sigmoid()
    scores_b = outputs["pred_logits_b"][0].squeeze(-1).sigmoid()
    best_a = scores_a.argmax().item()
    best_b = scores_b.argmax().item()

    gt_h, gt_w = batch["masks_a"].shape[-2:]

    # Use hires path if available, otherwise fallback to bilinear
    if "hires_pixel_a" in outputs and outputs["hires_pixel_a"] is not None:
        # High-res mask via dot product
        hq_a = outputs["hires_queries_a"][0, best_a].float()   # (thin_dim,)
        hp_a = outputs["hires_pixel_a"]
        if hp_a.ndim == 4:
            hp_a = hp_a[0]                              # (thin_dim, H, W)
        pred_mask_a = torch.einsum("c,chw->hw", hq_a, hp_a.float())

        hq_b = outputs["hires_queries_b"][0, best_b].float()
        hp_b = outputs["hires_pixel_b"]
        if hp_b.ndim == 4:
            hp_b = hp_b[0]
        pred_mask_b = torch.einsum("c,chw->hw", hq_b, hp_b.float())

        # Resize to GT if different from target_size
        if pred_mask_a.shape != (gt_h, gt_w):
            pred_mask_a = F.interpolate(
                pred_mask_a[None, None].float(), size=(gt_h, gt_w),
                mode="bilinear", align_corners=False,
            ).squeeze()
            pred_mask_b = F.interpolate(
                pred_mask_b[None, None].float(), size=(gt_h, gt_w),
                mode="bilinear", align_corners=False,
            ).squeeze()
    else:
        # Fallback: bilinear upsample from 288
        pred_mask_a = outputs["pred_masks_a"][0, best_a]
        pred_mask_b = outputs["pred_masks_b"][0, best_b]
        if pred_mask_a.shape != (gt_h, gt_w):
            pred_mask_a = F.interpolate(
                pred_mask_a[None, None].float(), size=(gt_h, gt_w),
                mode="bilinear", align_corners=False,
            ).squeeze()
            pred_mask_b = F.interpolate(
                pred_mask_b[None, None].float(), size=(gt_h, gt_w),
                mode="bilinear", align_corners=False,
            ).squeeze()

    pred_a = (pred_mask_a.sigmoid().cpu().numpy() > 0.5).astype(np.float32)
    pred_b = (pred_mask_b.sigmoid().cpu().numpy() > 0.5).astype(np.float32)
    return pred_a, pred_b


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2SAM v2")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results/v2")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--cell_size", type=int, default=256,
                        help="Cell size for visualization grid")
    parser.add_argument("--no_vis", action="store_true",
                        help="Skip generating visualization images")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Dataset ----------------------------------------------------- #
    data_root = args.data_root or cfg["data"]["data_root"]
    dataset = V2Dataset(
        data_root=data_root,
        metadata_file=cfg["data"].get("metadata_file", "metadata.json"),
        image_size=cfg["model"].get("image_size", 1008),
    )

    train_n = cfg["data"].get("train_size", 10)
    val_n = cfg["data"].get("val_size", train_n)
    if args.split == "train":
        eval_indices = list(range(train_n))
    elif args.split == "val":
        eval_indices = list(range(train_n, train_n + val_n))
    elif args.split == "test":
        eval_indices = list(range(train_n + val_n, len(dataset)))
    else:
        eval_indices = list(range(len(dataset)))
    print(f"Split '{args.split}': {len(eval_indices)} samples")

    # ---- Output dirs ------------------------------------------------- #
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================= #
    #  Pass 1: Zero-shot (fresh model, no checkpoint)                     #
    # ================================================================= #
    print("\n--- Pass 1: Zero-shot (untrained model) ---")
    print("Building zero-shot model...")
    zs_model = Qwen2SAMv2(cfg, device=str(device))
    zs_model.qwen.eval()
    zs_model.projector.eval()
    zs_model.sam3.eval()

    collator = V2Collator(
        processor=zs_model.processor,
        system_prompt=cfg["data"].get("system_prompt", ""),
        user_prompt=cfg["data"].get("user_prompt", ""),
        seg_a_id=zs_model.seg_a_id,
        seg_b_id=zs_model.seg_b_id,
        inference=True,  # generic template, no GT texture names
    )

    t0 = time.time()
    zs_metrics_all = []
    zs_preds = {}  # idx -> (pred_a, pred_b)
    for i, idx in enumerate(eval_indices):
        raw_sample = dataset[idx]
        crop_name = f"sample_{idx}"
        gt_a_np = raw_sample["mask_a"].numpy()
        gt_b_np = raw_sample["mask_b"].numpy()

        single_batch = collator([raw_sample])
        pred_a, pred_b = predict_single(zs_model, single_batch, device)
        if pred_a is None:
            zs_preds[idx] = (None, None)
            continue

        metrics = compute_sample_metrics(pred_a, pred_b, gt_a_np, gt_b_np, crop_name)
        zs_metrics_all.append(metrics)
        zs_preds[idx] = (pred_a, pred_b)

        if (i + 1) % 20 == 0 or (i + 1) == len(eval_indices):
            running_iou = np.mean([m["mean_iou"] for m in zs_metrics_all])
            print(f"  ZS: {i+1}/{len(eval_indices)} | running mIoU: {running_iou:.4f}")

    zs_elapsed = time.time() - t0
    zs_summary = aggregate_metrics(zs_metrics_all, "zero_shot")
    print(f"  Zero-shot done ({zs_elapsed:.1f}s): mIoU={zs_summary['mean_iou']:.4f}")

    # Free zero-shot model
    del zs_model
    torch.cuda.empty_cache()

    # ================================================================= #
    #  Pass 2: Trained model                                              #
    # ================================================================= #
    print("\n--- Pass 2: Trained model ---")
    print("Building trained model...")
    model = Qwen2SAMv2(cfg, device=str(device))

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.projector.load_state_dict(ckpt["projector_state_dict"])
    if "sam3_trainable_state_dict" in ckpt:
        model.sam3.load_state_dict(ckpt["sam3_trainable_state_dict"], strict=False)
    if "qwen_lora_state_dict" in ckpt:
        model.qwen.load_state_dict(ckpt["qwen_lora_state_dict"], strict=False)
    if "hires_head_state_dict" in ckpt and hasattr(model, "hires_head") and model.hires_head is not None:
        model.hires_head.load_state_dict(ckpt["hires_head_state_dict"])
        print("  HiRes head loaded")
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}")

    model.qwen.eval()
    model.projector.eval()
    model.sam3.eval()

    t1 = time.time()
    trained_metrics_all = []
    for i, idx in enumerate(eval_indices):
        raw_sample = dataset[idx]
        crop_name = f"sample_{idx}"
        gt_a_np = raw_sample["mask_a"].numpy()
        gt_b_np = raw_sample["mask_b"].numpy()
        image_bgr = cv2.cvtColor(np.array(raw_sample["image"]), cv2.COLOR_RGB2BGR)

        single_batch = collator([raw_sample])
        pred_a, pred_b = predict_single(model, single_batch, device)
        if pred_a is None:
            continue

        t_metrics = compute_sample_metrics(pred_a, pred_b, gt_a_np, gt_b_np, crop_name)
        trained_metrics_all.append(t_metrics)

        # Generate visualization (zero-shot vs trained)
        if not args.no_vis:
            zs_a, zs_b = zs_preds.get(idx, (None, None))
            if zs_a is not None:
                zs_met = next((m for m in zs_metrics_all if m["crop_name"] == crop_name), {})
                grid = create_grid_figure(
                    image_bgr, gt_a_np, gt_b_np,
                    zs_a, zs_b, pred_a, pred_b,
                    t_metrics, zs_met,
                    title=crop_name, cell_size=args.cell_size,
                )
                cv2.imwrite(str(vis_dir / f"{crop_name}_eval.png"), grid)

        if (i + 1) % 20 == 0 or (i + 1) == len(eval_indices):
            running_iou = np.mean([m["mean_iou"] for m in trained_metrics_all])
            print(f"  Trained: {i+1}/{len(eval_indices)} | running mIoU: {running_iou:.4f}")

    trained_elapsed = time.time() - t1
    trained_summary = aggregate_metrics(trained_metrics_all, "trained")
    print(f"  Trained done ({trained_elapsed:.1f}s): mIoU={trained_summary['mean_iou']:.4f}")

    # ---- Save CSVs --------------------------------------------------- #
    save_metrics_csv(zs_metrics_all, output_dir / "metrics_zero_shot.csv")
    save_metrics_csv(trained_metrics_all, output_dir / f"metrics_trained.csv")

    # ---- Compute improvement ----------------------------------------- #
    improvement = {
        "mean_iou": trained_summary["mean_iou"] - zs_summary["mean_iou"],
        "mean_dice": trained_summary["mean_dice"] - zs_summary["mean_dice"],
        "mean_ari": trained_summary["mean_ari"] - zs_summary["mean_ari"],
    }
    improvement_pct = {}
    for k in improvement:
        base = zs_summary[k]
        improvement_pct[k] = round(100 * improvement[k] / base, 2) if abs(base) > 1e-8 else 0.0

    summary = {
        "trained": trained_summary,
        "zero_shot": zs_summary,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Copy training_curves.png
    ckpt_dir = Path(args.checkpoint).parent
    curves_src = ckpt_dir / "training_curves.png"
    if curves_src.exists():
        shutil.copy2(str(curves_src), str(output_dir / "training_curves.png"))
        print(f"Copied training_curves.png from {ckpt_dir}")

    # ---- Analysis ---------------------------------------------------- #
    sorted_by_iou = sorted(trained_metrics_all, key=lambda m: m["mean_iou"])
    n_show = min(5, len(sorted_by_iou))
    worst = sorted_by_iou[:n_show]
    best = sorted_by_iou[-n_show:][::-1]
    n_good = sum(1 for m in trained_metrics_all if m["mean_iou"] >= 0.7)
    n_medium = sum(1 for m in trained_metrics_all if 0.4 <= m["mean_iou"] < 0.7)
    n_bad = sum(1 for m in trained_metrics_all if m["mean_iou"] < 0.4)

    # ---- Print summary ------------------------------------------------ #
    total_elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Qwen2SAM v2 Evaluation — {args.split} set ({total_elapsed:.1f}s)")
    print(f"{'='*65}")
    print(f"  {'':20s} {'Zero-shot':>12s}  {'Trained':>12s}  {'Improvement':>12s}")
    print(f"  {'─'*58}")
    print(f"  {'Mean IoU':20s} {zs_summary['mean_iou']:12.4f}  {trained_summary['mean_iou']:12.4f}  {improvement['mean_iou']:+12.4f} ({improvement_pct['mean_iou']:+.1f}%)")
    print(f"  {'Mean IoU (A)':20s} {zs_summary['mean_iou_a']:12.4f}  {trained_summary['mean_iou_a']:12.4f}")
    print(f"  {'Mean IoU (B)':20s} {zs_summary['mean_iou_b']:12.4f}  {trained_summary['mean_iou_b']:12.4f}")
    print(f"  {'Mean Dice':20s} {zs_summary['mean_dice']:12.4f}  {trained_summary['mean_dice']:12.4f}  {improvement['mean_dice']:+12.4f} ({improvement_pct['mean_dice']:+.1f}%)")
    print(f"  {'Mean ARI':20s} {zs_summary['mean_ari']:12.4f}  {trained_summary['mean_ari']:12.4f}  {improvement['mean_ari']:+12.4f} ({improvement_pct['mean_ari']:+.1f}%)")
    print(f"  {'Samples':20s} {zs_summary['num_samples']:12d}  {trained_summary['num_samples']:12d}")
    print(f"{'='*65}")
    print(f"\n  Quality distribution (trained):")
    print(f"    Good  (IoU >= 0.7): {n_good:3d} ({100*n_good/max(len(trained_metrics_all),1):.1f}%)")
    print(f"    Medium (0.4-0.7):  {n_medium:3d} ({100*n_medium/max(len(trained_metrics_all),1):.1f}%)")
    print(f"    Bad   (IoU < 0.4): {n_bad:3d} ({100*n_bad/max(len(trained_metrics_all),1):.1f}%)")
    print(f"\n  Top {n_show} best samples:")
    for m in best:
        print(f"    {m['crop_name']:>15s}  mIoU={m['mean_iou']:.4f}  dice={m['mean_dice']:.4f}  ari={m['ari']:.4f}")
    print(f"\n  Top {n_show} worst samples:")
    for m in worst:
        print(f"    {m['crop_name']:>15s}  mIoU={m['mean_iou']:.4f}  dice={m['mean_dice']:.4f}  ari={m['ari']:.4f}")
    print(f"\n{'='*65}")
    print(f"  Output: {output_dir}/")
    print(f"    summary.json             — zero-shot vs trained comparison")
    print(f"    metrics_zero_shot.csv    — per-sample zero-shot metrics")
    print(f"    metrics_trained.csv      — per-sample trained metrics")
    if not args.no_vis:
        print(f"    visualizations/          — {len(trained_metrics_all)} sample grids")
    if (output_dir / "training_curves.png").exists():
        print(f"    training_curves.png      — loss & IoU curves")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
