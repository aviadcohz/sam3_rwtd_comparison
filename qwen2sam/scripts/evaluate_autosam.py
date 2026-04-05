"""
AutoSAM Training + Evaluation on RWTD.

Trains AutoSAM's ModelEmb (HarDNet backbone) on RWTD training split,
then evaluates on all 253 samples. Produces:
  - metrics_autosam.csv  (per-sample IoU, Dice, ARI)
  - summary.json         (aggregated metrics)
  - visualizations/      (per-sample overlay + mask PNGs)
  - comparison figures    (GT | Generic ZS | Qw3 SemSeg | AutoSAM)

AutoSAM uses only the image — no text or point prompts. SAM is frozen;
only the small ModelEmb (~2M params) is trained via BCE + Dice loss.

Usage:
  python -m qwen2sam.scripts.evaluate_autosam \
      --data_root /home/aviad/RWTD \
      --output_dir eval_results/autosam

  # Skip training, load existing checkpoint:
  python -m qwen2sam.scripts.evaluate_autosam \
      --data_root /home/aviad/RWTD \
      --output_dir eval_results/autosam \
      --no_train --checkpoint eval_results/autosam/best_model.pth
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Add project root + AutoSAM to path so this script works when run directly
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT.parent / "AutoSAM"))

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from qwen2sam.models.autosam_model import ModelEmb, norm_batch, sam_call, dice_loss
from qwen2sam.scripts.evaluate_v2 import (
    compute_sample_metrics, aggregate_metrics, save_metrics_csv,
    mask_overlay, binary_mask_image,
)


# ===================================================================== #
#  Dataset                                                                 #
# ===================================================================== #

class RWTDAutoSAMDataset(Dataset):
    """RWTD dataset for AutoSAM: returns SAM-ready images + GT masks."""

    def __init__(self, metadata, indices, sam_transform, Idim=256):
        self.samples = [metadata[i] for i in indices]
        self.sam_transform = sam_transform
        self.Idim = Idim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img_path = entry["image_path"]
        mask_a_path = entry["mask_a_path"]
        mask_b_path = entry["mask_b_path"]
        crop_name = entry["crop_name"]

        # Load image (RGB)
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]

        # Apply SAM transform (resize longest side to 1024)
        transformed = self.sam_transform.apply_image(image)
        img_tensor = torch.as_tensor(transformed).permute(2, 0, 1).float()
        img_sz = torch.tensor(img_tensor.shape[1:3])
        original_sz = torch.tensor([h, w])

        # Load GT masks
        mask_a = np.array(Image.open(mask_a_path).convert("L")).astype(
            np.float32) / 255.0
        mask_b = np.array(Image.open(mask_b_path).convert("L")).astype(
            np.float32) / 255.0

        gt_a = torch.from_numpy(mask_a)
        gt_b = torch.from_numpy(mask_b)

        return {
            "image": img_tensor,
            "gt_a": gt_a,
            "gt_b": gt_b,
            "original_sz": original_sz,
            "img_sz": img_sz,
            "crop_name": crop_name,
        }


def collate_fn(batch):
    """Custom collate: pad images to same size for batching."""
    max_h = max(b["image"].shape[1] for b in batch)
    max_w = max(b["image"].shape[2] for b in batch)

    images = []
    for b in batch:
        img = b["image"]
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        img = F.pad(img, (0, pad_w, 0, pad_h))
        images.append(img)

    return {
        "image": torch.stack(images),
        "gt_a": torch.stack([b["gt_a"] for b in batch]),
        "gt_b": torch.stack([b["gt_b"] for b in batch]),
        "original_sz": torch.stack([b["original_sz"] for b in batch]),
        "img_sz": torch.stack([b["img_sz"] for b in batch]),
        "crop_name": [b["crop_name"] for b in batch],
    }


# ===================================================================== #
#  Training                                                                #
# ===================================================================== #

def train_autosam(model_emb, sam, train_dataset, val_dataset, args, device):
    """Train ModelEmb on RWTD training split."""
    Idim = args.idim
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn) if val_dataset else None

    optimizer = torch.optim.Adam(
        model_emb.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_iou = -1.0
    patience_counter = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining AutoSAM ModelEmb on {len(train_dataset)} samples "
          f"(val: {len(val_dataset) if val_dataset else 0})")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, "
          f"Batch: {args.batch_size}, Idim: {Idim}")

    for epoch in range(1, args.epochs + 1):
        model_emb.train()
        epoch_losses = []

        for batch in train_loader:
            imgs = batch["image"].to(device)
            gt_a = batch["gt_a"].to(device)  # (B, H, W)
            original_sz = batch["original_sz"]
            img_sz = batch["img_sz"]

            # Small image for ModelEmb
            imgs_small = F.interpolate(
                imgs, (Idim, Idim), mode='bilinear', align_corners=True)

            # Forward: ModelEmb → SAM
            dense_embeddings = model_emb(imgs_small)

            batched_input = []
            for i in range(imgs.shape[0]):
                batched_input.append({"image": imgs[i]})

            # SAM is frozen (requires_grad=False) but we need gradients
            # to flow back through dense_embeddings to ModelEmb
            low_res_masks = sam_call(batched_input, sam, dense_embeddings)

            # Resize GT to match mask output
            mask_size = low_res_masks.shape[2:]
            gt_resized = F.interpolate(
                gt_a.unsqueeze(1).float(), mask_size,
                mode='nearest')  # (B, 1, H', W')

            # Try both assignments (mask_a vs 1-mask_a), pick best
            loss_a = criterion(low_res_masks, gt_resized) + \
                dice_loss(low_res_masks.sigmoid(), gt_resized)
            loss_b = criterion(low_res_masks, 1 - gt_resized) + \
                dice_loss(low_res_masks.sigmoid(), 1 - gt_resized)

            # Per-sample best assignment
            loss = torch.min(loss_a, loss_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = np.mean(epoch_losses)

        # Validation
        val_iou = -1.0
        if val_loader and (epoch % 5 == 0 or epoch <= 5):
            val_iou = evaluate_iou(model_emb, sam, val_loader, Idim, device)

        if epoch % 10 == 0 or epoch <= 5 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")

        # Save best
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model_emb.state_dict(),
                       str(output_dir / "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience and epoch >= 20:
            print(f"  Early stopping at epoch {epoch} "
                  f"(best val IoU: {best_val_iou:.4f})")
            break

    # Load best checkpoint
    best_path = output_dir / "best_model.pth"
    if best_path.exists():
        model_emb.load_state_dict(torch.load(str(best_path),
                                             map_location=device))
        print(f"  Loaded best checkpoint (val IoU: {best_val_iou:.4f})")

    return model_emb


@torch.no_grad()
def evaluate_iou(model_emb, sam, loader, Idim, device):
    """Quick IoU evaluation on a dataset."""
    model_emb.eval()
    ious = []

    for batch in loader:
        imgs = batch["image"].to(device)
        gt_a = batch["gt_a"].numpy()
        gt_b = batch["gt_b"].numpy()

        imgs_small = F.interpolate(
            imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        dense_embeddings = model_emb(imgs_small)
        batched_input = [{"image": imgs[i]} for i in range(imgs.shape[0])]
        low_res_masks = norm_batch(sam_call(batched_input, sam,
                                           dense_embeddings))

        for i in range(imgs.shape[0]):
            mask = low_res_masks[i, 0].cpu().numpy()
            # Resize to GT size
            h, w = gt_a[i].shape
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            pred_a = (mask > 0.5).astype(np.float32)
            pred_b = 1.0 - pred_a
            met = compute_sample_metrics(
                pred_a, pred_b, gt_a[i], gt_b[i], "")
            ious.append(met["mean_iou"])

    model_emb.train()
    return np.mean(ious) if ious else 0.0


# ===================================================================== #
#  Evaluation                                                              #
# ===================================================================== #

@torch.no_grad()
def evaluate_autosam(model_emb, sam, dataset, args, device):
    """Full evaluation on all samples with metrics + visualizations."""
    model_emb.eval()
    Idim = args.idim
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn)

    all_metrics = []
    t0 = time.time()

    for i, batch in enumerate(loader):
        imgs = batch["image"].to(device)
        gt_a = batch["gt_a"][0].numpy()
        gt_b = batch["gt_b"][0].numpy()
        crop_name = batch["crop_name"][0]
        original_sz = batch["original_sz"]
        img_sz = batch["img_sz"]

        # Forward
        imgs_small = F.interpolate(
            imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        dense_embeddings = model_emb(imgs_small)
        batched_input = [{"image": imgs[0]}]
        low_res_masks = norm_batch(sam_call(batched_input, sam,
                                           dense_embeddings))

        # Resize to GT size
        h, w = gt_a.shape
        mask = low_res_masks[0, 0].cpu().numpy()
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        pred_a = (mask > 0.5).astype(np.float32)
        pred_b = 1.0 - pred_a

        # Metrics (Hungarian matching inside)
        met = compute_sample_metrics(pred_a, pred_b, gt_a, gt_b, crop_name)
        all_metrics.append(met)

        # Visualization
        if not args.no_vis:
            # Load original image for visualization
            entry = dataset.samples[i]
            image_bgr = cv2.imread(entry["image_path"])
            _save_autosam_vis(
                image_bgr, gt_a, gt_b, pred_a, pred_b, met,
                crop_name, vis_dir, args.cell_size)

        if (i + 1) % 50 == 0 or (i + 1) == len(dataset):
            avg_iou = np.mean([m["mean_iou"] for m in all_metrics])
            print(f"  {i+1}/{len(dataset)} | AutoSAM mIoU: {avg_iou:.4f}")

    elapsed = time.time() - t0

    # Save metrics
    save_metrics_csv(all_metrics, output_dir / "metrics_autosam.csv")

    autosam_summary = aggregate_metrics(all_metrics, "autosam")
    summary = {"autosam": autosam_summary}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print(f"\n{'='*70}")
    print(f"  AutoSAM Evaluation — {len(dataset)} samples ({elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"  Mean IoU:  {autosam_summary['mean_iou']:.4f}")
    print(f"  Mean Dice: {autosam_summary['mean_dice']:.4f}")
    print(f"  Mean ARI:  {autosam_summary['mean_ari']:.4f}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*70}")

    return all_metrics


def _save_autosam_vis(image_bgr, gt_a, gt_b, pred_a, pred_b, met,
                      crop_name, vis_dir, cell_size=320):
    """Save per-sample AutoSAM visualization."""
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

    img = ri(image_bgr)
    ga, gb = rm(gt_a), rm(gt_b)
    pa, pb = rm(pred_a), rm(pred_b)

    def label_bar(text, width):
        bar = np.zeros((lbl_h, width, 3), dtype=np.uint8) + 35
        cv2.putText(bar, text, (8, lbl_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1,
                    cv2.LINE_AA)
        return bar

    iou = met.get("mean_iou", 0)
    ari = met.get("ari", met.get("mean_ari", 0))

    # GT column
    gt_overlay = mask_overlay(img, ga, gb)
    gt_masks = binary_mask_image(ga, gb, ch, cw)
    gt_lbl = label_bar("Ground Truth", cw)
    gt_col = np.vstack([gt_lbl, gt_overlay,
                        np.zeros((sep, cw, 3), dtype=np.uint8),
                        gt_masks])

    # AutoSAM column
    as_overlay = mask_overlay(img, pa, pb)
    as_masks = binary_mask_image(pa, pb, ch, cw)
    as_lbl = label_bar(
        f"AutoSAM   mIoU: {iou:.3f}   ARI: {ari:.3f}", cw)
    as_col = np.vstack([as_lbl, as_overlay,
                        np.zeros((sep, cw, 3), dtype=np.uint8),
                        as_masks])

    # Side by side
    spacer = np.zeros((gt_col.shape[0], sep * 2, 3), dtype=np.uint8)
    fig = np.hstack([gt_col, spacer, as_col])

    cv2.imwrite(str(vis_dir / f"{crop_name}_05_autosam.png"), fig)


# ===================================================================== #
#  Comparison figure generation                                            #
# ===================================================================== #

def generate_comparison_figures(dataset, args):
    """
    Generate 4-column comparison: GT | Generic ZS | Qw3 SemSeg | AutoSAM.

    Loads existing visualization PNGs from sam3_oracle_points and stitches
    them together with AutoSAM visualizations.
    """
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    base_vis_dir = Path(args.base_eval_dir) / "visualizations"

    # Load metrics for labels
    autosam_metrics = _load_csv_as_dict(output_dir / "metrics_autosam.csv")

    print(f"\nGenerating comparison figures for {len(dataset)} samples...")
    count = 0

    for i in range(len(dataset)):
        entry = dataset.samples[i]
        crop_name = entry["crop_name"]

        # Load existing visualization PNGs
        gt_path = base_vis_dir / f"{crop_name}_01_gt.png"
        baseline_path = base_vis_dir / f"{crop_name}_02_baseline.png"
        qwen_path = base_vis_dir / f"{crop_name}_04_qwen.png"
        autosam_path = vis_dir / f"{crop_name}_05_autosam.png"

        if not all(p.exists() for p in [gt_path, baseline_path, qwen_path, autosam_path]):
            continue

        gt_img = cv2.imread(str(gt_path))
        baseline_img = cv2.imread(str(baseline_path))
        qwen_img = cv2.imread(str(qwen_path))
        autosam_img = cv2.imread(str(autosam_path))

        if any(x is None for x in [gt_img, baseline_img, qwen_img, autosam_img]):
            continue

        # Resize all to same height
        target_h = gt_img.shape[0]

        def resize_to_h(img, h):
            scale = h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)

        baseline_img = resize_to_h(baseline_img, target_h)
        qwen_img = resize_to_h(qwen_img, target_h)
        autosam_img = resize_to_h(autosam_img, target_h)

        # Add AutoSAM mIoU to header
        as_met = autosam_metrics.get(crop_name, {})
        as_iou = as_met.get("mean_iou", 0)

        sep = 6
        spacer = np.zeros((target_h, sep, 3), dtype=np.uint8)

        fig = np.hstack([
            gt_img, spacer,
            baseline_img, spacer,
            qwen_img, spacer,
            autosam_img
        ])

        cv2.imwrite(str(vis_dir / f"{crop_name}_05_comparison.png"), fig)
        count += 1

        if (count) % 50 == 0:
            print(f"  {count}/{len(dataset)} comparison figures generated")

    print(f"  Done: {count} comparison figures in {vis_dir}")


def _load_csv_as_dict(csv_path):
    """Load CSV metrics into {crop_name: {mean_iou, ...}} dict."""
    import csv
    result = {}
    if not csv_path.exists():
        return result
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cn = row["crop_name"]
            result[cn] = {k: float(v) if k != "crop_name" else v
                          for k, v in row.items()}
    return result


# ===================================================================== #
#  Main                                                                    #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="AutoSAM Training + Evaluation on RWTD")
    parser.add_argument("--data_root", type=str, default="/home/aviad/RWTD")
    parser.add_argument("--sam_checkpoint", type=str,
                        default="/home/aviad/.cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam_model_type", type=str, default="vit_h")
    parser.add_argument("--output_dir", type=str,
                        default=str(_PROJECT_ROOT / "eval_results" / "autosam"))
    parser.add_argument("--base_eval_dir", type=str,
                        default=str(_PROJECT_ROOT / "eval_results" / "sam3_oracle_points"),
                        help="Directory with existing metrics CSVs")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--idim", type=int, default=256,
                        help="Input size for ModelEmb")
    parser.add_argument("--arch", type=int, default=85,
                        help="HarDNet architecture: 39, 68, or 85")
    parser.add_argument("--no_train", action="store_true",
                        help="Skip training, load checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained ModelEmb checkpoint")

    # Evaluation
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--no_comparison", action="store_true",
                        help="Skip comparison figure generation")
    parser.add_argument("--cell_size", type=int, default=320)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load metadata & splits ------------------------------------ #
    data_root = Path(args.data_root)
    with open(data_root / "metadata_phase1.json") as f:
        metadata = json.load(f)
    with open(data_root / "splits.json") as f:
        splits = json.load(f)

    train_indices = splits["train_indices"]
    test_indices = splits["test_indices"]
    all_indices = list(range(len(metadata)))

    print(f"RWTD: {len(metadata)} samples "
          f"(train: {len(train_indices)}, test: {len(test_indices)})")

    # ---- Build SAM -------------------------------------------------- #
    print(f"Loading SAM ({args.sam_model_type}) from {args.sam_checkpoint}")
    sam = sam_model_registry[args.sam_model_type](
        checkpoint=args.sam_checkpoint)
    sam.to(device)
    sam.eval()
    for p in sam.parameters():
        p.requires_grad = False

    transform = ResizeLongestSide(sam.image_encoder.img_size)

    # ---- Build ModelEmb --------------------------------------------- #
    model_emb = ModelEmb(arch=args.arch, depth_wise=False, pretrained=True)
    model_emb.to(device)

    # ---- Training --------------------------------------------------- #
    if not args.no_train:
        # Split train into train/val (80/20)
        n_val = max(1, len(train_indices) // 5)
        val_idx = train_indices[:n_val]
        trn_idx = train_indices[n_val:]

        train_ds = RWTDAutoSAMDataset(metadata, trn_idx, transform, args.idim)
        val_ds = RWTDAutoSAMDataset(metadata, val_idx, transform, args.idim)

        model_emb = train_autosam(
            model_emb, sam, train_ds, val_ds, args, device)
    elif args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model_emb.load_state_dict(
            torch.load(args.checkpoint, map_location=device))
    else:
        print("WARNING: No training and no checkpoint — using random weights!")

    # ---- Evaluation on all 253 samples ------------------------------ #
    eval_ds = RWTDAutoSAMDataset(metadata, all_indices, transform, args.idim)
    all_metrics = evaluate_autosam(model_emb, sam, eval_ds, args, device)

    # ---- Comparison figures ----------------------------------------- #
    if not args.no_comparison:
        generate_comparison_figures(eval_ds, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
