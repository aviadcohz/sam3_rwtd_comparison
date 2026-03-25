"""
Phase 1 Training Loop: SAM Specialization (Domain Adaptation).

Fine-tunes SAM 2.1's Mask Decoder to segment texture regions using
BBox + Point prompts. The image encoder and prompt encoder stay frozen.

Two forward passes per sample (batched efficiently):
  - bbox + point_a → predict Mask A
  - bbox + point_b → predict Mask B

Loss: BCE + Dice + IoU prediction (averaged over Mask A and Mask B)

Usage:
  python -m qwen2sam.training.train_phase1 --config configs/phase1.yaml --data_root /path/to/data

  # Resume from checkpoint:
  python -m qwen2sam.training.train_phase1 --config configs/phase1.yaml --resume checkpoints/phase1/epoch_10.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset, random_split

# Phase 1-specific imports (only needed when running as main, not as utility library)
try:
    from qwen2sam.data.dataset_phase1 import Phase1Dataset
    from qwen2sam.models.losses import phase1_paired_loss, compute_mask_iou
    from qwen2sam.models.sam_wrapper import SAMPhase1Trainer
except ImportError:
    pass  # These modules were removed; utility functions still importable


# -------------------------------------------------------------------- #
#  Utilities                                                             #
# -------------------------------------------------------------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_amp_dtype(cfg: dict) -> torch.dtype:
    dtype_str = cfg.get("amp", {}).get("dtype", "bfloat16")
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]


def get_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def load_splits(splits_path: str) -> dict:
    """Load a pre-computed train/test split from JSON."""
    with open(splits_path) as f:
        return json.load(f)


class AverageMeter:
    """Tracks running average of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


# -------------------------------------------------------------------- #
#  Warmup + cosine LR scheduler                                         #
# -------------------------------------------------------------------- #

class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay.

    Preserves per-group learning rate ratios: each parameter group's
    initial LR (set when the optimizer is constructed) is scaled by the
    same warmup/cosine factor so that differential LR settings (e.g.
    sam_decoder_lr_scale) are maintained throughout training.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, steps_per_epoch):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.min_lr = min_lr
        # Store each group's initial LR so we can scale relative to it
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    @property
    def base_lr(self):
        """Backward compat: return the first group's base LR."""
        return self.base_lrs[0]

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup: scale factor from 0 → 1
            scale = self.current_step / max(self.warmup_steps, 1)
        else:
            # Cosine decay: scale factor from 1 → min_lr/base_lr
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = self.min_lr / max(self.base_lrs[0], 1e-12) + 0.5 * (
                1.0 - self.min_lr / max(self.base_lrs[0], 1e-12)
            ) * (1.0 + np.cos(np.pi * progress))
        for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = blr * scale

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]
        if "base_lrs" in state_dict:
            self.base_lrs = state_dict["base_lrs"]


# -------------------------------------------------------------------- #
#  Training curve plots                                                   #
# -------------------------------------------------------------------- #

def plot_training_curves(
    history: list[dict],
    output_dir: str,
    phase_name: str = "Phase 1",
):
    """
    Save loss + IoU training curves as PNG.

    Args:
        history: list of dicts per epoch, each with keys like
                 train_loss, val_loss, train_iou, val_iou
        output_dir: directory to save plots
        phase_name: label for the plot title
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping training curves")
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    train_iou = [h["train_iou"] for h in history]
    val_loss = [h.get("val_loss") for h in history]
    val_iou = [h.get("val_iou") for h in history]
    has_val = any(v is not None for v in val_loss)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, train_loss, "b-", linewidth=1.5, label="Train")
    if has_val:
        vl = [(e, v) for e, v in zip(epochs, val_loss) if v is not None]
        if vl:
            ax1.plot([x[0] for x in vl], [x[1] for x in vl],
                     "r-", linewidth=1.5, label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{phase_name} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # IoU
    ax2.plot(epochs, train_iou, "b-", linewidth=1.5, label="Train")
    if has_val:
        vi = [(e, v) for e, v in zip(epochs, val_iou) if v is not None]
        if vi:
            ax2.plot([x[0] for x in vi], [x[1] for x in vi],
                     "r-", linewidth=1.5, label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mIoU")
    ax2.set_title(f"{phase_name} — Mask IoU")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    out_path = str(Path(output_dir) / "training_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved: {out_path}")


# -------------------------------------------------------------------- #
#  Training step                                                         #
# -------------------------------------------------------------------- #

def train_one_epoch(
    model: SAMPhase1Trainer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    cfg: dict,
    device: torch.device,
    epoch: int,
) -> dict:
    """Run one training epoch. Returns dict of average metrics."""
    model.sam_model.sam_mask_decoder.train()
    amp_enabled = cfg.get("amp", {}).get("enabled", True)
    amp_dtype = get_amp_dtype(cfg)
    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)
    max_grad_norm = cfg["training"].get("max_grad_norm", 1.0)
    log_every = cfg.get("logging", {}).get("log_every_n_steps", 10)
    loss_cfg = cfg.get("loss", {})

    meters = {k: AverageMeter() for k in ["total", "bce", "dice", "iou_pred", "mask_iou"]}
    t0 = time.time()

    for step, batch in enumerate(loader):
        # ---- Move to device ----------------------------------------- #
        images = batch["image"].to(device)                # (B, 3, 1024, 1024)
        boxes = batch["bbox"].to(device)                  # (B, 4)
        gt_a = batch["mask_a"].to(device)                 # (B, 1024, 1024)
        gt_b = batch["mask_b"].to(device)                 # (B, 1024, 1024)
        points_a = batch["points_a"].to(device)           # (B, N, 2)
        labels_a = batch["labels_a"].to(device)           # (B, N)
        points_b = batch["points_b"].to(device)           # (B, N, 2)
        labels_b = batch["labels_b"].to(device)           # (B, N)

        # ---- Encode image (frozen, no grad) ------------------------- #
        features = model.encode_image(images)

        # ---- Forward: predict both masks (batched) ------------------ #
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            pred_a, iou_a, pred_b, iou_b = model.forward_paired(
                features, points_a, labels_a, points_b, labels_b, boxes
            )

            # Squeeze mask dimension: (B, 1, H, W) → (B, H, W)
            pred_a = pred_a.squeeze(1)
            pred_b = pred_b.squeeze(1)
            iou_a = iou_a.squeeze(1)
            iou_b = iou_b.squeeze(1)

            # ---- Loss --------------------------------------------------- #
            loss, metrics = phase1_paired_loss(
                pred_a, iou_a, gt_a,
                pred_b, iou_b, gt_b,
                bce_weight=loss_cfg.get("bce_weight", 1.0),
                dice_weight=loss_cfg.get("dice_weight", 1.0),
                iou_weight=loss_cfg.get("iou_weight", 1.0),
            )
            loss = loss / grad_accum

        # ---- Backward ------------------------------------------------ #
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_parameters(), max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # ---- Metrics ------------------------------------------------- #
        B = images.shape[0]
        for k in ["total", "bce", "dice", "iou_pred"]:
            meters[k].update(metrics[k], B)

        with torch.no_grad():
            mask_iou = (
                compute_mask_iou(pred_a, gt_a).mean()
                + compute_mask_iou(pred_b, gt_b).mean()
            ) / 2.0
            meters["mask_iou"].update(mask_iou.item(), B)

        # ---- Log ----------------------------------------------------- #
        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(
                f"  [Epoch {epoch+1}] Step {step+1}/{len(loader)} | "
                f"loss={meters['total'].avg:.4f} "
                f"bce={meters['bce'].avg:.4f} "
                f"dice={meters['dice'].avg:.4f} "
                f"iou_pred={meters['iou_pred'].avg:.4f} "
                f"mask_iou={meters['mask_iou'].avg:.4f} "
                f"lr={get_lr(optimizer):.2e} "
                f"({elapsed:.1f}s)"
            )

    return {k: m.avg for k, m in meters.items()}


# -------------------------------------------------------------------- #
#  Validation step                                                       #
# -------------------------------------------------------------------- #

@torch.no_grad()
def validate(
    model: SAMPhase1Trainer,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> dict:
    """Run validation. Returns dict of average metrics."""
    model.sam_model.sam_mask_decoder.eval()
    amp_enabled = cfg.get("amp", {}).get("enabled", True)
    amp_dtype = get_amp_dtype(cfg)
    loss_cfg = cfg.get("loss", {})

    meters = {k: AverageMeter() for k in ["total", "bce", "dice", "iou_pred", "mask_iou"]}

    for batch in loader:
        images = batch["image"].to(device)
        boxes = batch["bbox"].to(device)
        gt_a = batch["mask_a"].to(device)
        gt_b = batch["mask_b"].to(device)
        points_a = batch["points_a"].to(device)
        labels_a = batch["labels_a"].to(device)
        points_b = batch["points_b"].to(device)
        labels_b = batch["labels_b"].to(device)

        features = model.encode_image(images)

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            pred_a, iou_a, pred_b, iou_b = model.forward_paired(
                features, points_a, labels_a, points_b, labels_b, boxes
            )
            pred_a = pred_a.squeeze(1)
            pred_b = pred_b.squeeze(1)
            iou_a = iou_a.squeeze(1)
            iou_b = iou_b.squeeze(1)

            loss, metrics = phase1_paired_loss(
                pred_a, iou_a, gt_a,
                pred_b, iou_b, gt_b,
                bce_weight=loss_cfg.get("bce_weight", 1.0),
                dice_weight=loss_cfg.get("dice_weight", 1.0),
                iou_weight=loss_cfg.get("iou_weight", 1.0),
            )

        B = images.shape[0]
        for k in ["total", "bce", "dice", "iou_pred"]:
            meters[k].update(metrics[k], B)

        mask_iou = (
            compute_mask_iou(pred_a, gt_a).mean()
            + compute_mask_iou(pred_b, gt_b).mean()
        ) / 2.0
        meters["mask_iou"].update(mask_iou.item(), B)

    return {k: m.avg for k, m in meters.items()}


# -------------------------------------------------------------------- #
#  Checkpointing                                                         #
# -------------------------------------------------------------------- #

def save_checkpoint(
    model: SAMPhase1Trainer,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    metrics: dict,
    save_path: str,
):
    torch.save(
        {
            "epoch": epoch,
            "mask_decoder_state_dict": model.sam_model.sam_mask_decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": metrics,
        },
        save_path,
    )
    print(f"  Checkpoint saved: {save_path}")


def load_checkpoint(
    model: SAMPhase1Trainer,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    checkpoint_path: str,
) -> int:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.sam_model.sam_mask_decoder.load_state_dict(ckpt["mask_decoder_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    epoch = ckpt["epoch"]
    print(f"  Resumed from epoch {epoch+1} ({checkpoint_path})")
    return epoch


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int):
    """Remove old checkpoints, keeping only the N most recent."""
    ckpts = sorted(Path(checkpoint_dir).glob("epoch_*.pt"))
    for ckpt in ckpts[:-keep_last_n]:
        ckpt.unlink()


# -------------------------------------------------------------------- #
#  Main                                                                  #
# -------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Phase 1: SAM Specialization")
    parser.add_argument("--config", type=str, required=True, help="Path to phase1.yaml")
    parser.add_argument("--data_root", type=str, default=None, help="Override data root")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--splits", type=str, default=None, help="Path to splits.json for fixed train/test split")
    args = parser.parse_args()

    # ---- Config ------------------------------------------------------ #
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        try:
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except AttributeError:
            pass

    # ---- Data -------------------------------------------------------- #
    data_root = args.data_root or cfg["data"]["data_root"]
    if not data_root:
        raise ValueError("data_root must be specified (config or --data_root)")

    dataset = Phase1Dataset(
        data_root=data_root,
        metadata_file=cfg["data"].get("metadata_file", "metadata.json"),
        image_size=cfg["model"].get("image_size", 1024),
        num_points_per_mask=cfg["data"].get("num_points_per_mask", 2),
    )

    # Train/val split
    if args.splits:
        splits = load_splits(args.splits)
        train_set = Subset(dataset, splits["train_indices"])
        val_set = Subset(dataset, splits["test_indices"])
        print(f"Using fixed splits from {args.splits}")
    else:
        val_frac = cfg["data"].get("val_split", 0.1)
        val_size = int(len(dataset) * val_frac)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
        )
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    batch_size = cfg["training"].get("batch_size", 4)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=cfg["data"].get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=cfg["data"].get("pin_memory", True),
    )

    # ---- Model ------------------------------------------------------- #
    print("Loading SAM 2.1...")
    model_cfg = cfg["model"]
    model = SAMPhase1Trainer(
        model_name=model_cfg.get("sam_model", "facebook/sam2.1-hiera-large"),
        sam_config=model_cfg.get("sam_config"),
        sam_checkpoint=model_cfg.get("sam_checkpoint"),
        device=str(device),
    )
    print(f"Trainable parameters (mask decoder): {model.num_trainable_params():,}")

    # ---- Optimizer --------------------------------------------------- #
    train_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    steps_per_epoch = len(train_loader) // train_cfg.get("gradient_accumulation_steps", 1)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        total_epochs=train_cfg.get("num_epochs", 50),
        min_lr=train_cfg.get("min_lr", 1e-6),
        steps_per_epoch=steps_per_epoch,
    )

    amp_enabled = cfg.get("amp", {}).get("enabled", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ---- Resume ------------------------------------------------------ #
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, scaler, args.resume) + 1

    # ---- Checkpoint dir ---------------------------------------------- #
    ckpt_cfg = cfg.get("checkpoint", {})
    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/phase1"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Wandb (optional) -------------------------------------------- #
    log_cfg = cfg.get("logging", {})
    use_wandb = log_cfg.get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=log_cfg.get("wandb_project", "qwen2sam"),
            name=log_cfg.get("wandb_run_name", "phase1"),
            config=cfg,
        )

    # ---- Training loop ----------------------------------------------- #
    num_epochs = train_cfg.get("num_epochs", 50)
    best_val_loss = float("inf")
    training_history = []

    print(f"\n{'='*60}")
    print("Phase 1: SAM Specialization — Training Start")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, num_epochs):
        t_epoch = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg, device, epoch
        )

        # Validate
        val_metrics = {}
        val_every = cfg.get("validation", {}).get("every_n_epochs", 1)
        if (epoch + 1) % val_every == 0:
            val_metrics = validate(model, val_loader, cfg, device)

        # Track history for plots
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["total"],
            "train_iou": train_metrics["mask_iou"],
            "val_loss": val_metrics.get("total"),
            "val_iou": val_metrics.get("mask_iou"),
        })

        # Epoch summary
        elapsed = time.time() - t_epoch
        print(
            f"\nEpoch {epoch+1}/{num_epochs} ({elapsed:.1f}s) | "
            f"Train: loss={train_metrics['total']:.4f} iou={train_metrics['mask_iou']:.4f}"
            + (
                f" | Val: loss={val_metrics['total']:.4f} iou={val_metrics['mask_iou']:.4f}"
                if val_metrics else ""
            )
        )

        # Wandb logging
        if use_wandb:
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
            log_dict["lr"] = get_lr(optimizer)
            log_dict["epoch"] = epoch + 1
            wandb.log(log_dict)

        # Save checkpoint
        save_every = ckpt_cfg.get("save_every_n_epochs", 5)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                str(ckpt_dir / f"epoch_{epoch+1:04d}.pt"),
            )
            keep_n = ckpt_cfg.get("keep_last_n", 3)
            cleanup_old_checkpoints(str(ckpt_dir), keep_n + 1)  # +1 for best

        # Save best model
        if val_metrics and val_metrics.get("total", float("inf")) < best_val_loss:
            best_val_loss = val_metrics["total"]
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch,
                val_metrics,
                str(ckpt_dir / "best.pt"),
            )
            print(f"  New best model (val_loss={best_val_loss:.4f})")

    # ---- Training curves --------------------------------------------- #
    if training_history:
        plot_training_curves(training_history, str(ckpt_dir), phase_name="Phase 1")

    # ---- Done -------------------------------------------------------- #
    print(f"\n{'='*60}")
    print("Phase 1 training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"{'='*60}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
