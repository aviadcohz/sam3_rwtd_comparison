"""
Standalone script: compute ONLY Semantic Seg Head metrics and patch results.json.

Skips CLIPSeg, DETR proposal extraction, STD computation, and all visualization.
Only loads SAM3 + Qwen3, generates 5 descriptions, gets semantic masks, evaluates.

Usage:
  python qwen2sam/scripts/run_semseg_only.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_parent_root = _project_root.parent
if str(_parent_root) not in sys.path:
    sys.path.insert(0, str(_parent_root))

from pair_optimizer import (
    QWEN_DIVERSE_PROMPT, QWEN_SYSTEM_PROMPT, N_DESCRIPTIONS, DESC_LABELS,
    load_qwen3_model, qwen3_generate_diverse, parse_diverse_output,
    run_detr_full, postprocess_mask_to_np, find_best_pair,
    compute_iou, compute_dice, compute_ari,
)


def main():
    import torch
    from PIL import Image
    from qwen2sam.models.qwen2sam_v3_tracker import Qwen2SAMv3Tracker
    from qwen2sam.training.train_phase1 import load_config, set_seed
    from qwen2sam.data.dataset_v2 import preprocess_image_for_sam3

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = _project_root / "qwen2sam/configs/v3_tracker_detexure.yaml"
    cfg = load_config(str(config_path))
    image_size = cfg["model"].get("image_size", 1008)

    # Load existing results
    results_path = _project_root / "eval_results/pair_optimizer/results.json"
    with open(results_path) as f:
        all_results = json.load(f)

    # Get sample list from existing results
    existing_samples = [r["crop_name"] for r in all_results["pair_optimizer"]]
    print(f"Will compute SemSeg for {len(existing_samples)} samples")

    meta = json.load(open(Path("/home/aviad/RWTD") / "metadata_phase1.json"))
    meta_by_name = {e["crop_name"]: e for e in meta}

    # Load models
    if "tracker" in cfg:
        cfg["tracker"].pop("v3_checkpoint", None)
    print("Loading SAM3...")
    sam3_model = Qwen2SAMv3Tracker(cfg, device=str(device))
    sam3_model.base.sam3.eval()

    qwen3, qwen3_proc = load_qwen3_model(device)

    semseg_results = []
    gamma = 2.5
    t0 = time.time()

    for i, crop_name in enumerate(existing_samples):
        entry = meta_by_name.get(crop_name)
        if entry is None:
            print(f"  SKIP {crop_name}: not in metadata")
            semseg_results.append({
                "crop_name": crop_name,
                "iou_a": 0, "iou_b": 0, "mean_iou": 0,
                "dice_a": 0, "dice_b": 0, "mean_dice": 0,
                "ari": 0, "best_pair": None,
            })
            continue

        print(f"  [{crop_name}] ({i+1}/{len(existing_samples)})", end="", flush=True)

        image_bgr = cv2.imread(entry["image_path"])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        gt_a = cv2.imread(entry["mask_a_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_b = cv2.imread(entry["mask_b_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_h, gt_w = gt_a.shape

        sam_img = preprocess_image_for_sam3(image_rgb, image_size).unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = sam3_model.base.sam3.backbone.forward_image(sam_img)
            backbone_out["img_batch_all_stages"] = sam_img

            # Generate descriptions
            raw = qwen3_generate_diverse(qwen3, qwen3_proc, image_pil, device)
            descs_a, descs_b, ok = parse_diverse_output(raw)
            if not ok:
                descs_a = [d if d else f"texture A variant {j+1}" for j, d in enumerate(descs_a)]
                descs_b = [d if d else f"texture B variant {j+1}" for j, d in enumerate(descs_b)]

            # Get semantic masks for each description
            semantic_masks_a = []
            semantic_masks_b = []

            for j in range(N_DESCRIPTIONS):
                # Texture A
                t_out_a = sam3_model.base.sam3.backbone.forward_text(
                    [descs_a[j]], device=device)
                feat_a = {
                    "prompt": t_out_a["language_features"].squeeze(1),
                    "mask": t_out_a["language_mask"].squeeze(0),
                }
                _, _, sem_a = run_detr_full(sam3_model, backbone_out, feat_a, 1, device)
                if sem_a is not None:
                    semantic_masks_a.append(
                        postprocess_mask_to_np(sem_a[0, 0], gt_h, gt_w))

                # Texture B
                t_out_b = sam3_model.base.sam3.backbone.forward_text(
                    [descs_b[j]], device=device)
                feat_b = {
                    "prompt": t_out_b["language_features"].squeeze(1),
                    "mask": t_out_b["language_mask"].squeeze(0),
                }
                _, _, sem_b = run_detr_full(sam3_model, backbone_out, feat_b, 1, device)
                if sem_b is not None:
                    semantic_masks_b.append(
                        postprocess_mask_to_np(sem_b[0, 0], gt_h, gt_w))

        # Pair optimizer on semantic masks
        if semantic_masks_a and semantic_masks_b:
            semseg_sorted = find_best_pair(
                semantic_masks_a, semantic_masks_b, descs_a, descs_b, gamma=gamma)

            if semseg_sorted:
                sem_best = semseg_sorted[0]
                sem_win_a = semantic_masks_a[sem_best["idx_a"]]
                sem_win_b = semantic_masks_b[sem_best["idx_b"]]

                swa = cv2.resize(sem_win_a, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
                swb = cv2.resize(sem_win_b, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
                sem_wta_a = (swa > swb).astype(np.float32)
                sem_wta_b = (swb > swa).astype(np.float32)

                # Degenerate WTA fallback
                pix_a = sem_wta_a.sum() / sem_wta_a.size
                pix_b = sem_wta_b.sum() / sem_wta_b.size
                if min(pix_a, pix_b) < 0.02:
                    sem_wta_a = (swa > 0.5).astype(np.float32)
                    sem_wta_b = (swb > 0.5).astype(np.float32)

                # Label swap
                iou_d = compute_iou(sem_wta_a, gt_a) + compute_iou(sem_wta_b, gt_b)
                iou_s = compute_iou(sem_wta_a, gt_b) + compute_iou(sem_wta_b, gt_a)
                if iou_s > iou_d:
                    sem_wta_a, sem_wta_b = sem_wta_b, sem_wta_a

                sem_iou_a = compute_iou(sem_wta_a, gt_a)
                sem_iou_b = compute_iou(sem_wta_b, gt_b)
                sem_dice_a = compute_dice(sem_wta_a, gt_a)
                sem_dice_b = compute_dice(sem_wta_b, gt_b)
                sem_ari = compute_ari(sem_wta_a, sem_wta_b, gt_a, gt_b)
                sem_miou = (sem_iou_a + sem_iou_b) / 2.0
                sem_mdice = (sem_dice_a + sem_dice_b) / 2.0

                print(f"  mIoU={sem_miou:.4f}")
            else:
                sem_iou_a = sem_iou_b = sem_dice_a = sem_dice_b = sem_ari = 0.0
                sem_miou = sem_mdice = 0.0
                sem_best = None
                print(f"  no valid pairs")
        else:
            sem_iou_a = sem_iou_b = sem_dice_a = sem_dice_b = sem_ari = 0.0
            sem_miou = sem_mdice = 0.0
            sem_best = None
            print(f"  no semantic masks")

        semseg_results.append({
            "crop_name": crop_name,
            "iou_a": sem_iou_a, "iou_b": sem_iou_b,
            "mean_iou": sem_miou,
            "dice_a": sem_dice_a, "dice_b": sem_dice_b,
            "mean_dice": sem_mdice,
            "ari": sem_ari,
            "best_pair": sem_best,
        })

    elapsed = time.time() - t0

    # Patch results.json
    all_results["semantic_seg"] = semseg_results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nPatched {results_path} with semantic_seg key")

    # Print summary
    valid = [r for r in semseg_results if r["mean_iou"] > 0]
    if valid:
        m_iou = np.mean([r["mean_iou"] for r in valid])
        m_dice = np.mean([r["mean_dice"] for r in valid])
        m_ari = np.nanmean([r["ari"] for r in valid])

        # Load other approaches for comparison
        po = [r for r in all_results["pair_optimizer"] if r.get("status") == "ok"]
        s3 = all_results.get("sam3_diverse", [])
        qt = all_results.get("qwen_txt_baseline", [])

        po_miou = np.mean([r["mean_iou"] for r in po]) if po else 0
        s3_miou = np.mean([r["mean_iou"] for r in s3]) if s3 else 0
        qt_miou = np.mean([r["mean_iou"] for r in qt]) if qt else 0

        po_mdice = np.mean([r["mean_dice"] for r in po]) if po else 0
        s3_mdice = np.mean([r["mean_dice"] for r in s3]) if s3 else 0
        qt_mdice = np.mean([r["mean_dice"] for r in qt]) if qt else 0

        po_ari = np.nanmean([r["ari"] for r in po]) if po else 0
        s3_ari = np.nanmean([r["ari"] for r in s3]) if s3 else 0
        qt_ari = np.nanmean([r["ari"] for r in qt]) if qt else 0

        print(f"\n{'='*82}")
        print(f"  Complete Summary — {len(valid)} samples ({elapsed:.1f}s)")
        print(f"{'='*82}")
        print(f"\n  {'─'*82}")
        print(f"  {'':15s} {'CSeg PairOpt':>14s} {'SAM3 Diverse':>14s} "
              f"{'SemSeg':>14s} {'QwenTxt':>14s}")
        print(f"  {'─'*82}")
        print(f"  {'mIoU':15s} {po_miou:14.4f} {s3_miou:14.4f} "
              f"{m_iou:14.4f} {qt_miou:14.4f}")
        print(f"  {'mDice':15s} {po_mdice:14.4f} {s3_mdice:14.4f} "
              f"{m_dice:14.4f} {qt_mdice:14.4f}")
        print(f"  {'mARI':15s} {po_ari:14.4f} {s3_ari:14.4f} "
              f"{m_ari:14.4f} {qt_ari:14.4f}")
        print(f"  {'─'*82}")


if __name__ == "__main__":
    main()
