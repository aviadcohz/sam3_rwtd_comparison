"""
Dataset and Collator for Qwen2SAM v2 (SAM3-based).

Key differences from Phase 3 dataset:
  - SAM3 uses 1008x1008 input (not 1024)
  - SAM3 normalization: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
  - GT boxes provided in cxcywh normalized format for DETR loss
  - GT masks at 1008x1008 resolution

Data flow:
  Dataset.__getitem__  →  raw PIL image, text, SAM3 tensor, masks, boxes
  V2Collator           →  batched Qwen inputs + SAM3 inputs + targets
"""

import json
import random
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from qwen2sam.data.dataset_phase3 import create_labels


# SAM3 normalization constants
SAM3_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SAM3_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SAM3_SIZE = 1008


def preprocess_image_for_sam3(image: np.ndarray, size: int = SAM3_SIZE) -> torch.Tensor:
    """
    Resize and normalize an image for SAM3 input.

    Args:
        image: (H, W, 3) uint8 RGB image
        size: target resolution (default 1008)

    Returns:
        (3, size, size) float32 tensor
    """
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - SAM3_MEAN) / SAM3_STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)


def resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a binary mask using nearest-neighbor interpolation."""
    resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0.5).astype(np.float32)


def embed_mask_in_canvas(crop_mask, bbox, canvas_h, canvas_w):
    """Embed a crop-sized mask into a full image canvas."""
    x1, y1, x2, y2 = [int(c) for c in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(canvas_w, x2), min(canvas_h, y2)
    bbox_h, bbox_w = y2 - y1, x2 - x1

    if bbox_h <= 0 or bbox_w <= 0:
        return np.zeros((canvas_h, canvas_w), dtype=np.float32)

    if crop_mask.shape[0] != bbox_h or crop_mask.shape[1] != bbox_w:
        resized = cv2.resize(
            crop_mask.astype(np.float32), (bbox_w, bbox_h),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        resized = crop_mask.astype(np.float32)

    resized = (resized > 0.5).astype(np.float32)
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    canvas[y1:y2, x1:x2] = resized
    return canvas


def _mask_to_box(mask: np.ndarray, canvas_h: int, canvas_w: int):
    """
    Compute tight bounding box from a binary mask in normalized coordinates.

    Returns:
        (cxcywh_tensor, xyxy_tensor) — both (4,) float32 tensors, normalized [0,1]
    """
    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        # Empty mask — return zero-area centered box
        return (
            torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float32),
            torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32),
        )
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    # Normalize to [0, 1]
    x1_n, x2_n = x1 / canvas_w, x2 / canvas_w
    y1_n, y2_n = y1 / canvas_h, y2 / canvas_h
    cx = (x1_n + x2_n) / 2.0
    cy = (y1_n + y2_n) / 2.0
    bw = x2_n - x1_n
    bh = y2_n - y1_n
    return (
        torch.tensor([cx, cy, bw, bh], dtype=torch.float32),
        torch.tensor([x1_n, y1_n, x2_n, y2_n], dtype=torch.float32),
    )


def augment_image_and_masks(
    image_np: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply random augmentations consistently to image and masks.

    Augmentations:
      - Horizontal flip (50%)
      - Vertical flip (50%)
      - Color jitter (brightness, contrast, saturation) — image only
      - Random rotation (±15°) — image + masks
    """
    # Horizontal flip
    if random.random() < 0.5:
        image_np = np.ascontiguousarray(image_np[:, ::-1])
        mask_a = np.ascontiguousarray(mask_a[:, ::-1])
        mask_b = np.ascontiguousarray(mask_b[:, ::-1])

    # Vertical flip
    if random.random() < 0.5:
        image_np = np.ascontiguousarray(image_np[::-1])
        mask_a = np.ascontiguousarray(mask_a[::-1])
        mask_b = np.ascontiguousarray(mask_b[::-1])

    # Color jitter (image only, before normalization)
    if random.random() < 0.8:
        img_float = image_np.astype(np.float32)
        # Brightness
        brightness = random.uniform(0.8, 1.2)
        img_float = img_float * brightness
        # Contrast
        contrast = random.uniform(0.8, 1.2)
        mean = img_float.mean()
        img_float = (img_float - mean) * contrast + mean
        # Saturation
        saturation = random.uniform(0.85, 1.15)
        gray = np.mean(img_float, axis=2, keepdims=True)
        img_float = gray + (img_float - gray) * saturation
        image_np = np.clip(img_float, 0, 255).astype(np.uint8)

    # Random rotation (±15°)
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        h, w = image_np.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image_np = cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
        mask_a = cv2.warpAffine(mask_a, M, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_b = cv2.warpAffine(mask_b, M, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Gaussian noise (image only)
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(5, 15), image_np.shape).astype(np.float32)
        image_np = np.clip(image_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Random erasing (image + masks)
    if random.random() < 0.3:
        h, w = image_np.shape[:2]
        erase_h = random.randint(h // 10, h // 4)
        erase_w = random.randint(w // 10, w // 4)
        top = random.randint(0, h - erase_h)
        left = random.randint(0, w - erase_w)
        image_np[top:top + erase_h, left:left + erase_w] = np.random.randint(
            0, 256, (erase_h, erase_w, 3), dtype=np.uint8
        )

    # Gaussian blur (image only)
    if random.random() < 0.2:
        ksize = random.choice([3, 5])
        image_np = cv2.GaussianBlur(image_np, (ksize, ksize), 0)

    return image_np, mask_a, mask_b


class V2Dataset(Dataset):
    """
    Dataset for Qwen2SAM v2 training.

    Each sample provides:
      - image: PIL Image (for Qwen processor)
      - assistant_text: response with <SEG_A>/<SEG_B> tokens
      - sam_image: (3, 1008, 1008) preprocessed for SAM3
      - mask_a / mask_b: (1008, 1008) aligned GT masks
      - gt_box_cxcywh: (4,) crop box in cxcywh normalized
      - gt_box_xyxy: (4,) crop box in xyxy normalized
    """

    def __init__(
        self,
        data_root: str,
        metadata_file: str = "metadata.json",
        image_size: int = SAM3_SIZE,
        augment: bool = False,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.augment = augment

        with open(self.data_root / metadata_file) as f:
            self.metadata = json.load(f)

        self.metadata = self._validate(self.metadata)
        aug_str = " (augmentation ON)" if augment else ""
        print(f"V2Dataset: {len(self.metadata)} valid samples (SAM3 @ {image_size}){aug_str}")

    def _validate(self, entries):
        valid = []
        for e in entries:
            if not all(k in e for k in [
                "image_path", "coords", "mask_a_path", "mask_b_path",
                "texture_a", "texture_b",
            ]):
                continue
            if not all(Path(e[k]).exists() for k in [
                "image_path", "mask_a_path", "mask_b_path"
            ]):
                continue
            if not e.get("texture_a") or not e.get("texture_b"):
                continue
            valid.append(e)
        return valid

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]

        # ---- Load image ------------------------------------------------ #
        image_pil = Image.open(entry["image_path"]).convert("RGB")
        image_np = np.array(image_pil)
        orig_h, orig_w = image_np.shape[:2]

        # ---- Assistant text -------------------------------------------- #
        texture_a = entry["texture_a"]
        texture_b = entry["texture_b"]
        assistant_text = (
            f"The transition is from {texture_a} <SEG_A> "
            f"to {texture_b} <SEG_B>."
        )

        # ---- Mask alignment -------------------------------------------- #
        mask_a_raw = cv2.imread(entry["mask_a_path"], cv2.IMREAD_GRAYSCALE)
        mask_b_raw = cv2.imread(entry["mask_b_path"], cv2.IMREAD_GRAYSCALE)
        bbox = entry["coords"]

        mask_a_bin = (mask_a_raw > 127).astype(np.float32)
        mask_b_bin = (mask_b_raw > 127).astype(np.float32)

        full_mask_a = embed_mask_in_canvas(mask_a_bin, bbox, orig_h, orig_w)
        full_mask_b = embed_mask_in_canvas(mask_b_bin, bbox, orig_h, orig_w)

        mask_a_sam = resize_mask(full_mask_a, self.image_size, self.image_size)
        mask_b_sam = resize_mask(full_mask_b, self.image_size, self.image_size)

        # ---- Augmentation (applied to resized image + masks) ----------- #
        # Always resize to image_size for Qwen spatial consistency.
        # Qwen2.5-VL tiles images based on dimensions — different sizes
        # produce different visual token layouts, breaking spatial features.
        image_resized = cv2.resize(image_np, (self.image_size, self.image_size),
                                    interpolation=cv2.INTER_LINEAR)
        if self.augment:
            image_resized, mask_a_sam, mask_b_sam = augment_image_and_masks(
                image_resized, mask_a_sam, mask_b_sam
            )
            # Re-binarize masks after augmentation
            mask_a_sam = (mask_a_sam > 0.5).astype(np.float32)
            mask_b_sam = (mask_b_sam > 0.5).astype(np.float32)

        image_pil = Image.fromarray(image_resized)
        sam_image = preprocess_image_for_sam3(image_resized, self.image_size)

        # ---- GT bounding box (recompute from augmented masks) ---------- #
        gt_box_cxcywh, gt_box_xyxy = _mask_to_box(
            mask_a_sam + mask_b_sam, self.image_size, self.image_size
        )

        return {
            "image": image_pil,
            "assistant_text": assistant_text,
            "sam_image": sam_image,                              # (3, 1008, 1008)
            "mask_a": torch.from_numpy(mask_a_sam),              # (1008, 1008)
            "mask_b": torch.from_numpy(mask_b_sam),              # (1008, 1008)
            "gt_box_cxcywh": gt_box_cxcywh,                     # (4,)
            "gt_box_xyxy": gt_box_xyxy,                          # (4,)
            "texture_a": texture_a,
            "texture_b": texture_b,
        }


# ===================================================================== #
#  Collator                                                               #
# ===================================================================== #

class V2Collator:
    """
    Collate function for Qwen2SAM v2 DataLoader.

    Same structure as Phase3Collator but with SAM3-specific fields.
    """

    INFERENCE_TEMPLATE = "The transition is from texture <SEG_A> to texture <SEG_B>."

    def __init__(
        self,
        processor,
        system_prompt: str,
        user_prompt: str,
        seg_a_id: int,
        seg_b_id: int,
        text_embedder=None,
        inference: bool = False,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.seg_a_id = seg_a_id
        self.seg_b_id = seg_b_id
        self.text_embedder = text_embedder
        self.inference = inference

    def __call__(self, samples: list[dict]) -> dict:
        # ---- Build chat messages --------------------------------------- #
        texts = []
        images = []
        for s in samples:
            # In inference mode, use a generic template (no GT texture names)
            assistant_text = (
                self.INFERENCE_TEMPLATE if self.inference
                else s["assistant_text"]
            )
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.user_prompt},
                    ],
                },
                {"role": "assistant", "content": assistant_text},
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
            images.append(s["image"])

        # ---- Tokenize -------------------------------------------------- #
        qwen_inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True,
        )

        # ---- Labels ---------------------------------------------------- #
        labels = create_labels(
            qwen_inputs["input_ids"], qwen_inputs["attention_mask"],
            self.tokenizer,
        )

        # ---- Find <SEG_A>/<SEG_B> positions (last occurrence) ---------- #
        seg_a_positions = []
        seg_b_positions = []
        for i in range(len(samples)):
            ids = qwen_inputs["input_ids"][i]
            a_pos = (ids == self.seg_a_id).nonzero(as_tuple=True)[0]
            b_pos = (ids == self.seg_b_id).nonzero(as_tuple=True)[0]
            seg_a_positions.append(a_pos[-1].item() if len(a_pos) > 0 else -1)
            seg_b_positions.append(b_pos[-1].item() if len(b_pos) > 0 else -1)

        # ---- Stack SAM3 inputs ----------------------------------------- #
        sam_images = torch.stack([s["sam_image"] for s in samples])
        masks_a = torch.stack([s["mask_a"] for s in samples])
        masks_b = torch.stack([s["mask_b"] for s in samples])
        gt_boxes_cxcywh = torch.stack([s["gt_box_cxcywh"] for s in samples])
        gt_boxes_xyxy = torch.stack([s["gt_box_xyxy"] for s in samples])

        batch_dict = {
            **{k: v for k, v in qwen_inputs.items()},
            "labels": labels,
            "seg_a_positions": torch.tensor(seg_a_positions, dtype=torch.long),
            "seg_b_positions": torch.tensor(seg_b_positions, dtype=torch.long),
            "sam_images": sam_images,
            "masks_a": masks_a,
            "masks_b": masks_b,
            "gt_boxes_cxcywh": gt_boxes_cxcywh,
            "gt_boxes_xyxy": gt_boxes_xyxy,
        }

        # ---- Alignment targets ----------------------------------------- #
        if self.text_embedder is not None:
            target_a = torch.stack([self.text_embedder[s["texture_a"]] for s in samples])
            target_b = torch.stack([self.text_embedder[s["texture_b"]] for s in samples])
            batch_dict["align_target_a"] = target_a
            batch_dict["align_target_b"] = target_b

        return batch_dict
