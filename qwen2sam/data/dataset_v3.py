"""
Dataset and Collator for Qwen2SAM v3 (multi-token description architecture).

Key difference from v2: uses <START_SEG_A>...<END_SEG_A> bracketed
descriptions instead of single <SEG_A> tokens. The rich texture descriptions
from metadata become the training targets for Qwen's LM loss.

Data flow:
  V3Dataset.__getitem__  →  raw PIL image, text with descriptions, SAM3 tensor, masks, boxes
  V3Collator             →  batched Qwen inputs + SAM3 inputs + targets
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
from qwen2sam.data.dataset_v2 import (
    SAM3_MEAN, SAM3_STD, SAM3_SIZE,
    preprocess_image_for_sam3,
    resize_mask,
    embed_mask_in_canvas,
    _mask_to_box,
    augment_image_and_masks,
)


class V3Dataset(Dataset):
    """
    Dataset for Qwen2SAM v3 training.

    Uses enriched texture descriptions from metadata as the training text,
    bracketed by START/END markers for description extraction.
    """

    def __init__(
        self,
        data_root: str,
        metadata_file: str = "metadata_phase1.json",
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
        print(f"V3Dataset: {len(self.metadata)} valid samples (SAM3 @ {image_size}){aug_str}")

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

        # ---- Texture descriptions -------------------------------------- #
        texture_a = entry["texture_a"]
        texture_b = entry["texture_b"]

        # Use enriched descriptions as the training text
        # The descriptions become the content between START/END markers
        assistant_text = (
            f"The transition is from "
            f"<START_SEG_A> {texture_a} <END_SEG_A> "
            f"to "
            f"<START_SEG_B> {texture_b} <END_SEG_B>."
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

        # ---- Augmentation ---------------------------------------------- #
        # Always resize to image_size for Qwen spatial consistency.
        # Qwen2.5-VL tiles images based on dimensions — different sizes
        # produce different visual token layouts, breaking spatial features
        # used by CoordHead (Stage 2 point regression).
        image_resized = cv2.resize(image_np, (self.image_size, self.image_size),
                                    interpolation=cv2.INTER_LINEAR)
        if self.augment:
            image_resized, mask_a_sam, mask_b_sam = augment_image_and_masks(
                image_resized, mask_a_sam, mask_b_sam
            )
            mask_a_sam = (mask_a_sam > 0.5).astype(np.float32)
            mask_b_sam = (mask_b_sam > 0.5).astype(np.float32)

        image_pil = Image.fromarray(image_resized)
        sam_image = preprocess_image_for_sam3(image_resized, self.image_size)

        # ---- GT bounding box ------------------------------------------- #
        gt_box_cxcywh, gt_box_xyxy = _mask_to_box(
            mask_a_sam + mask_b_sam, self.image_size, self.image_size
        )

        return {
            "image": image_pil,
            "assistant_text": assistant_text,
            "sam_image": sam_image,
            "mask_a": torch.from_numpy(mask_a_sam),
            "mask_b": torch.from_numpy(mask_b_sam),
            "gt_box_cxcywh": gt_box_cxcywh,
            "gt_box_xyxy": gt_box_xyxy,
            "texture_a": texture_a,
            "texture_b": texture_b,
        }


class V3Collator:
    """
    Collate function for Qwen2SAM v3 DataLoader.

    Uses START/END description markers instead of single SEG tokens.
    """

    INFERENCE_TEMPLATE = (
        "The transition is from "
        "<START_SEG_A> first texture region <END_SEG_A> "
        "to "
        "<START_SEG_B> second texture region <END_SEG_B>."
    )

    def __init__(
        self,
        processor,
        system_prompt: str,
        user_prompt: str,
        token_ids: dict,
        text_embedder=None,
        inference: bool = False,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.token_ids = token_ids
        self.text_embedder = text_embedder
        self.inference = inference

    def __call__(self, samples: list[dict]) -> dict:
        # ---- Build chat messages --------------------------------------- #
        texts = []
        images = []
        for s in samples:
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

        # ---- Labels for LM loss --------------------------------------- #
        labels = create_labels(
            qwen_inputs["input_ids"], qwen_inputs["attention_mask"],
            self.tokenizer,
        )

        # ---- Stack SAM3 inputs ----------------------------------------- #
        sam_images = torch.stack([s["sam_image"] for s in samples])
        masks_a = torch.stack([s["mask_a"] for s in samples])
        masks_b = torch.stack([s["mask_b"] for s in samples])
        gt_boxes_cxcywh = torch.stack([s["gt_box_cxcywh"] for s in samples])
        gt_boxes_xyxy = torch.stack([s["gt_box_xyxy"] for s in samples])

        batch_dict = {
            **{k: v for k, v in qwen_inputs.items()},
            "labels": labels,
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
