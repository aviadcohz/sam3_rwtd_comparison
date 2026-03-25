"""
Phase 3 Dataset and Collator for Brain-Bridge Alignment.

Uses the same underlying data as Phase 1/2 (image + masks + metadata),
but adds text generation targets for Qwen3-VL training.

Dataset returns raw data per sample. The Collator handles:
  - Constructing chat messages with <SEG_A>/<SEG_B> tokens
  - Tokenization via Qwen's processor (handles image tokens)
  - Label creation (mask input, keep assistant response)
  - Finding <SEG_A>/<SEG_B> positions in tokenized sequence
  - SAM image preprocessing and mask alignment

Data flow:
  Dataset.__getitem__  →  raw PIL image, text, SAM tensor, masks
  Phase3Collator       →  batched Qwen inputs + SAM inputs + masks
"""

import json
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

def _get_mask_utils():
    """Lazy import to avoid circular dependency with dataset_v2."""
    from qwen2sam.data.dataset_v2 import (
        embed_mask_in_canvas,
        preprocess_image_for_sam3 as preprocess_image_for_sam,
        resize_mask,
    )
    return embed_mask_in_canvas, preprocess_image_for_sam, resize_mask


class Phase3Dataset(Dataset):
    """
    Dataset for Phase 3 joint training.

    Each sample provides:
      - image: PIL Image (for Qwen processor)
      - assistant_text: response with <SEG_A>/<SEG_B> tokens
      - sam_image: (3, 1024, 1024) preprocessed for SAM
      - mask_a / mask_b: (1024, 1024) aligned GT masks
    """

    def __init__(
        self,
        data_root: str,
        metadata_file: str = "metadata.json",
        image_size: int = 1024,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size

        # Lazy import to break circular dependency
        self._embed_mask_in_canvas, self._preprocess_image_for_sam, self._resize_mask = _get_mask_utils()

        metadata_path = self.data_root / metadata_file
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.metadata = self._validate(self.metadata)
        print(f"Phase3Dataset: {len(self.metadata)} valid samples")

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
            # Need texture labels for text generation
            if not e.get("texture_a") or not e.get("texture_b"):
                continue
            valid.append(e)
        return valid

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]

        # ---- Load image ---------------------------------------------- #
        image_pil = Image.open(entry["image_path"]).convert("RGB")
        image_np = np.array(image_pil)
        orig_h, orig_w = image_np.shape[:2]

        # ---- Construct assistant response text ----------------------- #
        texture_a = entry["texture_a"]
        texture_b = entry["texture_b"]
        assistant_text = (
            f"The transition is from {texture_a} <SEG_A> "
            f"to {texture_b} <SEG_B>."
        )

        # ---- SAM image preprocessing -------------------------------- #
        sam_image = self._preprocess_image_for_sam(image_np, self.image_size)

        # ---- Mask alignment (same as Phase 1) ------------------------ #
        mask_a_raw = cv2.imread(entry["mask_a_path"], cv2.IMREAD_GRAYSCALE)
        mask_b_raw = cv2.imread(entry["mask_b_path"], cv2.IMREAD_GRAYSCALE)
        bbox = entry["coords"]

        mask_a_bin = (mask_a_raw > 127).astype(np.float32)
        mask_b_bin = (mask_b_raw > 127).astype(np.float32)

        full_mask_a = self._embed_mask_in_canvas(mask_a_bin, bbox, orig_h, orig_w)
        full_mask_b = self._embed_mask_in_canvas(mask_b_bin, bbox, orig_h, orig_w)

        mask_a_sam = self._resize_mask(full_mask_a, self.image_size, self.image_size)
        mask_b_sam = self._resize_mask(full_mask_b, self.image_size, self.image_size)

        return {
            "image": image_pil,
            "assistant_text": assistant_text,
            "sam_image": sam_image,                              # (3, 1024, 1024)
            "mask_a": torch.from_numpy(mask_a_sam),              # (1024, 1024)
            "mask_b": torch.from_numpy(mask_b_sam),              # (1024, 1024)
            "description": entry.get("description", ""),
            "texture_a": texture_a,
            "texture_b": texture_b,
        }


# ===================================================================== #
#  Label creation                                                         #
# ===================================================================== #

def _find_response_start(input_ids: torch.Tensor, tokenizer) -> int:
    """
    Find where the assistant response content starts in the token sequence.

    Searches for the last <|im_start|> token (Qwen's assistant header),
    then skips past the "assistant\\n" tokens to find where the actual
    response content begins.

    Returns:
        Index of the first response content token.
    """
    ids = input_ids.tolist()

    # Get special token IDs
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    # Find all <|im_start|> positions
    im_starts = [i for i, t in enumerate(ids) if t == im_start_id]

    if len(im_starts) < 2:
        # Fallback: don't mask anything
        return 0

    # Last <|im_start|> is the assistant section
    last_start = im_starts[-1]

    # Skip past "assistant\n" header tokens
    # Typically: <|im_start|> + "assistant" + "\n" = 2-4 tokens
    # Scan forward to find the newline
    nl_tokens = tokenizer.encode("\n", add_special_tokens=False)
    nl_id = nl_tokens[0] if nl_tokens else None

    for i in range(last_start + 1, min(last_start + 10, len(ids))):
        if nl_id is not None and ids[i] == nl_id:
            return i + 1

    # Fallback: skip a small fixed offset
    return min(last_start + 3, len(ids))


def create_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    """
    Create labels for LM loss: mask input tokens, keep assistant response.

    Args:
        input_ids: (B, seq_len)
        attention_mask: (B, seq_len)
        tokenizer: Qwen tokenizer

    Returns:
        labels: (B, seq_len) with -100 for masked positions
    """
    B = input_ids.shape[0]
    labels = input_ids.clone()

    for i in range(B):
        content_start = _find_response_start(input_ids[i], tokenizer)
        labels[i, :content_start] = -100

    # Also mask padding tokens
    labels[attention_mask == 0] = -100

    return labels


# ===================================================================== #
#  Collator                                                               #
# ===================================================================== #

class Phase3Collator:
    """
    Collate function for Phase 3 DataLoader.

    Handles:
      - Building Qwen chat messages with <SEG_A>/<SEG_B>
      - Batch tokenization via Qwen processor
      - Label creation (mask input, keep response)
      - Finding <SEG> token positions
      - Stacking SAM images and GT masks
    """

    def __init__(
        self,
        processor,
        system_prompt: str,
        user_prompt: str,
        seg_a_id: int,
        seg_b_id: int,
        text_embedder=None,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.seg_a_id = seg_a_id
        self.seg_b_id = seg_b_id
        self.text_embedder = text_embedder

    def __call__(self, samples: list[dict]) -> dict:
        # ---- Build chat messages ------------------------------------- #
        texts = []
        images = []
        for s in samples:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.user_prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": s["assistant_text"],
                },
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
            images.append(s["image"])

        # ---- Tokenize + process images (single pass) ----------------- #
        qwen_inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # ---- Create labels ------------------------------------------- #
        labels = create_labels(
            qwen_inputs["input_ids"],
            qwen_inputs["attention_mask"],
            self.tokenizer,
        )

        # ---- Find <SEG_A>/<SEG_B> positions -------------------------- #
        # Use the LAST occurrence of each token, not the first.
        # The system prompt may also contain literal <SEG_A>/<SEG_B> text
        # (e.g. "mark it with <SEG_A> and <SEG_B> tokens"), so taking
        # the first match would pick the system prompt instead of the
        # assistant response.  The assistant response is always last.
        seg_a_positions = []
        seg_b_positions = []
        for i in range(len(samples)):
            ids = qwen_inputs["input_ids"][i]
            a_pos = (ids == self.seg_a_id).nonzero(as_tuple=True)[0]
            b_pos = (ids == self.seg_b_id).nonzero(as_tuple=True)[0]
            seg_a_positions.append(a_pos[-1].item() if len(a_pos) > 0 else -1)
            seg_b_positions.append(b_pos[-1].item() if len(b_pos) > 0 else -1)

        # ---- Stack SAM inputs ---------------------------------------- #
        sam_images = torch.stack([s["sam_image"] for s in samples])
        masks_a = torch.stack([s["mask_a"] for s in samples])
        masks_b = torch.stack([s["mask_b"] for s in samples])

        batch_dict = {
            # Qwen inputs
            **{k: v for k, v in qwen_inputs.items()},
            "labels": labels,
            # SEG token positions
            "seg_a_positions": torch.tensor(seg_a_positions, dtype=torch.long),
            "seg_b_positions": torch.tensor(seg_b_positions, dtype=torch.long),
            # SAM inputs
            "sam_images": sam_images,
            "masks_a": masks_a,
            "masks_b": masks_b,
        }

        # ---- Alignment target embeddings (if embedder provided) ------ #
        if self.text_embedder is not None:
            target_a = torch.stack([self.text_embedder[s["texture_a"]] for s in samples])
            target_b = torch.stack([self.text_embedder[s["texture_b"]] for s in samples])
            batch_dict["align_target_a"] = target_a  # (B, hidden_dim)
            batch_dict["align_target_b"] = target_b  # (B, hidden_dim)

        return batch_dict
