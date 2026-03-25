"""
Qwen2SAM v3: Multi-token description architecture.

Architecture:
  Qwen2.5-VL (frozen + LoRA) → generates rich texture descriptions
    → Extract all tokens between <START_SEG_A>...<END_SEG_A> markers
    → Per-token projection (2048 → 256) with padding mask
    → Injected as variable-length prompt into SAM3 Fusion Encoder
    → DETR Decoder (200 queries)
    → SAM3 Segmentation Head → masks + boxes + scores

Key difference from v2: instead of compressing texture info into a single
SEG token, v3 gives SAM3 cross-attention access to each word of the
description independently (material, pattern, size, boundary, etc.).

Two passes per image: one for texture_a, one for texture_b.
Backbone visual features are computed once and reused.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen2sam.models.qwen2sam_v2 import (
    load_qwen_processor,
    load_qwen_model,
    apply_lora,
    HiResHead,
)


# ===================================================================== #
#  Description tokens                                                      #
# ===================================================================== #

START_SEG_A = "<START_SEG_A>"
END_SEG_A = "<END_SEG_A>"
START_SEG_B = "<START_SEG_B>"
END_SEG_B = "<END_SEG_B>"

# Also keep legacy SEG tokens for backward compat with alignment
SEG_A_TOKEN = "<SEG_A>"
SEG_B_TOKEN = "<SEG_B>"


def add_v3_tokens(processor, model):
    """Add all v3 special tokens and return their IDs."""
    tokenizer = processor.tokenizer
    new_tokens = [
        START_SEG_A, END_SEG_A,
        START_SEG_B, END_SEG_B,
    ]
    num_added = tokenizer.add_tokens(new_tokens, special_tokens=True)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in new_tokens}
    return ids


# ===================================================================== #
#  Description Projector                                                   #
# ===================================================================== #

class DescriptionProjector(nn.Module):
    """
    Per-token MLP projector for variable-length descriptions.

    Projects each description token independently from LLM space (2048)
    to SAM3 prompt space (256). Handles batched inputs with padding.
    """

    def __init__(self, llm_dim: int = 2048, sam_dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, sam_dim),
        )

    def forward(self, desc_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            desc_embeds: (B, max_len, llm_dim) — padded description tokens

        Returns:
            (B, max_len, sam_dim) — projected, padding positions are garbage
                                     (masked out by prompt_key_padding_mask)
        """
        return self.proj(desc_embeds)


# ===================================================================== #
#  Description extraction                                                  #
# ===================================================================== #

def extract_description_tokens(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    start_id: int,
    end_id: int,
    max_desc_len: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract hidden states between START and END marker tokens.

    For each sample in the batch, finds the first occurrence of start_id
    and end_id, then extracts hidden_states[start+1 : end] (exclusive of
    the markers themselves). Pads/truncates to max_desc_len.

    Args:
        hidden_states: (B, seq_len, hidden_dim) — from Qwen's last layer
        input_ids: (B, seq_len) — token IDs
        start_id: token ID for START marker
        end_id: token ID for END marker
        max_desc_len: maximum description length (pad/truncate to this)

    Returns:
        desc_embeds: (B, max_desc_len, hidden_dim) — padded with zeros
        desc_mask: (B, max_desc_len) — True for PADDING positions
                   (compatible with SAM3's prompt_key_padding_mask)
        desc_lengths: (B,) — actual number of description tokens per sample
    """
    B, _, hidden_dim = hidden_states.shape
    device = hidden_states.device

    desc_embeds = torch.zeros(B, max_desc_len, hidden_dim, device=device,
                               dtype=hidden_states.dtype)
    desc_mask = torch.ones(B, max_desc_len, dtype=torch.bool, device=device)  # all padding
    desc_lengths = torch.zeros(B, dtype=torch.long, device=device)

    for b in range(B):
        ids = input_ids[b]

        # Find first occurrence of start and end markers
        start_positions = (ids == start_id).nonzero(as_tuple=True)[0]
        end_positions = (ids == end_id).nonzero(as_tuple=True)[0]

        if len(start_positions) == 0 or len(end_positions) == 0:
            # Markers not found — leave as padding (will be masked out)
            continue

        start_pos = start_positions[0].item()
        end_pos = end_positions[0].item()

        if end_pos <= start_pos + 1:
            # Empty description (START immediately followed by END)
            continue

        # Extract tokens between markers (exclusive)
        desc_start = start_pos + 1
        desc_end = min(end_pos, desc_start + max_desc_len)
        length = desc_end - desc_start

        desc_embeds[b, :length] = hidden_states[b, desc_start:desc_end]
        desc_mask[b, :length] = False  # False = valid token
        desc_lengths[b] = length

    return desc_embeds, desc_mask, desc_lengths


def masked_mean_pool(embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool embeddings over valid (non-padding) positions.

    Args:
        embeds: (B, N, D)
        mask: (B, N) — True for padding positions

    Returns:
        (B, D) — mean-pooled over valid positions
    """
    valid = (~mask).float().unsqueeze(-1)  # (B, N, 1)
    num_valid = valid.sum(dim=1).clamp(min=1.0)  # (B, 1)
    return (embeds * valid).sum(dim=1) / num_valid


# ===================================================================== #
#  Qwen2SAM v3 Model                                                      #
# ===================================================================== #

class Qwen2SAMv3(nn.Module):
    """
    Multi-token description model for texture boundary segmentation.

    Instead of a single SEG token, extracts variable-length descriptions
    between START/END markers and feeds them as multi-token prompts to
    SAM3's fusion encoder.
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.cfg = cfg
        model_cfg = cfg["model"]
        self.max_desc_len = model_cfg.get("max_desc_tokens", 16)

        # ---- Qwen2.5-VL ------------------------------------------------ #
        qwen_dtype = getattr(torch, model_cfg.get("qwen_dtype", "bfloat16"))
        self.processor = load_qwen_processor(model_cfg["qwen_model"])
        self.qwen = load_qwen_model(model_cfg["qwen_model"], dtype=qwen_dtype)

        # Add v3 tokens
        self.token_ids = add_v3_tokens(self.processor, self.qwen)
        self.start_a_id = self.token_ids[START_SEG_A]
        self.end_a_id = self.token_ids[END_SEG_A]
        self.start_b_id = self.token_ids[START_SEG_B]
        self.end_b_id = self.token_ids[END_SEG_B]

        self.qwen = apply_lora(self.qwen, model_cfg)
        if model_cfg.get("gradient_checkpointing", True):
            self.qwen.enable_input_require_grads()
            self.qwen.gradient_checkpointing_enable()
        self.qwen.to(self.device)

        # ---- SAM3 ------------------------------------------------------ #
        self.sam3 = self._load_sam3(model_cfg)
        self._configure_sam3_training(model_cfg)
        self.sam3.to(device=self.device)

        # ---- Description Projector -------------------------------------- #
        qwen_cfg = getattr(self.qwen.config, "text_config", self.qwen.config)
        llm_dim = qwen_cfg.hidden_size  # 2048 for 3B model
        self.llm_dim = llm_dim

        self.projector = DescriptionProjector(
            llm_dim=llm_dim,
            sam_dim=256,
            hidden_dim=model_cfg.get("desc_projector_hidden", 1024),
        )
        self.projector.to(self.device)

        # ---- Alignment projector (pooled description → sentence-embed dim) #
        align_dim = model_cfg.get("align_embed_dim", 0)
        if align_dim > 0:
            self.align_projector = nn.Linear(llm_dim, align_dim)
            self.align_projector.to(self.device)
            print(f"  AlignProjector: {llm_dim} → {align_dim}")
        else:
            self.align_projector = None

        # ---- HiRes Head ------------------------------------------------ #
        hires_cfg = cfg.get("hires", {})
        if hires_cfg.get("enabled", False):
            self.hires_head = HiResHead(
                in_dim=256,
                thin_dim=hires_cfg.get("thin_dim", 32),
                target_size=model_cfg.get("image_size", 1008),
            )
            self.hires_head.to(self.device)
        else:
            self.hires_head = None

    # ------------------------------------------------------------------ #
    #  SAM3 loading (reused from v2)                                       #
    # ------------------------------------------------------------------ #

    def _load_sam3(self, model_cfg):
        from sam3.model_builder import build_sam3_image_model
        import sam3 as sam3_module
        from pathlib import Path

        bpe_path = str(
            Path(sam3_module.__path__[0]) / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        )
        checkpoint_path = model_cfg.get("sam3_checkpoint", None)

        model = build_sam3_image_model(
            bpe_path=bpe_path,
            eval_mode=False,
            checkpoint_path=checkpoint_path,
            load_from_HF=(checkpoint_path is None),
            enable_segmentation=True,
            device=self.device,
        )
        return model

    def _configure_sam3_training(self, model_cfg):
        for param in self.sam3.parameters():
            param.requires_grad = False

        for param in self.sam3.transformer.encoder.parameters():
            param.requires_grad = True
        self.sam3.transformer.decoder.query_embed.requires_grad_(True)
        if self.sam3.segmentation_head is not None:
            for param in self.sam3.segmentation_head.parameters():
                param.requires_grad = True
        if hasattr(self.sam3, "dot_prod_scoring") and self.sam3.dot_prod_scoring is not None:
            for param in self.sam3.dot_prod_scoring.parameters():
                param.requires_grad = True
        if hasattr(self.sam3, "class_embed") and self.sam3.class_embed is not None:
            self.sam3.class_embed.requires_grad_(True)
        if hasattr(self.sam3.transformer.decoder, "bbox_embed"):
            for param in self.sam3.transformer.decoder.bbox_embed.parameters():
                param.requires_grad = True
        if not model_cfg.get("freeze_decoder_layers", True):
            for param in self.sam3.transformer.decoder.parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------ #
    #  Parameter groups                                                    #
    # ------------------------------------------------------------------ #

    def get_parameter_groups(self, base_lr: float) -> list[dict]:
        sam3_lr_scale = self.cfg["model"].get("sam3_lr_scale", 1.0)

        qwen_params = [p for p in self.qwen.parameters() if p.requires_grad]
        proj_params = list(self.projector.parameters())
        sam3_params = [p for p in self.sam3.parameters() if p.requires_grad]

        groups = [
            {"params": qwen_params, "lr": base_lr, "name": "qwen_lora"},
            {"params": proj_params, "lr": base_lr, "name": "projector"},
        ]
        if sam3_params:
            groups.append({
                "params": sam3_params,
                "lr": base_lr * sam3_lr_scale,
                "name": "sam3_trainable",
            })
        if self.hires_head is not None:
            groups.append({
                "params": list(self.hires_head.parameters()),
                "lr": base_lr * sam3_lr_scale,
                "name": "hires_head",
            })
        if self.align_projector is not None:
            groups.append({
                "params": list(self.align_projector.parameters()),
                "lr": base_lr,
                "name": "align_projector",
            })
        return groups

    def num_trainable_params(self) -> dict[str, int]:
        qwen_n = sum(p.numel() for p in self.qwen.parameters() if p.requires_grad)
        proj_n = sum(p.numel() for p in self.projector.parameters())
        sam3_n = sum(p.numel() for p in self.sam3.parameters() if p.requires_grad)
        hires_n = sum(p.numel() for p in self.hires_head.parameters()) if self.hires_head else 0
        align_n = sum(p.numel() for p in self.align_projector.parameters()) if self.align_projector else 0
        return {
            "qwen_lora": qwen_n,
            "projector": proj_n,
            "sam3_trainable": sam3_n,
            "hires_head": hires_n,
            "align_projector": align_n,
            "total": qwen_n + proj_n + sam3_n + hires_n + align_n,
        }

    # ------------------------------------------------------------------ #
    #  SAM3 pipeline (reused from v2 with mask support)                    #
    # ------------------------------------------------------------------ #

    def _get_img_feats(self, backbone_out, img_ids):
        n_levels = self.sam3.num_feature_levels
        vis_feats = backbone_out["backbone_fpn"][-n_levels:]
        vis_pos_enc = backbone_out["vision_pos_enc"][-n_levels:]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]

        img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
        return img_feats, img_pos_embeds, vis_feat_sizes

    def _run_sam3_from_backbone(
        self,
        backbone_out: dict,
        prompt_embed: torch.Tensor,
        prompt_padding_mask: torch.Tensor = None,
    ) -> dict:
        """
        Run SAM3 encoder → decoder → seg head with an injected prompt.

        Args:
            backbone_out: dict from sam3.backbone.forward_image()
            prompt_embed: (B, N, 256) multi-token description prompt
            prompt_padding_mask: (B, N) True for padding positions
        """
        from sam3.model.model_misc import inverse_sigmoid
        from sam3.model.box_ops import box_cxcywh_to_xyxy

        B = prompt_embed.shape[0]
        device = prompt_embed.device
        img_ids = torch.arange(B, device=device)

        img_feats, img_pos_embeds, vis_feat_sizes = self._get_img_feats(
            backbone_out, img_ids
        )

        # Build prompt in seq-first format: (N, B, 256)
        if prompt_embed.ndim == 2:
            prompt = prompt_embed.unsqueeze(0)
            N_tokens = 1
        else:
            prompt = prompt_embed.transpose(0, 1)
            N_tokens = prompt_embed.shape[1]

        if prompt_padding_mask is None:
            prompt_mask = torch.zeros(B, N_tokens, dtype=torch.bool, device=device)
        else:
            prompt_mask = prompt_padding_mask

        prompt_pos = torch.zeros_like(prompt)

        # Fusion encoder
        memory_dict = self.sam3.transformer.encoder(
            src=[f.clone() for f in img_feats],
            src_key_padding_mask=None,
            src_pos=[p.clone() for p in img_pos_embeds],
            prompt=prompt,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
        )

        encoder_hidden_states = memory_dict["memory"]
        pos_embed = memory_dict["pos_embed"]
        padding_mask = memory_dict["padding_mask"]

        # DETR decoder
        query_embed = self.sam3.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).expand(-1, B, -1).clone()

        hs, reference_boxes, dec_presence_out, _ = self.sam3.transformer.decoder(
            tgt=tgt,
            memory=encoder_hidden_states,
            memory_key_padding_mask=padding_mask,
            pos=pos_embed,
            reference_boxes=None,
            level_start_index=memory_dict["level_start_index"],
            spatial_shapes=memory_dict["spatial_shapes"],
            valid_ratios=memory_dict["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )
        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)

        # Score prediction
        if self.sam3.use_dot_prod_scoring:
            outputs_class = self.sam3.dot_prod_scoring(hs, prompt, prompt_mask)
        else:
            outputs_class = self.sam3.class_embed(hs)

        # Box prediction
        anchor_offsets = self.sam3.transformer.decoder.bbox_embed(hs)
        ref_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (ref_inv_sig + anchor_offsets).sigmoid()
        outputs_xyxy = box_cxcywh_to_xyxy(outputs_coord)

        # Segmentation head
        seg_head = self.sam3.segmentation_head
        enc_hs = encoder_hidden_states
        if seg_head.cross_attend_prompt is not None:
            tgt2 = seg_head.cross_attn_norm(enc_hs)
            tgt2 = seg_head.cross_attend_prompt(
                query=tgt2, key=prompt, value=prompt,
                key_padding_mask=prompt_mask,
            )[0]
            enc_hs = tgt2 + enc_hs

        pixel_embed = seg_head._embed_pixels(
            backbone_feats=backbone_out["backbone_fpn"],
            image_ids=img_ids,
            encoder_hidden_states=enc_hs,
        )

        instance_embeds = seg_head.instance_seg_head(pixel_embed)
        mask_pred = seg_head.mask_predictor(hs[-1], instance_embeds)

        # Semantic segmentation mask (text-independent, from pixel decoder)
        semantic_mask = seg_head.semantic_seg_head(pixel_embed)

        result = {
            "pred_masks": mask_pred,
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_boxes_xyxy": outputs_xyxy[-1],
            "semantic_mask": semantic_mask,
        }

        if self.hires_head is not None:
            mask_embed_queries = seg_head.mask_predictor.mask_embed(hs[-1])
            hires_out = self.hires_head(pixel_embed, mask_embed_queries)
            result["hires_pixel"] = hires_out["hires_pixel"]
            result["hires_queries"] = hires_out["hires_queries"]

        return result

    # ------------------------------------------------------------------ #
    #  Full forward pass                                                   #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        qwen_inputs: dict,
        sam_images: torch.Tensor,
        seg_grad_to_lm: bool = False,
    ) -> dict:
        """
        Full forward pass: Qwen → extract descriptions → project → SAM3.

        Description tokens are found between START_SEG_A/END_SEG_A and
        START_SEG_B/END_SEG_B markers in the token sequence.

        Args:
            qwen_inputs: tokenized inputs (input_ids, attention_mask, etc.)
            sam_images: (B, 3, 1008, 1008) preprocessed for SAM3
            seg_grad_to_lm: if False, detach description embeds before projector

        Returns:
            dict with predictions for both textures + alignment embeddings
        """
        B = sam_images.shape[0]

        # ---- 1. Qwen forward (teacher forcing) ------------------------- #
        qwen_outputs = self.qwen(**qwen_inputs, output_hidden_states=True)
        lm_loss = qwen_outputs.loss
        hidden_states = qwen_outputs.hidden_states[-1]  # (B, seq, hidden_dim)
        input_ids = qwen_inputs["input_ids"]

        # ---- 2. Extract description tokens ------------------------------ #
        desc_a, mask_a, len_a = extract_description_tokens(
            hidden_states, input_ids,
            self.start_a_id, self.end_a_id,
            max_desc_len=self.max_desc_len,
        )  # desc_a: (B, max_len, hidden_dim), mask_a: (B, max_len)

        desc_b, mask_b, len_b = extract_description_tokens(
            hidden_states, input_ids,
            self.start_b_id, self.end_b_id,
            max_desc_len=self.max_desc_len,
        )

        # ---- 3. Alignment path (before detach) -------------------------- #
        # Mean-pool description tokens for alignment
        pooled_a = masked_mean_pool(desc_a, mask_a)  # (B, hidden_dim)
        pooled_b = masked_mean_pool(desc_b, mask_b)

        if self.align_projector is not None:
            align_a = self.align_projector(pooled_a)
            align_b = self.align_projector(pooled_b)
        else:
            align_a, align_b = pooled_a, pooled_b

        # ---- 4. Gradient isolation -------------------------------------- #
        if not seg_grad_to_lm:
            desc_a = desc_a.detach()
            desc_b = desc_b.detach()

        # ---- 5. Project descriptions to SAM3 space ---------------------- #
        prompt_a = self.projector(desc_a)  # (B, max_len, 256)
        prompt_b = self.projector(desc_b)

        # ---- 6. SAM3 backbone (frozen, computed once) ------------------- #
        self.sam3.eval()
        with torch.no_grad():
            backbone_out = self.sam3.backbone.forward_image(sam_images)
        backbone_out["img_batch_all_stages"] = sam_images

        # ---- 7. SAM3 pipeline: two passes ------------------------------- #
        out_a = self._run_sam3_from_backbone(backbone_out, prompt_a, mask_a)
        out_b = self._run_sam3_from_backbone(backbone_out, prompt_b, mask_b)

        result = {
            "lm_loss": lm_loss,
            # Texture A predictions
            "pred_masks_a": out_a["pred_masks"],
            "pred_logits_a": out_a["pred_logits"],
            "pred_boxes_a": out_a["pred_boxes"],
            "pred_boxes_xyxy_a": out_a["pred_boxes_xyxy"],
            # Texture B predictions
            "pred_masks_b": out_b["pred_masks"],
            "pred_logits_b": out_b["pred_logits"],
            "pred_boxes_b": out_b["pred_boxes"],
            "pred_boxes_xyxy_b": out_b["pred_boxes_xyxy"],
            # Alignment embeddings
            "align_a": align_a,
            "align_b": align_b,
            # Description info
            "desc_lengths_a": len_a,
            "desc_lengths_b": len_b,
        }

        if self.hires_head is not None:
            result["hires_pixel_a"] = out_a["hires_pixel"]
            result["hires_queries_a"] = out_a["hires_queries"]
            result["hires_pixel_b"] = out_b["hires_pixel"]
            result["hires_queries_b"] = out_b["hires_queries"]

        return result
