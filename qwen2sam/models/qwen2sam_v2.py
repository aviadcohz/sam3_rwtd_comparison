"""
Qwen2SAM v2: Joint model using SAM 3's DETR architecture.

Architecture:
  Qwen2.5-VL (frozen + LoRA) → <SEG_A>/<SEG_B> hidden states
    → MLP Projector (2048 → 256)
    → Injected into SAM3 Fusion Encoder (replacing text encoder output)
    → DETR Decoder (200 queries, Hungarian matching)
    → SAM3 Segmentation Head (PixelDecoder + MaskPredictor)
    → Predicted masks + boxes + scores

Two passes per image: one for texture_a, one for texture_b.
Backbone visual features are computed once and reused.

Trainable components (v1):
  - LoRA adapters on Qwen (q_proj, v_proj)
  - MLP Projector
  - Fusion Encoder (9.5M params)
  - Object queries (200 learned embeddings)
  - Segmentation Head (2.3M)
  - Scoring head + Box head

Frozen:
  - ViT backbone (446M)
  - VETextEncoder (354M, completely bypassed)
  - Decoder layers (unfreeze later if plateau)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from qwen2sam.models.projector import SegTokenProjector, MultiTokenProjector, QFormerProjector


# ===================================================================== #
#  Qwen utility functions (previously in qwen2sam.py)                     #
# ===================================================================== #

SEG_A_TOKEN = "<SEG_A>"
SEG_B_TOKEN = "<SEG_B>"


def load_qwen_processor(model_name: str):
    """Load Qwen2.5-VL processor."""
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_name)


def load_qwen_model(model_name: str, dtype=torch.bfloat16):
    """Load Qwen2.5-VL model."""
    from transformers import Qwen2_5_VLForConditionalGeneration
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype,
    )


def add_seg_tokens(processor, model):
    """Add <SEG_A> and <SEG_B> special tokens to tokenizer and resize embeddings."""
    tokenizer = processor.tokenizer
    new_tokens = [SEG_A_TOKEN, SEG_B_TOKEN]
    num_added = tokenizer.add_tokens(new_tokens, special_tokens=True)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    seg_a_id = tokenizer.convert_tokens_to_ids(SEG_A_TOKEN)
    seg_b_id = tokenizer.convert_tokens_to_ids(SEG_B_TOKEN)
    return seg_a_id, seg_b_id


def apply_lora(model, model_cfg: dict):
    """Apply LoRA adapters to Qwen model."""
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=model_cfg.get("lora_r", 16),
        lora_alpha=model_cfg.get("lora_alpha", 32),
        lora_dropout=model_cfg.get("lora_dropout", 0.0),
        target_modules=model_cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


class HiResHead(nn.Module):
    """
    Thin-channel upsampler for high-resolution mask prediction.

    Projects 256-d pixel_embed and query embeddings into thin_dim (32),
    upsamples pixel features from 288→1008, then computes masks via dot product.

    ~30K new parameters. Memory: ~65MB per texture pass in bf16.
    """

    def __init__(self, in_dim: int = 256, thin_dim: int = 32, target_size: int = 1008):
        super().__init__()
        self.target_size = target_size
        self.thin_dim = thin_dim

        # Project pixel embeddings: 256 → thin_dim
        self.pixel_proj = nn.Conv2d(in_dim, thin_dim, kernel_size=1, bias=False)

        # Learned upsample: 288 → 576 via transposed conv
        self.upsample_conv = nn.ConvTranspose2d(
            thin_dim, thin_dim, kernel_size=2, stride=2, bias=False,
        )
        self.upsample_norm = nn.GroupNorm(8, thin_dim)

        # Refine after bilinear 576→target_size
        self.refine_conv = nn.Conv2d(thin_dim, thin_dim, kernel_size=3, padding=1, bias=False)
        self.refine_norm = nn.GroupNorm(8, thin_dim)

        # Project query embeddings: 256 → thin_dim
        self.query_proj = nn.Linear(in_dim, thin_dim, bias=False)

    def forward(
        self,
        pixel_embed: torch.Tensor,
        query_embed: torch.Tensor,
    ) -> dict:
        """
        Args:
            pixel_embed: (B, 256, H, W) or (256, H, W) — from PixelDecoder at 288×288
            query_embed: (B, Q, 256) — mask_embed(obj_queries[-1])

        Returns:
            dict with hires_pixel (B, thin, target, target) and hires_queries (B, Q, thin)
        """
        squeeze = False
        if pixel_embed.ndim == 3:
            pixel_embed = pixel_embed.unsqueeze(0)
            squeeze = True

        # Project to thin channel space: (B, 256, 288, 288) → (B, 32, 288, 288)
        x = self.pixel_proj(pixel_embed)

        # Learned upsample: 288 → 576
        x = F.relu(self.upsample_norm(self.upsample_conv(x)))

        # Bilinear upsample: 576 → target_size
        x = F.interpolate(
            x, size=(self.target_size, self.target_size),
            mode="bilinear", align_corners=False,
        )

        # Refine at target resolution
        x = F.relu(self.refine_norm(self.refine_conv(x)))

        if squeeze:
            x = x.squeeze(0)

        # Project queries: (B, Q, 256) → (B, Q, 32)
        hires_queries = self.query_proj(query_embed)

        return {"hires_pixel": x, "hires_queries": hires_queries}


class Qwen2SAMv2(nn.Module):
    """
    Joint Qwen2.5-VL + SAM3 model for texture boundary segmentation.

    Qwen provides semantic understanding via LoRA-adapted <SEG> tokens.
    SAM3 provides segmentation via its DETR-based architecture.
    The MLP projector bridges the two embedding spaces (2048 → 256).
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.cfg = cfg
        model_cfg = cfg["model"]

        # ---- Qwen2.5-VL ------------------------------------------------ #
        qwen_dtype = getattr(torch, model_cfg.get("qwen_dtype", "bfloat16"))
        self.processor = load_qwen_processor(model_cfg["qwen_model"])
        self.qwen = load_qwen_model(model_cfg["qwen_model"], dtype=qwen_dtype)

        self.seg_a_id, self.seg_b_id = add_seg_tokens(self.processor, self.qwen)

        self.qwen = apply_lora(self.qwen, model_cfg)
        if model_cfg.get("gradient_checkpointing", True):
            self.qwen.enable_input_require_grads()
            self.qwen.gradient_checkpointing_enable()
        self.qwen.to(self.device)

        # ---- SAM3 ------------------------------------------------------ #
        self.sam3 = self._load_sam3(model_cfg)
        self._configure_sam3_training(model_cfg)
        # Move SAM3 to correct device (keep float32 — autocast handles precision)
        self.sam3.to(device=self.device)

        # ---- Projector -------------------------------------------------- #
        # text_config for composite VL models, fallback for older transformers
        qwen_cfg = getattr(self.qwen.config, "text_config", self.qwen.config)
        llm_dim = qwen_cfg.hidden_size  # 2048 for 3B model
        projector_type = model_cfg.get("projector_type", "single_token")

        if projector_type == "qformer":
            self.projector = QFormerProjector(
                llm_dim=llm_dim,
                sam_dim=256,
                num_tokens=model_cfg.get("projector_num_tokens", 8),
                num_layers=model_cfg.get("projector_num_layers", 8),
                num_heads=model_cfg.get("projector_num_heads", 8),
            )
            self.projector_num_layers = model_cfg.get("projector_num_layers", 8)
        elif projector_type == "multi_token":
            self.projector = MultiTokenProjector(
                llm_dim=llm_dim,
                sam_dim=256,
                num_tokens=model_cfg.get("projector_num_tokens", 8),
                variant=model_cfg.get("projector_variant", "reshape"),
            )
        else:
            self.projector = SegTokenProjector(
                llm_dim=llm_dim,
                sam_dim=256,
                hidden_dim=model_cfg.get("projector_hidden_dim", 1024),
            )
        self.projector.to(self.device)

        # ---- Alignment projector (Qwen 2048 → sentence-embed dim) ------- #
        align_dim = model_cfg.get("align_embed_dim", 0)
        if align_dim > 0:
            self.align_projector = nn.Linear(llm_dim, align_dim)
            self.align_projector.to(self.device)
            print(f"  AlignProjector: {llm_dim} → {align_dim}")
        else:
            self.align_projector = None

        # ---- HiRes Head (optional, for high-res mask prediction) --------- #
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
    #  SAM3 loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_sam3(self, model_cfg: dict):
        """Load SAM3 model with checkpoint."""
        from sam3.model_builder import build_sam3_image_model
        import sam3 as sam3_module

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

    def _configure_sam3_training(self, model_cfg: dict):
        """Configure which SAM3 components are frozen/trainable."""
        # Freeze everything first
        for param in self.sam3.parameters():
            param.requires_grad = False

        # Unfreeze: Fusion encoder (9.5M)
        for param in self.sam3.transformer.encoder.parameters():
            param.requires_grad = True

        # Unfreeze: Object queries
        self.sam3.transformer.decoder.query_embed.requires_grad_(True)

        # Unfreeze: Segmentation head (2.3M)
        if self.sam3.segmentation_head is not None:
            for param in self.sam3.segmentation_head.parameters():
                param.requires_grad = True

        # Unfreeze: Scoring head
        if hasattr(self.sam3, "dot_prod_scoring") and self.sam3.dot_prod_scoring is not None:
            for param in self.sam3.dot_prod_scoring.parameters():
                param.requires_grad = True
        if hasattr(self.sam3, "class_embed") and self.sam3.class_embed is not None:
            self.sam3.class_embed.requires_grad_(True)

        # Unfreeze: Box head
        if hasattr(self.sam3.transformer.decoder, "bbox_embed"):
            for param in self.sam3.transformer.decoder.bbox_embed.parameters():
                param.requires_grad = True

        # Optionally unfreeze decoder layers
        if not model_cfg.get("freeze_decoder_layers", True):
            for param in self.sam3.transformer.decoder.parameters():
                param.requires_grad = True

        # Note: SAM3's activation checkpointing conflicts with AMP autocast.
        # We solve this by running SAM3 outside autocast (weights are already bfloat16).

    # ------------------------------------------------------------------ #
    #  Parameter groups for optimizer                                      #
    # ------------------------------------------------------------------ #

    def get_parameter_groups(self, base_lr: float) -> list[dict]:
        """Create parameter groups with per-component learning rates."""
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
        """Count trainable parameters per component."""
        qwen_n = sum(p.numel() for p in self.qwen.parameters() if p.requires_grad)
        proj_n = sum(p.numel() for p in self.projector.parameters())
        sam3_n = sum(p.numel() for p in self.sam3.parameters() if p.requires_grad)
        hires_n = sum(p.numel() for p in self.hires_head.parameters()) if self.hires_head is not None else 0
        align_n = sum(p.numel() for p in self.align_projector.parameters()) if self.align_projector is not None else 0
        return {
            "qwen_lora": qwen_n,
            "projector": proj_n,
            "sam3_trainable": sam3_n,
            "hires_head": hires_n,
            "align_projector": align_n,
            "total": qwen_n + proj_n + sam3_n + hires_n + align_n,
        }

    # ------------------------------------------------------------------ #
    #  SAM3 pipeline with injected prompt                                  #
    # ------------------------------------------------------------------ #

    def _get_img_feats(self, backbone_out, img_ids):
        """Extract multi-scale image features (mirrors Sam3Image._get_img_feats)."""
        n_levels = self.sam3.num_feature_levels
        vis_feats = backbone_out["backbone_fpn"][-n_levels:]
        vis_pos_enc = backbone_out["vision_pos_enc"][-n_levels:]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]

        # (B, C, H, W) → (HW, B, C)  seq-first
        img_feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
        return img_feats, img_pos_embeds, vis_feat_sizes

    def _run_sam3_from_backbone(
        self,
        backbone_out: dict,
        prompt_embed: torch.Tensor,
    ) -> dict:
        """
        Run SAM3 encoder → decoder → seg head with an injected prompt.

        The prompt_embed replaces SAM3's text encoder output at the
        fusion encoder injection point.

        Args:
            backbone_out: dict from sam3.backbone.forward_image()
            prompt_embed: (B, 256) single-token or (B, N, 256) multi-token

        Returns:
            dict with pred_masks, pred_logits, pred_boxes, pred_boxes_xyxy
        """
        from sam3.model.model_misc import inverse_sigmoid
        from sam3.model.box_ops import box_cxcywh_to_xyxy

        B = prompt_embed.shape[0]
        device = prompt_embed.device
        img_ids = torch.arange(B, device=device)

        # ---- Image features -------------------------------------------- #
        img_feats, img_pos_embeds, vis_feat_sizes = self._get_img_feats(
            backbone_out, img_ids
        )

        # ---- Build prompt (injected embedding, no geometry) ------------ #
        if prompt_embed.ndim == 2:
            # Single-token: (B, 256) → (1, B, 256)
            prompt = prompt_embed.unsqueeze(0)
            N_tokens = 1
        else:
            # Multi-token: (B, N, 256) → (N, B, 256) seq-first
            prompt = prompt_embed.transpose(0, 1)
            N_tokens = prompt_embed.shape[1]
        prompt_mask = torch.zeros(B, N_tokens, dtype=torch.bool, device=device)
        prompt_pos = torch.zeros_like(prompt)

        # ---- Fusion encoder -------------------------------------------- #
        memory_dict = self.sam3.transformer.encoder(
            src=[f.clone() for f in img_feats],
            src_key_padding_mask=None,
            src_pos=[p.clone() for p in img_pos_embeds],
            prompt=prompt,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
        )

        encoder_hidden_states = memory_dict["memory"]    # (HW, B, 256)
        pos_embed = memory_dict["pos_embed"]             # (HW, B, 256)
        padding_mask = memory_dict["padding_mask"]

        # ---- DETR decoder ---------------------------------------------- #
        query_embed = self.sam3.transformer.decoder.query_embed.weight  # (Q, 256)
        tgt = query_embed.unsqueeze(1).expand(-1, B, -1).clone()  # (Q, B, 256)

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
        # hs: (num_layers, Q, B, 256) → (num_layers, B, Q, 256)
        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)

        # ---- Score prediction ------------------------------------------ #
        if self.sam3.use_dot_prod_scoring:
            outputs_class = self.sam3.dot_prod_scoring(hs, prompt, prompt_mask)
        else:
            outputs_class = self.sam3.class_embed(hs)

        # ---- Box prediction -------------------------------------------- #
        anchor_offsets = self.sam3.transformer.decoder.bbox_embed(hs)
        ref_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (ref_inv_sig + anchor_offsets).sigmoid()
        outputs_xyxy = box_cxcywh_to_xyxy(outputs_coord)

        # ---- Segmentation head (inlined for pixel_embed access) --------- #
        seg_head = self.sam3.segmentation_head

        # Cross-attention on encoder hidden states (if configured)
        enc_hs = encoder_hidden_states
        if seg_head.cross_attend_prompt is not None:
            tgt2 = seg_head.cross_attn_norm(enc_hs)
            tgt2 = seg_head.cross_attend_prompt(
                query=tgt2, key=prompt, value=prompt,
                key_padding_mask=prompt_mask,
            )[0]
            enc_hs = tgt2 + enc_hs

        # Pixel embedding (PixelDecoder: 72→144→288 with skip connections)
        pixel_embed = seg_head._embed_pixels(
            backbone_feats=backbone_out["backbone_fpn"],
            image_ids=img_ids,
            encoder_hidden_states=enc_hs,
        )  # (B, 256, 288, 288) or (256, 288, 288) for B=1

        # Standard 288×288 mask prediction
        instance_embeds = seg_head.instance_seg_head(pixel_embed)
        mask_pred = seg_head.mask_predictor(hs[-1], instance_embeds)

        # Use last decoder layer outputs
        result = {
            "pred_masks": mask_pred,                # (B, Q, 288, 288)
            "pred_logits": outputs_class[-1],       # (B, Q, 1)
            "pred_boxes": outputs_coord[-1],        # (B, Q, 4) cxcywh
            "pred_boxes_xyxy": outputs_xyxy[-1],    # (B, Q, 4) xyxy
        }

        # High-res features (when HiResHead is enabled)
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
        seg_a_positions: torch.Tensor,
        seg_b_positions: torch.Tensor,
        seg_grad_to_lm: bool = False,
    ) -> dict:
        """
        Full forward pass: Qwen → project → SAM3 pipeline → masks.

        Args:
            qwen_inputs: tokenized inputs for Qwen (input_ids, attention_mask, etc.)
            sam_images: (B, 3, 1008, 1008) preprocessed for SAM3
            seg_a_positions: (B,) positions of <SEG_A> in token sequence
            seg_b_positions: (B,) positions of <SEG_B> in token sequence
            seg_grad_to_lm: if False, detach hidden states before projector

        Returns:
            dict with predictions for both textures + alignment embeddings
        """
        B = sam_images.shape[0]

        # ---- 1. Qwen forward (teacher forcing) ------------------------- #
        qwen_outputs = self.qwen(**qwen_inputs, output_hidden_states=True)
        lm_loss = qwen_outputs.loss
        hidden_states = qwen_outputs.hidden_states[-1]  # (B, seq, hidden_dim)

        # ---- 2. Extract <SEG> token hidden states ---------------------- #
        batch_idx = torch.arange(B, device=hidden_states.device)
        h_a = hidden_states[batch_idx, seg_a_positions]  # (B, hidden_dim)
        h_b = hidden_states[batch_idx, seg_b_positions]  # (B, hidden_dim)

        # Alignment path (before detach) — keeps gradient to LoRA
        if self.align_projector is not None:
            align_a = self.align_projector(h_a)
            align_b = self.align_projector(h_b)
        else:
            align_a, align_b = h_a, h_b

        # Gradient isolation for segmentation path
        if not seg_grad_to_lm:
            h_a = h_a.detach()
            h_b = h_b.detach()

        # ---- 3. Project to SAM3 prompt space (256-dim) ------------------ #
        if isinstance(self.projector, QFormerProjector):
            # Stack last K layers at SEG positions for cross-attention
            all_hs = qwen_outputs.hidden_states  # tuple of (num_layers+1) tensors
            K = self.projector_num_layers
            layer_stack_a = torch.stack(
                [hs[batch_idx, seg_a_positions] for hs in all_hs[-K:]], dim=1
            )  # (B, K, hidden_dim)
            layer_stack_b = torch.stack(
                [hs[batch_idx, seg_b_positions] for hs in all_hs[-K:]], dim=1
            )  # (B, K, hidden_dim)
            if not seg_grad_to_lm:
                layer_stack_a = layer_stack_a.detach()
                layer_stack_b = layer_stack_b.detach()
            prompt_a = self.projector(layer_stack_a)  # (B, N, 256)
            prompt_b = self.projector(layer_stack_b)  # (B, N, 256)
        else:
            prompt_a = self.projector(h_a)  # (B, 256) or (B, N, 256)
            prompt_b = self.projector(h_b)

        # ---- 4. SAM3 backbone (frozen, computed once) ------------------- #
        # SAM3 runs in eval mode to disable activation checkpointing
        # (which conflicts with AMP autocast). Gradients still flow for
        # trainable params since requires_grad is set independently.
        self.sam3.eval()
        with torch.no_grad():
            backbone_out = self.sam3.backbone.forward_image(sam_images)
        backbone_out["img_batch_all_stages"] = sam_images

        # ---- 5. SAM3 pipeline: two passes ------------------------------- #
        out_a = self._run_sam3_from_backbone(backbone_out, prompt_a)
        out_b = self._run_sam3_from_backbone(backbone_out, prompt_b)

        result = {
            "lm_loss": lm_loss,
            # Texture A predictions
            "pred_masks_a": out_a["pred_masks"],          # (B, Q, H, W)
            "pred_logits_a": out_a["pred_logits"],        # (B, Q, 1)
            "pred_boxes_a": out_a["pred_boxes"],          # (B, Q, 4) cxcywh
            "pred_boxes_xyxy_a": out_a["pred_boxes_xyxy"],
            # Texture B predictions
            "pred_masks_b": out_b["pred_masks"],
            "pred_logits_b": out_b["pred_logits"],
            "pred_boxes_b": out_b["pred_boxes"],
            "pred_boxes_xyxy_b": out_b["pred_boxes_xyxy"],
            # Alignment embeddings (connected to LoRA)
            "align_a": align_a,
            "align_b": align_b,
        }

        # HiRes features (when enabled)
        if self.hires_head is not None:
            result["hires_pixel_a"] = out_a["hires_pixel"]
            result["hires_queries_a"] = out_a["hires_queries"]
            result["hires_pixel_b"] = out_b["hires_pixel"]
            result["hires_queries_b"] = out_b["hires_queries"]

        return result
