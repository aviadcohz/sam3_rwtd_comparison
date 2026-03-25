"""
Qwen2SAM v3_tracker: Stage 2 — Multi-Token Descriptions + SAM3 Tracker.

Architecture:
  Stage 1 (v3 DETR):
    Qwen → <START_SEG_A>...<END_SEG_A> descriptions → DescriptionProjector
    → SAM3 Fusion Encoder (multi-token prompt) → DETR Decoder → Coarse masks

  Stage 2 (Tracker):
    Qwen → <POINT_A_1..4>/<POINT_B_1..4> → CoordHead MLP → (x,y) coordinates
    → SAM3 PromptEncoder (points + coarse DETR mask as dense prompt)
    → SAM3 MaskDecoder → Refined masks (288×288 → 1008×1008)

Key design (from v2_tracker ablation):
  - Coarse DETR masks: DETACHED (tracker loss doesn't destabilize DETR)
  - Point coordinates: NOT detached (gradient flows back to LoRA)
  - No PointProjector (ablation showed coords-only is best)
  - Per texture: 4 positive + 4 negative points = 8 total
  - Backbone + trunk features computed once, shared by DETR and tracker
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from qwen2sam.models.qwen2sam_v3 import Qwen2SAMv3
from qwen2sam.models.qwen2sam_v2_tracker import (
    POINT_TOKENS_A,
    POINT_TOKENS_B,
    add_point_tokens,
    CoordHead,
)


class Qwen2SAMv3Tracker(nn.Module):
    """
    Joint v3 DETR + Tracker model for texture boundary segmentation.

    Composes Qwen2SAMv3 (multi-token description DETR) with SAM3 tracker
    heads and a coordinate regression head.
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.cfg = cfg
        tracker_cfg = cfg.get("tracker", {})
        self.num_points = tracker_cfg.get("num_points_per_texture", 4)

        # ---- Build base Qwen2SAMv3 model (DETR path) -------------------- #
        print("Building base Qwen2SAMv3 model...")
        self.base = Qwen2SAMv3(cfg, device=device)

        # ---- Add POINT tokens to Qwen ----------------------------------- #
        self.point_token_ids = add_point_tokens(
            self.base.processor, self.base.qwen,
            num_points_per_texture=self.num_points,
        )
        print(f"  Added {len(self.point_token_ids)} POINT tokens")

        # ---- Coordinate regression head ---------------------------------- #
        qwen_cfg = getattr(self.base.qwen.config, "text_config", self.base.qwen.config)
        self.coord_head = CoordHead(
            hidden_dim=qwen_cfg.hidden_size,
            mid_dim=tracker_cfg.get("coord_head_dim", 256),
        )
        self.coord_head.to(self.device)

        # ---- SAM3 tracker components ------------------------------------- #
        self._add_sam2_neck()
        self._build_sam_heads()
        self._load_sam3_tracker_weights(cfg)

        # ---- Load Stage 1 (v3) checkpoint -------------------------------- #
        v3_ckpt = tracker_cfg.get("v3_checkpoint", None)
        if v3_ckpt is not None:
            self._load_stage1_checkpoint(v3_ckpt)

        # ---- Freeze sam2_convs ------------------------------------------- #
        for param in self.sam2_convs.parameters():
            param.requires_grad = False

        # ---- Optionally freeze DETR components --------------------------- #
        if tracker_cfg.get("freeze_detr", False):
            print("  Freezing DETR components (projector + SAM3 trainable)")
            for param in self.base.sam3.parameters():
                param.requires_grad = False
            for param in self.base.projector.parameters():
                param.requires_grad = False

        # Print trainable params
        counts = self.num_trainable_params()
        print("Trainable parameters:")
        for k, v in counts.items():
            print(f"  {k}: {v:,}")

    # ------------------------------------------------------------------ #
    #  Setup helpers                                                        #
    # ------------------------------------------------------------------ #

    def _add_sam2_neck(self):
        backbone_neck = self.base.sam3.backbone.vision_backbone
        self.sam2_convs = deepcopy(backbone_neck.convs)
        self.sam2_convs.to(self.device)

    def _build_sam_heads(self):
        from sam3.sam.mask_decoder import MaskDecoder
        from sam3.sam.prompt_encoder import PromptEncoder
        from sam3.sam.transformer import TwoWayTransformer

        image_size = 1008
        backbone_stride = 14
        embed_dim = 256

        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=(image_size // backbone_stride,) * 2,
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=embed_dim, mlp_dim=2048, num_heads=8,
            ),
            transformer_dim=embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=0.05,
            dynamic_multimask_stability_thresh=0.98,
        )
        self.sam_prompt_encoder.to(self.device)
        self.sam_mask_decoder.to(self.device)

    def _load_sam3_tracker_weights(self, cfg):
        checkpoint_path = cfg.get("tracker", {}).get("sam3_checkpoint", None)
        if checkpoint_path is None:
            checkpoint_path = cfg.get("model", {}).get("sam3_checkpoint", None)
        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                repo_id="facebook/sam3", filename="sam3.pt"
            )

        print(f"Loading SAM3 tracker weights: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        sam2_state = {}
        for k, v in state.items():
            if k.startswith("detector.backbone.vision_backbone.sam2_convs."):
                new_k = k.replace("detector.backbone.vision_backbone.sam2_convs.", "")
                sam2_state[new_k] = v
        missing, _ = self.sam2_convs.load_state_dict(sam2_state, strict=False)
        print(f"  sam2_convs: loaded {len(sam2_state)} keys, {len(missing)} missing")

        prompt_state, decoder_state = {}, {}
        for k, v in state.items():
            if k.startswith("tracker.sam_prompt_encoder."):
                prompt_state[k[len("tracker.sam_prompt_encoder."):]] = v
            elif k.startswith("tracker.sam_mask_decoder."):
                decoder_state[k[len("tracker.sam_mask_decoder."):]] = v

        m1, _ = self.sam_prompt_encoder.load_state_dict(prompt_state, strict=False)
        m2, _ = self.sam_mask_decoder.load_state_dict(decoder_state, strict=False)
        print(f"  sam_prompt_encoder: {len(prompt_state)} keys, {len(m1)} missing")
        print(f"  sam_mask_decoder: {len(decoder_state)} keys, {len(m2)} missing")

    def _load_stage1_checkpoint(self, checkpoint_path: str):
        print(f"Loading Stage 1 (v3) checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "projector_state_dict" in ckpt:
            missing, unexpected = self.base.projector.load_state_dict(
                ckpt["projector_state_dict"], strict=False
            )
            print(f"  Projector: {len(missing)} missing, {len(unexpected)} unexpected")

        if "sam3_trainable_state_dict" in ckpt:
            self.base.sam3.load_state_dict(
                ckpt["sam3_trainable_state_dict"], strict=False
            )
            print(f"  Loaded SAM3 trainable: {len(ckpt['sam3_trainable_state_dict'])} keys")

        if "qwen_lora_state_dict" in ckpt:
            self.base.qwen.load_state_dict(
                ckpt["qwen_lora_state_dict"], strict=False
            )
            print(f"  Loaded Qwen LoRA: {len(ckpt['qwen_lora_state_dict'])} keys")

        if "align_projector_state_dict" in ckpt and self.base.align_projector is not None:
            self.base.align_projector.load_state_dict(ckpt["align_projector_state_dict"])
            print("  Loaded align projector")

        print(f"  Stage 1 epoch: {ckpt.get('epoch', '?')}")

    # ------------------------------------------------------------------ #
    #  Parameter groups & counting                                         #
    # ------------------------------------------------------------------ #

    def num_trainable_params(self) -> dict[str, int]:
        qwen_n = sum(p.numel() for p in self.base.qwen.parameters() if p.requires_grad)
        proj_n = sum(p.numel() for p in self.base.projector.parameters() if p.requires_grad)
        sam3_n = sum(p.numel() for p in self.base.sam3.parameters() if p.requires_grad)
        coord_n = sum(p.numel() for p in self.coord_head.parameters())
        align_n = sum(p.numel() for p in self.base.align_projector.parameters()) if self.base.align_projector else 0
        prompt_n = sum(p.numel() for p in self.sam_prompt_encoder.parameters() if p.requires_grad)
        decoder_n = sum(p.numel() for p in self.sam_mask_decoder.parameters() if p.requires_grad)
        total = qwen_n + proj_n + sam3_n + coord_n + align_n + prompt_n + decoder_n
        return {
            "qwen_lora": qwen_n,
            "desc_projector": proj_n,
            "sam3_detr": sam3_n,
            "coord_head": coord_n,
            "align_projector": align_n,
            "sam_prompt_encoder": prompt_n,
            "sam_mask_decoder": decoder_n,
            "total": total,
        }

    def get_parameter_groups(self, base_lr: float) -> list[dict]:
        tracker_cfg = self.cfg.get("tracker", {})
        detr_lr_scale = tracker_cfg.get("detr_lr_scale", 0.1)
        tracker_lr_scale = tracker_cfg.get("tracker_lr_scale", 1.0)

        groups = []

        qwen_params = [p for p in self.base.qwen.parameters() if p.requires_grad]
        if qwen_params:
            groups.append({"params": qwen_params, "lr": base_lr, "name": "qwen_lora"})

        proj_params = [p for p in self.base.projector.parameters() if p.requires_grad]
        if proj_params:
            groups.append({"params": proj_params, "lr": base_lr * detr_lr_scale, "name": "desc_projector"})

        sam3_params = [p for p in self.base.sam3.parameters() if p.requires_grad]
        if sam3_params:
            groups.append({"params": sam3_params, "lr": base_lr * detr_lr_scale, "name": "sam3_detr"})

        coord_params = list(self.coord_head.parameters())
        if self.base.align_projector is not None:
            coord_params.extend(list(self.base.align_projector.parameters()))
        groups.append({"params": coord_params, "lr": base_lr, "name": "coord_head"})

        tracker_params = (
            list(self.sam_prompt_encoder.parameters())
            + list(self.sam_mask_decoder.parameters())
        )
        if tracker_params:
            groups.append({"params": tracker_params, "lr": base_lr * tracker_lr_scale, "name": "tracker_sam"})

        return groups

    # ------------------------------------------------------------------ #
    #  SAM2 feature extraction                                             #
    # ------------------------------------------------------------------ #

    def _get_sam2_features(self, trunk_output: torch.Tensor):
        x = trunk_output.detach()
        with torch.no_grad():
            sam2_features = []
            for i in range(min(len(self.sam2_convs), 3)):
                sam2_features.append(self.sam2_convs[i](x))

        sam2_features[0] = self.sam_mask_decoder.conv_s0(sam2_features[0])
        sam2_features[1] = self.sam_mask_decoder.conv_s1(sam2_features[1])

        image_embed = sam2_features[2]
        high_res_feats = [sam2_features[0], sam2_features[1]]
        return image_embed, high_res_feats

    # ------------------------------------------------------------------ #
    #  Coarse mask extraction from DETR                                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _get_coarse_masks(self, outputs: dict):
        B = outputs["pred_logits_a"].shape[0]
        masks_a, masks_b = [], []
        for b in range(B):
            best_a = outputs["pred_logits_a"][b].squeeze(-1).sigmoid().argmax()
            best_b = outputs["pred_logits_b"][b].squeeze(-1).sigmoid().argmax()
            masks_a.append(outputs["pred_masks_a"][b, best_a])
            masks_b.append(outputs["pred_masks_b"][b, best_b])
        coarse_a = torch.stack(masks_a).unsqueeze(1)
        coarse_b = torch.stack(masks_b).unsqueeze(1)
        return coarse_a, coarse_b

    # ------------------------------------------------------------------ #
    #  Tracker refinement                                                  #
    # ------------------------------------------------------------------ #

    def _refine_one(
        self,
        image_embed: torch.Tensor,
        high_res_feats: list[torch.Tensor],
        coarse_mask: torch.Tensor,
        pos_coords: torch.Tensor,
        neg_coords: torch.Tensor,
    ) -> torch.Tensor:
        B = image_embed.shape[0]
        device = image_embed.device

        all_coords = torch.cat([pos_coords, neg_coords], dim=1)
        pos_labels = torch.ones(B, pos_coords.shape[1], dtype=torch.int32, device=device)
        neg_labels = torch.zeros(B, neg_coords.shape[1], dtype=torch.int32, device=device)
        all_labels = torch.cat([pos_labels, neg_labels], dim=1)

        mask_input_size = self.sam_prompt_encoder.mask_input_size
        if coarse_mask.shape[-2:] != mask_input_size:
            sam_mask_prompt = F.interpolate(
                coarse_mask.float(), size=mask_input_size,
                mode="bilinear", align_corners=False, antialias=True,
            )
        else:
            sam_mask_prompt = coarse_mask.float()

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(all_coords, all_labels), boxes=None, masks=sam_mask_prompt,
        )

        image_pe = self.sam_prompt_encoder.get_dense_pe()

        low_res_multimasks, ious, _, _ = self.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats,
        )
        return low_res_multimasks

    # ------------------------------------------------------------------ #
    #  Full forward pass                                                   #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        qwen_inputs: dict,
        sam_images: torch.Tensor,
        point_positions: torch.Tensor,
        seg_grad_to_lm: bool = False,
    ) -> dict:
        """
        Full forward: v3 DETR descriptions + tracker refinement with points.

        Args:
            qwen_inputs: tokenized inputs (with START/END markers + POINT tokens)
            sam_images: (B, 3, 1008, 1008)
            point_positions: (B, 2*num_points) positions of POINT tokens
            seg_grad_to_lm: if False, detach description embeds
        """
        B = sam_images.shape[0]
        N = self.num_points

        # ---- 1. Register trunk hook to capture ViT features -------------- #
        trunk_cache = {}

        def _trunk_hook(module, input, output):
            trunk_cache["xs"] = output

        trunk = self.base.sam3.backbone.vision_backbone.trunk
        hook = trunk.register_forward_hook(_trunk_hook)

        # ---- 2. Qwen forward (teacher forcing) --------------------------- #
        qwen_outputs = self.base.qwen(**qwen_inputs, output_hidden_states=True)
        lm_loss = qwen_outputs.loss
        hidden_states = qwen_outputs.hidden_states[-1]  # (B, seq, 2048)
        input_ids = qwen_inputs["input_ids"]

        # ---- 3. Extract description tokens (v3 style) -------------------- #
        from qwen2sam.models.qwen2sam_v3 import extract_description_tokens, masked_mean_pool

        desc_a, mask_a, len_a = extract_description_tokens(
            hidden_states, input_ids,
            self.base.start_a_id, self.base.end_a_id,
            max_desc_len=self.base.max_desc_len,
        )
        desc_b, mask_b, len_b = extract_description_tokens(
            hidden_states, input_ids,
            self.base.start_b_id, self.base.end_b_id,
            max_desc_len=self.base.max_desc_len,
        )

        # ---- 4. Alignment path ------------------------------------------- #
        pooled_a = masked_mean_pool(desc_a, mask_a)
        pooled_b = masked_mean_pool(desc_b, mask_b)
        if self.base.align_projector is not None:
            align_a = self.base.align_projector(pooled_a)
            align_b = self.base.align_projector(pooled_b)
        else:
            align_a, align_b = pooled_a, pooled_b

        # ---- 5. Gradient isolation for descriptions ---------------------- #
        if not seg_grad_to_lm:
            desc_a = desc_a.detach()
            desc_b = desc_b.detach()

        # ---- 6. Project descriptions to SAM3 space ----------------------- #
        prompt_a = self.base.projector(desc_a)
        prompt_b = self.base.projector(desc_b)

        # ---- 7. Extract POINT tokens (NOT detached) ---------------------- #
        batch_idx = torch.arange(B, device=hidden_states.device)
        h_points = hidden_states[batch_idx[:, None], point_positions]  # (B, 2*N, 2048)

        # ---- 8. Coordinate regression ------------------------------------ #
        coords = self.coord_head(h_points)  # (B, 2*N, 2) in [0,1]
        abs_coords = coords * 1008.0

        # ---- 9. SAM3 backbone (frozen, computed once) -------------------- #
        self.base.sam3.eval()
        with torch.no_grad():
            backbone_out = self.base.sam3.backbone.forward_image(sam_images)
        backbone_out["img_batch_all_stages"] = sam_images

        # ---- 10. DETR passes (two passes, one per texture) --------------- #
        out_a = self.base._run_sam3_from_backbone(backbone_out, prompt_a, mask_a)
        out_b = self.base._run_sam3_from_backbone(backbone_out, prompt_b, mask_b)

        # ---- 11. Extract coarse masks (DETACHED) ------------------------- #
        v3_out = {
            "pred_logits_a": out_a["pred_logits"],
            "pred_logits_b": out_b["pred_logits"],
            "pred_masks_a": out_a["pred_masks"],
            "pred_masks_b": out_b["pred_masks"],
        }
        coarse_a, coarse_b = self._get_coarse_masks(v3_out)

        # ---- 12. SAM2 features from cached trunk output ------------------ #
        hook.remove()
        trunk_output = trunk_cache["xs"][-1]
        image_embed, high_res_feats = self._get_sam2_features(trunk_output)

        # ---- 13. Tracker refinement -------------------------------------- #
        coords_a = abs_coords[:, :N, :]
        coords_b = abs_coords[:, N:, :]

        refined_a = self._refine_one(
            image_embed, high_res_feats, coarse_a,
            pos_coords=coords_a, neg_coords=coords_b,
        )
        refined_b = self._refine_one(
            image_embed, high_res_feats, coarse_b,
            pos_coords=coords_b, neg_coords=coords_a,
        )

        # ---- 14. Build return dict --------------------------------------- #
        return {
            "lm_loss": lm_loss,
            # DETR predictions
            "pred_masks_a": out_a["pred_masks"],
            "pred_logits_a": out_a["pred_logits"],
            "pred_boxes_a": out_a["pred_boxes"],
            "pred_boxes_xyxy_a": out_a["pred_boxes_xyxy"],
            "pred_masks_b": out_b["pred_masks"],
            "pred_logits_b": out_b["pred_logits"],
            "pred_boxes_b": out_b["pred_boxes"],
            "pred_boxes_xyxy_b": out_b["pred_boxes_xyxy"],
            # Alignment embeddings
            "align_a": align_a,
            "align_b": align_b,
            # Tracker refined masks
            "refined_masks_a": refined_a,
            "refined_masks_b": refined_b,
            # Coarse DETR masks (for monitoring)
            "coarse_masks_a": coarse_a,
            "coarse_masks_b": coarse_b,
            # Point coordinates
            "point_coords": coords,
            # Description info
            "desc_lengths_a": len_a,
            "desc_lengths_b": len_b,
        }
