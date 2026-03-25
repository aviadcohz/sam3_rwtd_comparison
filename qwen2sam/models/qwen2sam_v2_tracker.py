"""
Qwen2SAM v2_tracker: Stage 2 — DETR + SAM3 Tracker with Qwen Point Coordinates.

Architecture:
  Stage 1 (DETR, from v2):
    Qwen → <SEG_A>/<SEG_B> → Projector → SAM3 Fusion Encoder → DETR Decoder
    → Segmentation Head → Coarse masks (288×288)

  Stage 2 (Tracker, new — dual path):
    Qwen → <POINT_A_1..4>/<POINT_B_1..4> → CoordHead MLP → (x,y) coordinates
    → SAM3 PromptEncoder (standard point pipeline) + coarse DETR mask (dense prompt)
    → SAM3 MaskDecoder → Refined masks (288×288 → 1008×1008)

    Additionally: POINT hidden states → PointProjector (autoencoder) → 256-dim
    → concatenated as extra sparse tokens into MaskDecoder
    → reconstruction loss (MSE) ensures information preservation

Key design:
  - Coarse DETR masks: DETACHED (tracker loss doesn't destabilize DETR)
  - Point coordinates: NOT detached (gradient flows back to LoRA via coord_head)
  - Point projections: NOT detached (gradient flows through autoencoder encoder)
  - Per texture: 4 positive + 4 negative points = 8 total (SAM3 training domain)
  - Backbone + trunk features computed once, shared by DETR and tracker paths
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from pathlib import Path

from qwen2sam.models.qwen2sam_v2 import Qwen2SAMv2
from qwen2sam.models.projector import QFormerProjector


# ===================================================================== #
#  Point token management                                                 #
# ===================================================================== #

POINT_TOKENS_A = [f"<POINT_A_{i}>" for i in range(1, 9)]  # up to 8, use first N
POINT_TOKENS_B = [f"<POINT_B_{i}>" for i in range(1, 9)]


def add_point_tokens(processor, model, num_points_per_texture: int = 4) -> list[int]:
    """
    Add POINT tokens to the tokenizer and resize model embeddings.

    Returns:
        list of token IDs: [POINT_A_1..N, POINT_B_1..N]
    """
    tokenizer = processor.tokenizer
    tokens = (
        POINT_TOKENS_A[:num_points_per_texture]
        + POINT_TOKENS_B[:num_points_per_texture]
    )
    new_tokens = tokenizer.add_tokens(tokens, special_tokens=True)
    if new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    return [tokenizer.convert_tokens_to_ids(t) for t in tokens]


# ===================================================================== #
#  Coordinate regression head                                             #
# ===================================================================== #

class CoordHead(nn.Module):
    """
    Regresses (x, y) coordinates from Qwen hidden states.

    Input:  (B, N, hidden_dim)  — N point token hidden states
    Output: (B, N, 2) in [0, 1] — normalized coordinates
    """

    def __init__(self, hidden_dim: int = 2048, mid_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


# ===================================================================== #
#  Autoencoder projection: Qwen → SAM3 embedding space                   #
# ===================================================================== #

class PointProjector(nn.Module):
    """
    Autoencoder that projects Qwen hidden states to SAM3 embedding space.

    Encoder: (B, N, qwen_dim) → (B, N, sam_dim)  — injected as extra sparse tokens
    Decoder: (B, N, sam_dim) → (B, N, qwen_dim)  — reconstruction loss target

    The reconstruction loss (MSE) ensures the 256-dim bottleneck preserves
    the semantic information from Qwen's 2048-dim representation.
    """

    def __init__(self, qwen_dim: int = 2048, sam_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(qwen_dim, sam_dim),
            nn.GELU(),
            nn.Linear(sam_dim, sam_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(sam_dim, sam_dim),
            nn.GELU(),
            nn.Linear(sam_dim, qwen_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)              # (B, N, sam_dim)
        reconstructed = self.decoder(latent)  # (B, N, qwen_dim)
        return latent, reconstructed


# ===================================================================== #
#  Qwen2SAM v2 Tracker Model                                             #
# ===================================================================== #

class Qwen2SAMv2Tracker(nn.Module):
    """
    Joint DETR + Tracker model for texture boundary segmentation.

    Composes a Qwen2SAMv2 (DETR path) with SAM3 tracker heads and
    a coordinate regression head. The DETR provides coarse masks,
    Qwen provides point coordinates, and the tracker refines masks.
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.cfg = cfg
        tracker_cfg = cfg.get("tracker", {})
        self.num_points = tracker_cfg.get("num_points_per_texture", 4)
        self.use_point_projector = tracker_cfg.get("use_point_projector", True)
        self.aggregate_proj_tokens = tracker_cfg.get("aggregate_proj_tokens", False)

        # ---- Build base Qwen2SAMv2 model (DETR path) -------------------- #
        print("Building base Qwen2SAMv2 model...")
        self.base = Qwen2SAMv2(cfg, device=device)

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

        # ---- Point projector (autoencoder: Qwen → SAM3 embedding) -------- #
        self.point_projector = PointProjector(
            qwen_dim=qwen_cfg.hidden_size,
            sam_dim=256,
        )
        self.point_projector.to(self.device)
        if not self.use_point_projector:
            # Freeze projector when disabled to avoid wasting compute
            for p in self.point_projector.parameters():
                p.requires_grad = False
            print("  PointProjector DISABLED (coords only)")

        # ---- Alignment projector (Qwen 2048 → embedder dim) --------------- #
        align_dim = tracker_cfg.get("align_embed_dim", 0)  # 0 = no projection (Qwen native)
        if align_dim > 0:
            self.align_projector = nn.Linear(qwen_cfg.hidden_size, align_dim)
            self.align_projector.to(self.device)
            print(f"  AlignProjector: {qwen_cfg.hidden_size} → {align_dim}")
        else:
            self.align_projector = None

        # ---- SAM3 tracker components ------------------------------------- #
        self._add_sam2_neck()
        self._build_sam_heads()
        self._load_sam3_tracker_weights(cfg)

        # ---- Load Stage 1 checkpoint ------------------------------------- #
        v2_ckpt = tracker_cfg.get("v2_checkpoint", None)
        if v2_ckpt is not None:
            self._load_stage1_checkpoint(v2_ckpt)

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
    #  Setup helpers (from v2_refine.py patterns)                          #
    # ------------------------------------------------------------------ #

    def _add_sam2_neck(self):
        """Add SAM2-style neck convolutions for tracker features."""
        backbone_neck = self.base.sam3.backbone.vision_backbone
        self.sam2_convs = deepcopy(backbone_neck.convs)
        self.sam2_convs.to(self.device)

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
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
                depth=2,
                embedding_dim=embed_dim,
                mlp_dim=2048,
                num_heads=8,
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

    def _load_sam3_tracker_weights(self, cfg: dict):
        """Load pretrained SAM3 tracker weights for sam2_convs + SAM heads."""
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

        # Load sam2_convs from detector backbone
        sam2_state = {}
        for k, v in state.items():
            if k.startswith("detector.backbone.vision_backbone.sam2_convs."):
                new_k = k.replace("detector.backbone.vision_backbone.sam2_convs.", "")
                sam2_state[new_k] = v
        missing, _ = self.sam2_convs.load_state_dict(sam2_state, strict=False)
        print(f"  sam2_convs: loaded {len(sam2_state)} keys, {len(missing)} missing")

        # Load SAM heads from tracker section
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
        """Load Stage 1 (v2) checkpoint into base model."""
        print(f"Loading Stage 1 checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Projector (use strict=False to handle architecture changes)
        if "projector_state_dict" in ckpt:
            missing, unexpected = self.base.projector.load_state_dict(
                ckpt["projector_state_dict"], strict=False
            )
            if missing or unexpected:
                print(f"  Projector: {len(missing)} missing, {len(unexpected)} unexpected keys (architecture changed)")
            else:
                print("  Loaded projector")

        # SAM3 trainable params
        if "sam3_trainable_state_dict" in ckpt:
            missing, unexpected = self.base.sam3.load_state_dict(
                ckpt["sam3_trainable_state_dict"], strict=False
            )
            loaded = len(ckpt["sam3_trainable_state_dict"]) - len(unexpected)
            print(f"  Loaded SAM3 trainable: {loaded} keys")

        # Qwen LoRA
        if "qwen_lora_state_dict" in ckpt:
            missing, unexpected = self.base.qwen.load_state_dict(
                ckpt["qwen_lora_state_dict"], strict=False
            )
            loaded = len(ckpt["qwen_lora_state_dict"]) - len(unexpected)
            print(f"  Loaded Qwen LoRA: {loaded} keys")

        print(f"  Stage 1 epoch: {ckpt.get('epoch', '?')}")

    # ------------------------------------------------------------------ #
    #  Parameter groups & counting                                         #
    # ------------------------------------------------------------------ #

    def num_trainable_params(self) -> dict[str, int]:
        qwen_n = sum(p.numel() for p in self.base.qwen.parameters() if p.requires_grad)
        proj_n = sum(p.numel() for p in self.base.projector.parameters() if p.requires_grad)
        sam3_n = sum(p.numel() for p in self.base.sam3.parameters() if p.requires_grad)
        coord_n = sum(p.numel() for p in self.coord_head.parameters())
        point_proj_n = sum(p.numel() for p in self.point_projector.parameters() if p.requires_grad)
        align_proj_n = sum(p.numel() for p in self.align_projector.parameters()) if self.align_projector else 0
        prompt_n = sum(p.numel() for p in self.sam_prompt_encoder.parameters() if p.requires_grad)
        decoder_n = sum(p.numel() for p in self.sam_mask_decoder.parameters() if p.requires_grad)
        total = qwen_n + proj_n + sam3_n + coord_n + point_proj_n + align_proj_n + prompt_n + decoder_n
        return {
            "qwen_lora": qwen_n,
            "projector": proj_n,
            "sam3_detr": sam3_n,
            "coord_head": coord_n,
            "point_projector": point_proj_n,
            "align_projector": align_proj_n,
            "sam_prompt_encoder": prompt_n,
            "sam_mask_decoder": decoder_n,
            "total": total,
        }

    def get_parameter_groups(self, base_lr: float) -> list[dict]:
        tracker_cfg = self.cfg.get("tracker", {})
        detr_lr_scale = tracker_cfg.get("detr_lr_scale", 0.1)
        tracker_lr_scale = tracker_cfg.get("tracker_lr_scale", 1.0)

        groups = []

        # Qwen LoRA (includes POINT token embeddings)
        qwen_params = [p for p in self.base.qwen.parameters() if p.requires_grad]
        if qwen_params:
            groups.append({"params": qwen_params, "lr": base_lr, "name": "qwen_lora"})

        # Projector (lower LR in Stage 2)
        proj_params = [p for p in self.base.projector.parameters() if p.requires_grad]
        if proj_params:
            groups.append({"params": proj_params, "lr": base_lr * detr_lr_scale, "name": "projector"})

        # SAM3 DETR (lower LR in Stage 2)
        sam3_params = [p for p in self.base.sam3.parameters() if p.requires_grad]
        if sam3_params:
            groups.append({"params": sam3_params, "lr": base_lr * detr_lr_scale, "name": "sam3_detr"})

        # Coordinate regression head + point projector + align projector (full LR)
        coord_proj_params = list(self.coord_head.parameters())
        proj_trainable = [p for p in self.point_projector.parameters() if p.requires_grad]
        coord_proj_params.extend(proj_trainable)
        if self.align_projector is not None:
            coord_proj_params.extend(list(self.align_projector.parameters()))
        groups.append({"params": coord_proj_params, "lr": base_lr, "name": "coord_head"})

        # Tracker SAM heads (full LR)
        tracker_params = (
            list(self.sam_prompt_encoder.parameters())
            + list(self.sam_mask_decoder.parameters())
        )
        if tracker_params:
            groups.append({"params": tracker_params, "lr": base_lr * tracker_lr_scale, "name": "tracker_sam"})

        return groups

    # ------------------------------------------------------------------ #
    #  SAM2 feature extraction (shared backbone)                           #
    # ------------------------------------------------------------------ #

    def _get_sam2_features(self, trunk_output: torch.Tensor):
        """
        Apply sam2_convs to cached ViT trunk output for tracker-style features.

        NOTE: No @torch.no_grad() — conv_s0/conv_s1 are trainable and need gradients.
        sam2_convs are frozen (requires_grad=False) so they won't accumulate grads.

        Args:
            trunk_output: (B, 1024, 72, 72) from ViT trunk hook

        Returns:
            image_embed:    (B, 256, 72, 72)
            high_res_feats: [(B, 32, 288, 288), (B, 64, 144, 144)]
        """
        x = trunk_output.detach()  # trunk output is from frozen ViT, detach explicitly

        # SAM2 neck convolutions (frozen, no grad)
        with torch.no_grad():
            sam2_features = []
            for i in range(min(len(self.sam2_convs), 3)):
                sam2_features.append(self.sam2_convs[i](x))

        # Apply conv_s0 and conv_s1 from mask decoder (TRAINABLE — need gradients)
        sam2_features[0] = self.sam_mask_decoder.conv_s0(sam2_features[0])  # 256→32
        sam2_features[1] = self.sam_mask_decoder.conv_s1(sam2_features[1])  # 256→64

        image_embed = sam2_features[2]                                       # (B, 256, 72, 72)
        high_res_feats = [sam2_features[0], sam2_features[1]]               # [(B,32,288), (B,64,144)]
        return image_embed, high_res_feats

    # ------------------------------------------------------------------ #
    #  Coarse mask extraction from DETR                                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _get_coarse_masks(self, v2_outputs: dict):
        """
        Select best-query mask from DETR output (by confidence score).

        Returns:
            coarse_a: (B, 1, H, W) raw logits, detached
            coarse_b: (B, 1, H, W) raw logits, detached
        """
        B = v2_outputs["pred_logits_a"].shape[0]

        masks_a, masks_b = [], []
        for b in range(B):
            best_a = v2_outputs["pred_logits_a"][b].squeeze(-1).sigmoid().argmax()
            best_b = v2_outputs["pred_logits_b"][b].squeeze(-1).sigmoid().argmax()
            masks_a.append(v2_outputs["pred_masks_a"][b, best_a])
            masks_b.append(v2_outputs["pred_masks_b"][b, best_b])

        coarse_a = torch.stack(masks_a).unsqueeze(1)  # (B, 1, H, W)
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
        extra_sparse: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Refine a single texture mask using point prompts + coarse mask.

        Args:
            image_embed:  (B, 256, 72, 72)
            high_res_feats: [(B, 32, 288, 288), (B, 64, 144, 144)]
            coarse_mask:  (B, 1, H, W) raw logits (detached)
            pos_coords:   (B, N_pos, 2) absolute pixel coords — positive points
            neg_coords:   (B, N_neg, 2) absolute pixel coords — negative points
            extra_sparse: (B, M, 256) optional projected embeddings from PointProjector

        Returns:
            low_res_mask: (B, 1, 288, 288) logits
        """
        B = image_embed.shape[0]
        device = image_embed.device

        # Concatenate positive + negative points
        all_coords = torch.cat([pos_coords, neg_coords], dim=1)  # (B, N_pos+N_neg, 2)
        pos_labels = torch.ones(B, pos_coords.shape[1], dtype=torch.int32, device=device)
        neg_labels = torch.zeros(B, neg_coords.shape[1], dtype=torch.int32, device=device)
        all_labels = torch.cat([pos_labels, neg_labels], dim=1)  # (B, N_total)

        # Resize coarse mask to prompt encoder input size (288×288)
        mask_input_size = self.sam_prompt_encoder.mask_input_size
        if coarse_mask.shape[-2:] != mask_input_size:
            sam_mask_prompt = F.interpolate(
                coarse_mask.float(),
                size=mask_input_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        else:
            sam_mask_prompt = coarse_mask.float()

        # Prompt encoder: points + mask → sparse + dense embeddings
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(all_coords, all_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )

        # Concatenate projected Qwen embeddings as extra sparse tokens
        if extra_sparse is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, extra_sparse], dim=1)

        image_pe = self.sam_prompt_encoder.get_dense_pe()

        # Mask decoder → refined masks
        low_res_multimasks, ious, _, _ = self.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

        return low_res_multimasks  # (B, 1, 288, 288)

    # ------------------------------------------------------------------ #
    #  Full forward pass                                                   #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        qwen_inputs: dict,
        sam_images: torch.Tensor,
        seg_a_positions: torch.Tensor,
        seg_b_positions: torch.Tensor,
        point_positions: torch.Tensor,
        seg_grad_to_lm: bool = False,
    ) -> dict:
        """
        Full forward: DETR coarse masks + tracker refinement with point coords.

        Args:
            qwen_inputs: tokenized inputs for Qwen
            sam_images: (B, 3, 1008, 1008) preprocessed
            seg_a_positions: (B,) positions of <SEG_A>
            seg_b_positions: (B,) positions of <SEG_B>
            point_positions: (B, 2*num_points) positions of POINT tokens
            seg_grad_to_lm: if False, detach SEG hidden states for seg path

        Returns:
            dict with DETR predictions + tracker refined masks + point coords
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

        # ---- 3. Extract SEG tokens --------------------------------------- #
        batch_idx = torch.arange(B, device=hidden_states.device)
        h_a = hidden_states[batch_idx, seg_a_positions]  # (B, 2048)
        h_b = hidden_states[batch_idx, seg_b_positions]

        # Project alignment embeddings if using sentence-transformer targets
        if self.align_projector is not None:
            align_a = self.align_projector(h_a)
            align_b = self.align_projector(h_b)
        else:
            align_a, align_b = h_a, h_b
        if not seg_grad_to_lm:
            h_a = h_a.detach()
            h_b = h_b.detach()

        # ---- 4. Extract POINT tokens (NOT detached) ---------------------- #
        # point_positions: (B, 2*N) — first N are A, last N are B
        h_points = hidden_states[batch_idx[:, None], point_positions]  # (B, 2*N, 2048)

        # ---- 5. Coordinate regression ------------------------------------ #
        coords = self.coord_head(h_points)  # (B, 2*N, 2) in [0,1]
        abs_coords = coords * 1008.0  # absolute pixel coordinates

        # ---- 5b. Point projection (autoencoder) -------------------------- #
        proj_a, proj_b, proj_reconstructed = None, None, None
        if self.use_point_projector:
            proj_latent, proj_reconstructed = self.point_projector(h_points)  # (B, 2N, 256), (B, 2N, 2048)
            proj_a = proj_latent[:, :N, :]   # (B, N, 256) — texture A projected embeddings
            proj_b = proj_latent[:, N:, :]   # (B, N, 256) — texture B projected embeddings
            if self.aggregate_proj_tokens:
                # Mean-pool N tokens → 1 token per texture (less disruption to decoder)
                proj_a = proj_a.mean(dim=1, keepdim=True)  # (B, 1, 256)
                proj_b = proj_b.mean(dim=1, keepdim=True)  # (B, 1, 256)

        # ---- 6. Project SEG → SAM3 prompt space -------------------------- #
        if isinstance(self.base.projector, QFormerProjector):
            # Stack last K layers at SEG positions for cross-attention
            all_hs = qwen_outputs.hidden_states
            K = self.base.projector_num_layers
            layer_stack_a = torch.stack(
                [hs[batch_idx, seg_a_positions] for hs in all_hs[-K:]], dim=1
            )  # (B, K, hidden_dim)
            layer_stack_b = torch.stack(
                [hs[batch_idx, seg_b_positions] for hs in all_hs[-K:]], dim=1
            )
            if not seg_grad_to_lm:
                layer_stack_a = layer_stack_a.detach()
                layer_stack_b = layer_stack_b.detach()
            prompt_a = self.base.projector(layer_stack_a)  # (B, N, 256)
            prompt_b = self.base.projector(layer_stack_b)
        else:
            prompt_a = self.base.projector(h_a)  # (B, 256) or (B, N, 256)
            prompt_b = self.base.projector(h_b)

        # ---- 7. SAM3 backbone (frozen, computed once) -------------------- #
        self.base.sam3.eval()
        with torch.no_grad():
            backbone_out = self.base.sam3.backbone.forward_image(sam_images)
        backbone_out["img_batch_all_stages"] = sam_images

        # ---- 8. DETR passes (two passes, one per texture) ---------------- #
        out_a = self.base._run_sam3_from_backbone(backbone_out, prompt_a)
        out_b = self.base._run_sam3_from_backbone(backbone_out, prompt_b)

        # ---- 9. Extract coarse masks (DETACHED) -------------------------- #
        v2_out = {
            "pred_logits_a": out_a["pred_logits"],
            "pred_logits_b": out_b["pred_logits"],
            "pred_masks_a": out_a["pred_masks"],
            "pred_masks_b": out_b["pred_masks"],
        }
        coarse_a, coarse_b = self._get_coarse_masks(v2_out)

        # ---- 10. SAM2 features from cached trunk output ------------------ #
        hook.remove()
        trunk_output = trunk_cache["xs"][-1]  # (B, 1024, 72, 72)
        image_embed, high_res_feats = self._get_sam2_features(trunk_output)

        # ---- 11. Tracker refinement -------------------------------------- #
        # Point coords: first N are texture A, last N are texture B
        coords_a = abs_coords[:, :N, :]   # (B, N, 2) — A's points
        coords_b = abs_coords[:, N:, :]   # (B, N, 2) — B's points

        # For texture A: A-points positive, B-points negative
        refined_a = self._refine_one(
            image_embed, high_res_feats, coarse_a,
            pos_coords=coords_a, neg_coords=coords_b,
            extra_sparse=proj_a,  # None if use_point_projector=False
        )

        # For texture B: B-points positive, A-points negative
        refined_b = self._refine_one(
            image_embed, high_res_feats, coarse_b,
            pos_coords=coords_b, neg_coords=coords_a,
            extra_sparse=proj_b,  # None if use_point_projector=False
        )

        # ---- 12. Build return dict --------------------------------------- #
        result = {
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
            # Tracker refined masks (low-res logits)
            "refined_masks_a": refined_a,  # (B, 1, 288, 288)
            "refined_masks_b": refined_b,
            # Coarse DETR masks (for monitoring)
            "coarse_masks_a": coarse_a,
            "coarse_masks_b": coarse_b,
            # Point coordinates (for supervision/logging)
            "point_coords": coords,  # (B, 2*N, 2) in [0,1]
        }
        # Point projector outputs (only when enabled)
        if proj_reconstructed is not None:
            result["point_reconstructed"] = proj_reconstructed  # (B, 2*N, 2048)
            result["point_hidden"] = h_points                    # (B, 2*N, 2048)
        return result
