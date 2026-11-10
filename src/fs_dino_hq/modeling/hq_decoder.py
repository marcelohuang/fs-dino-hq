# Portions of this file are adapted from:
#   segment_anything/modeling/mask_decoder.py (MaskDecoder.predict_masks)
#     https://github.com/facebookresearch/segment-anything
#     Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#   segment_anything/modeling/mask_decoder_hq.py (MaskDecoderHQ)
#     https://github.com/SysCV/sam-hq
#     Copyright (c) Meta Platforms, Inc. and affiliates. Modified by HQ-SAM team. All rights reserved.
# Both are licensed under the Apache License, Version 2.0 (full text in THIRD_PARTY_NOTICES.md).
# Modifications in this file: DINOv2/adapter feature sources (dino_l3_proj, early_proj,
# adapter_proj), the canonical Stage 2 base pass without the HQ token, and the dual-pass
# residual (predict_dual_canonical).
"""HQMaskDecoder: the SAM mask decoder with an HQ-SAM style high-quality branch
adapted to DINOv2-based features.

Adapted from the ``MaskDecoderHQ`` design of HQ-SAM (Ke et al., 2023,
https://github.com/SysCV/sam-hq, Apache-2.0); subclasses
``segment_anything.modeling.mask_decoder.MaskDecoder`` (Kirillov et al., 2023,
https://github.com/facebookresearch/segment-anything, Apache-2.0).
Differences from ``MaskDecoderHQ``:

  dino_l3_proj        replaces compress_vit_feat (768-channel DINOv2 block-3
                      tokens instead of 1280-channel SAM ViT-H features)
  early_proj          projects the distilled early-adapter feature; used
                      instead of dino_l3_proj when that feature is given
  adapter_proj        same architecture as embedding_encoder, applied to the
                      final-adapter image embedding
  decoder_mask_proj   same architecture as embedding_maskfeature
  predict_dual_canonical
                      residual over a separate decoder pass without the HQ
                      token (see the method docstring)

The SAM decoder weights (iou_token, mask_tokens, output_upscaling,
output_hypernetworks_mlps, transformer, iou_prediction_head) come from the
Stage 2 checkpoint and stay frozen during HQ training; only the modules in
``HQ_MODULE_NAMES`` are trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Type

from segment_anything.modeling.mask_decoder import MaskDecoder, MLP
from segment_anything.modeling.common import LayerNorm2d


class DinoL3Proj(nn.Module):
    """
    Projects DINOv2 layer-3 query feature [B, 768, 37, 37]
    → [B, 32, 256, 256].

    Step 1: bilinear upsample 37→64  (avoids non-power-of-2 ConvTranspose issues)
    Step 2: two ConvTranspose2d stride-2 blocks  64→128→256 spatially
            channels: 768→transformer_dim→transformer_dim//8
    """

    def __init__(self, in_ch: int = 768, transformer_dim: int = 256) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(in_ch, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 768, 37, 37]  (or whatever spatial size)
        x = F.interpolate(x, size=64, mode="bilinear", align_corners=False)
        # x: [B, 768, 64, 64]
        return self.convs(x)
        # output: [B, 32, 256, 256]


class HQMaskDecoder(MaskDecoder):
    """
    Extends MaskDecoder with an HQ branch:

    Tokens during forward_hq:
      position 0              : iou_token
      positions 1..4          : mask_tokens  (original 4, num_mask_tokens=4)
      position 5              : hf_token     (NEW – HQ output token)
      positions 6..           : sparse_prompt_embeddings (MVPG visual prompts)

    The model output comes from predict_dual_canonical (canonical base pass
    without the HQ token + HQ residual from a second pass). predict_with_hq is
    that second pass and returns:
      sam_logits      [B, 1, 256, 256]  — SAM-token output of the shared pass (diagnostic)
      hq_logits       [B, 1, 256, 256]  — HQ residual
      iou_pred        [B, 1]
    """

    HQ_MODULE_NAMES = frozenset([
        "hf_token",
        "hf_mlp",
        "dino_l3_proj",
        "adapter_proj",
        "early_proj",
        "decoder_mask_proj",
    ])

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        dino_l3_in_ch: int = 768,
    ) -> None:
        super().__init__(
            transformer_dim=transformer_dim,
            transformer=transformer,
            num_multimask_outputs=num_multimask_outputs,
            activation=activation,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
        )
        # After super().__init__:
        #   self.num_mask_tokens = num_multimask_outputs + 1 = 4   (for vit_h: 3+1)
        #   self.iou_token         = nn.Embedding(1, transformer_dim)
        #   self.mask_tokens       = nn.Embedding(4, transformer_dim)
        #   self.output_upscaling  = Sequential(ConvTranspose2d ×2) → [B,32,256,256]
        #   self.output_hypernetworks_mlps = ModuleList(4 × MLP)
        #   self.iou_prediction_head = MLP

        # ── HQ-specific modules ────────────────────────────────────────────
        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        # Replaces compress_vit_feat (input ch differs: 768 vs 1280)
        self.dino_l3_proj = DinoL3Proj(in_ch=dino_l3_in_ch, transformer_dim=transformer_dim)

        # Same arch as SAM-HQ embedding_encoder → partial-load weights available
        self.adapter_proj = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )

        # Dedicated projection for the DISTILLED early-adapter feature
        # ([B,256,64,64] → [B,32,256,256]).  Same architecture as adapter_proj
        # but NOT weight-tied: the early/local and global/final branches must be
        # able to specialize independently.  Randomly initialized; trained as
        # part of the HQ branch (listed in HQ_MODULE_NAMES).  Unused (but
        # harmlessly present) when the raw dino_l3_proj path is active.
        self.early_proj = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )

        # Same arch as SAM-HQ embedding_maskfeature → partial-load weights available
        self.decoder_mask_proj = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1),
        )

    # ------------------------------------------------------------------
    # Primary entry point used by FSDinoHQModel
    # ------------------------------------------------------------------

    def predict_with_hq(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        dino_l3_feat: torch.Tensor,
        adapter_feat: torch.Tensor,
        distilled_early_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image_embeddings     : [B, 256, 64, 64]  adapter output upsampled
            image_pe             : [1, 256, 64, 64]  from prompt_encoder.get_dense_pe()
            sparse_prompt_embeddings : [B, 1, 256]   MVPG semantic prompt
            dense_prompt_embeddings  : [B, 256, 64, 64]  MVPG dense prompt
            dino_l3_feat         : [B, 768, 37, 37]  query DINOv2 layer-3 feature
            adapter_feat         : [B, 256, 64, 64]  adapter output (post-upsample)
            distilled_early_feat : [B, 256, 64, 64]  (optional) distilled early adapter
                                   output.  When provided, this replaces the raw
                                   dino_l3_feat for the early/local HQ feature branch:
                                   ``early_proj(distilled_early_feat)`` is used instead
                                   of ``dino_l3_proj(dino_l3_feat)``.  early_proj is a
                                   dedicated module (same arch as adapter_proj, NOT
                                   weight-tied) so the early/local and global/final
                                   branches can specialize independently.
                                   When None (default), the original dino_l3_proj path
                                   is used and behaviour is identical to before.

        Returns dict with keys:
            sam_logits, hq_logits  each [B,1,256,256] (sam_logits: SAM-token
                                   output of this shared pass, diagnostic only)
            iou_pred  [B,1]
            upscaled_embedding_sam  [B,32,256,256]  (for debugging)
        """
        B = image_embeddings.shape[0]

        # ── Build global-local HQ features ────────────────────────────────
        # Each branch → [B, 32, 256, 256]
        if distilled_early_feat is not None:
            # Distilled early adapter path: dedicated early_proj (takes [B,256,64,64])
            feat_dino = self.early_proj(distilled_early_feat)     # early/local (distilled)
        else:
            feat_dino = self.dino_l3_proj(dino_l3_feat)           # early/local (raw DINO-L3)
        feat_adapter = self.adapter_proj(adapter_feat)             # global/final

        # ── Build token sequence  [iou | mask*4 | hf] ──────────────────
        # We do NOT modify self.num_mask_tokens so the parent's hypernetwork
        # indexing remains valid (4 entries in output_hypernetworks_mlps).
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight],
            dim=0,
        )  # [6, 256]
        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, 6, 256]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)  # [B, 6+P, 256]

        # ── Run transformer ────────────────────────────────────────────────
        src = image_embeddings + dense_prompt_embeddings     # [B, 256, 64, 64]
        pos_src = image_pe.expand(B, -1, -1, -1)             # [B, 256, 64, 64]
        b, c, h, w = src.shape

        hs, src_out = self.transformer(src, pos_src, tokens)
        # hs: [B, 6+P, 256]   src_out: [B, 64*64, 256]

        iou_token_out = hs[:, 0, :]              # [B, 256]
        mask_tokens_out = hs[:, 1:5, :]          # [B, 4, 256]  original 4 tokens
        hq_token_out = hs[:, 5, :]               # [B, 256]     HQ token

        # ── SAM upscaling ─────────────────────────────────────────────────
        src_2d = src_out.transpose(1, 2).view(b, c, h, w)           # [B,256,64,64]
        upscaled_embedding_sam = self.output_upscaling(src_2d)       # [B,32,256,256]

        # ── HQ feature fusion ─────────────────────────────────────────────
        feat_decoder = self.decoder_mask_proj(upscaled_embedding_sam)  # [B,32,256,256]
        hq_features = feat_dino + feat_adapter + feat_decoder          # [B,32,256,256]

        # ── SAM logits  (token index 0, multimask_output=False) ───────────
        sam_kernel = self.output_hypernetworks_mlps[0](mask_tokens_out[:, 0, :])
        # sam_kernel: [B, 32]
        c_up, h_up, w_up = upscaled_embedding_sam.shape[1:]
        sam_logits = (
            sam_kernel.unsqueeze(1) @
            upscaled_embedding_sam.view(b, c_up, h_up * w_up)
        ).view(b, 1, h_up, w_up)                                     # [B,1,256,256]

        # ── HQ logits ─────────────────────────────────────────────────────
        hq_kernel = self.hf_mlp(hq_token_out)                        # [B, 32]
        hq_logits = (
            hq_kernel.unsqueeze(1) @
            hq_features.view(b, c_up, h_up * w_up)
        ).view(b, 1, h_up, w_up)                                     # [B,1,256,256]

        # ── IoU prediction (from original head, using slot 0) ─────────────
        iou_pred = self.iou_prediction_head(iou_token_out)            # [B, 4]
        iou_pred = iou_pred[:, :1]                                    # [B, 1]

        return {
            "sam_logits": sam_logits,
            "hq_logits": hq_logits,
            "iou_pred": iou_pred,
            "upscaled_embedding_sam": upscaled_embedding_sam,
        }

    # ------------------------------------------------------------------
    # Residual scale
    # ------------------------------------------------------------------

    @property
    def residual_scale(self) -> float:
        """Coefficient of the HQ residual in the corrected output (fixed at 1)."""
        return 1.0

    # ------------------------------------------------------------------
    # Canonical Stage 2 base pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_canonical_stage2_base(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Stage 2 decoder pass without the HQ token ("canonical base").

        Token layout [iou | mask*4 | prompts], as in Stage 2 training. Uses only
        the frozen Stage 2 decoder modules (transformer, mask/IoU tokens,
        upscaling, hypernetworks); no HQ parameter can influence this output,
        so it is identical for any two HQ checkpoints built on the same Stage 2
        checkpoint.

        This differs from ``predict_with_hq()["sam_logits"]``: in that pass the
        HQ token sits in the same token sequence and changes the transformed
        mask tokens and image embedding ("shared-pass" SAM output).
        """
        B = image_embeddings.shape[0]
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)  # [B,5,256]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe.expand(B, -1, -1, -1)
        b, c, h, w = src.shape
        hs, src_out = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_token0_out = hs[:, 1, :]              # single-mask slot 0

        src_2d = src_out.transpose(1, 2).view(b, c, h, w)
        upscaled = self.output_upscaling(src_2d)   # [B,32,256,256]
        kern = self.output_hypernetworks_mlps[0](mask_token0_out)
        c_up, h_up, w_up = upscaled.shape[1:]
        logits = (kern.unsqueeze(1) @
                  upscaled.view(b, c_up, h_up * w_up)).view(b, 1, h_up, w_up)
        iou_pred = self.iou_prediction_head(iou_token_out)[:, :1]
        return {"canonical_base_logits": logits, "iou_pred": iou_pred}

    # ------------------------------------------------------------------
    # Dual pass: residual over the frozen canonical Stage 2 base
    # ------------------------------------------------------------------

    def predict_dual_canonical(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        dino_l3_feat: torch.Tensor,
        adapter_feat: torch.Tensor,
        distilled_early_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Two decoder passes; the HQ token cannot perturb the base.

        pass 1 (no grad, no HQ token)  -> canonical_base_logits  (frozen base)
        pass 2 (with the HQ token)     -> hq_logits              (residual)
        corrected = canonical_base_logits.detach() + alpha * hq_logits, alpha = 1

        Design note: when the HQ token shares the decoder pass with the SAM
        mask token (as in ``MaskDecoderHQ``), it also changes the SAM-token
        output; with a detached base nothing penalises that change, and in our
        experiments the residual then had to undo it. A separate pass without
        any HQ parameter keeps the Stage 2 output fixed.

        The shared-pass SAM output (``sam_logits``) is returned for
        diagnostics only; it is not part of the corrected output.
        """
        canon = self.predict_canonical_stage2_base(
            image_embeddings=image_embeddings, image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings)
        base = canon["canonical_base_logits"].detach()

        hq_out = self.predict_with_hq(
            image_embeddings=image_embeddings, image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            dino_l3_feat=dino_l3_feat, adapter_feat=adapter_feat,
            distilled_early_feat=distilled_early_feat)

        alpha = self.residual_scale
        corrected = base + alpha * hq_out["hq_logits"]
        return {
            "canonical_base_logits": base,
            "sam_logits": hq_out["sam_logits"],        # diagnostic only
            "hq_logits": hq_out["hq_logits"],
            "corrected_logits": corrected,
            "iou_pred": canon["iou_pred"],
            "residual_scale": torch.tensor(float(alpha)),
        }

    # ------------------------------------------------------------------
    # Override forward so the parent signature still works (used if someone
    # calls the raw decoder without HQ context).  Raises informatively.
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "HQMaskDecoder.forward() is disabled; use predict_dual_canonical() "
            "(or predict_with_hq()) through FSDinoHQModel."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def is_hq_key(cls, key: str) -> bool:
        """True if ``key`` (relative to the decoder) belongs to an HQ module.

        Matches child-module keys (``adapter_proj.0.weight``) as well as a
        direct parameter named exactly like a module.
        """
        return any(key == m or key.startswith(m + ".")
                   for m in cls.HQ_MODULE_NAMES)

    def hq_state_dict(self) -> Dict[str, torch.Tensor]:
        """Only the HQ-specific state (stored separately in HQ checkpoints)."""
        return {k: v for k, v in self.state_dict().items() if self.is_hq_key(k)}

    def hq_named_parameters(self):
        """Yield (name, param) for HQ-only parameters."""
        for name, param in self.named_parameters():
            if self.is_hq_key(name):
                yield name, param
