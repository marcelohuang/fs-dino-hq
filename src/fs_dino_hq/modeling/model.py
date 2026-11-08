"""FS-DINO-HQ model: DINOv2 features, meta visual prompt generator (MVPG),
SAM mask decoder and the high-quality (HQ) residual branch.

Data flow for one query and K annotated supports (all images letterboxed to
518x518, see ``fs_dino_hq.data.letterbox``):

  DINOv2 ViT-B/14 (frozen)    last-layer tokens  Fs_k, Fq   [768, 37, 37]
                              block-3 tokens     Fq_l3      [768, 37, 37]
  MVPG (per support k)        semantic prompt    [1, 256]
                              prototype-cosine   S_pcs      [1, 37, 37]
                              4D correlation  -> prior [1, 37, 37] and
                                                 dense prompt [256, 64, 64]
  K-shot aggregation          mean over supports of the prompts (see
                              ``fs_dino_hq.modeling.kshot``)
  final adapter               Fq_l3 -> SAM-like image embedding [256, 64, 64]
  SAM mask decoder            canonical base logits [1, 256, 256]
  early adapter (frozen)      Fq_l3 -> [256, 64, 64], input of the HQ branch
  HQ branch                   residual logits; corrected = base + residual

State-dict layout (unchanged from the training code that produced the
checkpoints described in docs/results.md):

    last_extractor.*, layer3_extractor.*   DINOv2 (frozen; not stored in checkpoints)
    adapter.*                              final adapter (Stage 2)
    semantic_branch.*, pcs_branch.*, corr_branch.*   MVPG (Stage 2)
    prompt_encoder.*                       SAM prompt encoder (frozen)
    hq_mask_decoder.*                      SAM decoder (Stage 2) + HQ modules
    early_adapter.*                        distilled early adapter (frozen)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .correlation import Correlation4DBranchV2
from .hq_decoder import HQMaskDecoder


# ─────────────────────────────────────────────────────────────────────────────
# Components
# ─────────────────────────────────────────────────────────────────────────────

class DINOv2LastLayerExtractor(nn.Module):
    """Normalised last-layer patch tokens of DINOv2 as a [B, C, h, w] map."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.forward_features(x)
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            patch_tokens = out["x_norm_patchtokens"]
        elif isinstance(out, dict) and "x_prenorm" in out:
            patch_tokens = out["x_prenorm"][:, 1:, :]
        else:
            raise RuntimeError("Cannot find last-layer patch tokens from DINO output.")
        b, n, c = patch_tokens.shape
        hw = int(math.sqrt(n))
        return patch_tokens.transpose(1, 2).reshape(b, c, hw, hw)


class DINOv2Layer3Extractor(nn.Module):
    """Raw output tokens of DINOv2 block ``block_index`` (default 2, the third
    block) as a [B, C, h, w] map, captured with a forward hook."""

    def __init__(self, model: nn.Module, block_index: int = 2) -> None:
        super().__init__()
        self.model = model
        self.block_index = block_index
        self._cached_tokens: Optional[torch.Tensor] = None
        self.hook_handle = self.model.blocks[block_index].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output) -> None:
        self._cached_tokens = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._cached_tokens = None
        _ = self.model.forward_features(x)
        tokens = self._cached_tokens
        if tokens is None:
            raise RuntimeError("Failed to capture DINO layer-3 tokens")
        b, n, c = tokens.shape
        if int(math.sqrt(n - 1)) ** 2 == (n - 1):
            tokens = tokens[:, 1:, :]
            n -= 1
        hw = int(math.sqrt(n))
        return tokens.transpose(1, 2).reshape(b, c, hw, hw)


class PrototypeCosineBranch(nn.Module):
    """Masked-average support prototype and its cosine map over the query."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, support_feat: torch.Tensor, support_mask: torch.Tensor,
                query_feat: torch.Tensor):
        masked_support = support_feat * support_mask
        denom = support_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        proto = masked_support.sum(dim=(2, 3), keepdim=True) / denom
        proto_n = F.normalize(proto, dim=1)
        query_n = F.normalize(query_feat, dim=1)
        sim = (proto_n * query_n).sum(dim=1, keepdim=True)
        return proto, sim


class SemanticAwarePromptBranch(nn.Module):
    """Sparse (semantic) prompt: masked, convolved and pooled support features."""

    def __init__(self, in_ch: int = 768, out_ch: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU())

    def forward(self, support_feat: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
        x = support_feat * support_mask
        x = self.conv1(x)
        x = self.conv2(x)
        denom = support_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        pooled = x.sum(dim=(2, 3), keepdim=True) / denom
        pooled = pooled.flatten(1)
        pooled = F.normalize(pooled, dim=1)
        return pooled.unsqueeze(1)   # [B, 1, 256]


class Correlation4DBranchV1(nn.Module):
    """Earlier correlation branch (both token grids downsampled to ``ds_hw``).

    No checkpoint produced by this package uses it. ``FSDinoHQModel`` still
    constructs it before the HQ decoder and then replaces it with
    ``Correlation4DBranchV2``: the construction order fixes how the global
    random generator is consumed, so a seeded HQ training run starts from the
    same randomly initialised HQ modules as the runs in docs/results.md (their
    ``initial_hq_state_sha256`` is recorded in each run's ``config.json``).
    """

    def __init__(self, feat_hw: int, ds_hw: int, out_prompt_ch: int = 256) -> None:
        super().__init__()
        self.feat_hw = feat_hw
        self.ds_hw = ds_hw
        n = ds_hw * ds_hw
        self.depthwise = nn.Conv2d(n, n, 3, padding=1, groups=n, bias=False)
        self.pointwise = nn.Conv2d(n, n, 1, bias=False)
        self.prior_head = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1, bias=True))
        self.dense_prompt_head = nn.Sequential(
            nn.Conv2d(1, out_prompt_ch, 1, bias=False), nn.BatchNorm2d(out_prompt_ch), nn.GELU())

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Correlation4DBranchV1 is only constructed for "
                                  "initialisation-order compatibility")


class BottleneckAdapter(nn.Module):
    """Final adapter: DINOv2 block-3 tokens -> SAM-like image embedding.

    Same parameter names and shapes as the distillation student
    (``fs_dino_hq.modeling.distill_adapter.BottleneckAdapter``) once the
    student's ``.layer.`` infix is removed.

    ``norm_first`` / ``dropout`` select the convention of the two transformer
    layers. Stage 2 and HQ checkpoints trained by this package use the
    distillation student's convention (``norm_first=True``, dropout 0) and
    record it as ``adapter_norm_first``; with identical weights the two
    conventions compute different functions. The constructor default
    (post-norm, dropout 0.1) is PyTorch's ``TransformerEncoderLayer`` default
    and is what a checkpoint without the flag was trained with.
    """

    def __init__(self, in_ch: int = 768, mid_ch: int = 256,
                 low_ch: int = 128, num_heads: int = 8,
                 norm_first: bool = False, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_first = bool(norm_first)
        self.dropout_p = float(dropout)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False), nn.BatchNorm2d(mid_ch), nn.GELU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_ch, low_ch, 1, bias=False), nn.BatchNorm2d(low_ch), nn.GELU())
        self.attn_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=low_ch, nhead=num_heads,
                                       dim_feedforward=512, batch_first=True,
                                       dropout=self.dropout_p,
                                       norm_first=self.norm_first)
            for _ in range(2)
        ])
        self.proj_out = nn.Conv2d(low_ch, 256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        for blk in self.attn_blocks:
            x = blk(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return self.proj_out(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class FSDinoHQModel(nn.Module):
    """DINOv2 + MVPG + SAM decoder + HQ branch (see the module docstring).

    ``sam_full`` only needs ``prompt_encoder`` and ``mask_decoder`` attributes
    (a full SAM model, or ``fs_dino_hq.modeling.backbones.build_sam_prompt_decoder()``).
    The distilled early adapter is attached after construction
    (``model.early_adapter = ...``); it is frozen and only feeds the HQ branch.

    The K-shot forward pass shared by HQ training and inference is
    ``fs_dino_hq.modeling.kshot.forward_kshot``; ``forward`` delegates to it.
    """

    def __init__(
        self,
        dino_model: nn.Module,
        sam_full: nn.Module,
        corr_arch: str = "v2_reduced",
        feat_hw: int = 37,
        ds_hw: int = 18,
        dino_ch: int = 768,
        transformer_dim: int = 256,
        adapter_norm_first: bool = False,
        adapter_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if corr_arch != "v2_reduced":
            raise ValueError(f"unsupported corr_arch {corr_arch!r}; this package builds 'v2_reduced'")

        # Frozen DINOv2 feature extractors (one shared backbone)
        self.last_extractor = DINOv2LastLayerExtractor(dino_model)
        self.layer3_extractor = DINOv2Layer3Extractor(dino_model, block_index=2)

        # Final adapter (convention recorded in the Stage 2 checkpoint)
        self.adapter = BottleneckAdapter(in_ch=dino_ch, norm_first=adapter_norm_first,
                                         dropout=adapter_dropout)

        # MVPG
        self.semantic_branch = SemanticAwarePromptBranch(in_ch=dino_ch, out_ch=transformer_dim)
        self.pcs_branch = PrototypeCosineBranch()
        # Replaced by Correlation4DBranchV2 below; see Correlation4DBranchV1.
        self.corr_branch = Correlation4DBranchV1(feat_hw=feat_hw, ds_hw=ds_hw,
                                                 out_prompt_ch=transformer_dim)

        # SAM prompt encoder (dense positional encoding only; frozen)
        self.prompt_encoder = sam_full.prompt_encoder

        # HQ mask decoder: SAM mask decoder + HQ modules
        orig_dec = sam_full.mask_decoder
        self.hq_mask_decoder = HQMaskDecoder(
            transformer_dim=orig_dec.transformer_dim,
            transformer=orig_dec.transformer,
            num_multimask_outputs=orig_dec.num_multimask_outputs,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            dino_l3_in_ch=dino_ch,
        )
        self.hq_mask_decoder.load_state_dict(orig_dec.state_dict(), strict=False)

        # Distilled early adapter, attached by the caller (registered as a
        # submodule on assignment).
        self.early_adapter = None

        self.corr_branch = Correlation4DBranchV2(feat_hw=feat_hw)
        self.corr_arch = corr_arch

    def forward(self, batch):
        """K-shot episode batch -> output dict (see ``kshot.forward_kshot``)."""
        from .kshot import forward_kshot
        device = next(self.hq_mask_decoder.parameters()).device
        return forward_kshot(self, batch, device)
