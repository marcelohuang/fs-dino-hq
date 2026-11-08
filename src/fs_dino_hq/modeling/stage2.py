"""Stage 2 training model: final adapter + MVPG + SAM mask decoder.

Follows Stage 2 of FS-DINO (meta visual prompt generator trained with the SAM
mask decoder on episodes); see docs/implementation_notes.md for where this
implementation differs from the paper.

  frozen     DINOv2 (last-layer tokens for MVPG, block-3 tokens for the adapter),
             SAM prompt encoder (dense positional encoding only)
  trainable  BottleneckAdapter (initialised from a distilled final adapter),
             SemanticAwarePromptBranch, Correlation4DBranchV2,
             SAM mask decoder (initialised from the SAM ViT-H checkpoint)
  unused     SAM image encoder (never part of this module or its checkpoints)
  loss       dice+BCE on the prior logits (37x37) + dice+BCE on the decoder
             logits (256x256), query mask resized with NEAREST to each grid
  K-shot     K > 1 supports per query are aggregated at prompt level exactly as
             fs_dino_hq.modeling.kshot does for HQ training and inference

State-dict layout: ``adapter.*``, ``semantic_branch.*``, ``pcs_branch.*``,
``corr_branch.*``, ``sam_full.prompt_encoder.*``, ``sam_full.mask_decoder.*``;
``fs_dino_hq.training.checkpoints.load_stage2_strict`` maps it onto
``FSDinoHQModel``. The two DINOv2 extractors are excluded from saved
checkpoints (``stage2_state_dict``).

The adapter is built with the distillation student's transformer convention
(``norm_first=True``, dropout 0) so the distilled function is carried over
exactly; the convention is recorded in every checkpoint (``adapter_norm_first``).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training.losses import seg_loss
from .correlation import Correlation4DBranchV2
from .model import (BottleneckAdapter, DINOv2LastLayerExtractor,
                    DINOv2Layer3Extractor, PrototypeCosineBranch,
                    SemanticAwarePromptBranch)

DINO_PREFIXES = ("last_extractor.", "layer3_extractor.")
TRAINABLE_TOP = ("adapter", "semantic_branch", "corr_branch", "sam_full.mask_decoder")
FROZEN_TOP = ("last_extractor", "layer3_extractor", "sam_full.prompt_encoder")


class SamDecoderHead(nn.Module):
    """``sam_full`` container holding only the SAM parts Stage 2 uses."""

    def __init__(self, sam_full: nn.Module) -> None:
        super().__init__()
        self.prompt_encoder = sam_full.prompt_encoder
        self.mask_decoder = sam_full.mask_decoder


class Stage2Model(nn.Module):
    def __init__(self, dino_model: nn.Module, sam_full: nn.Module, feat_hw: int = 37,
                 dino_ch: int = 768, transformer_dim: int = 256,
                 adapter_norm_first: bool = True, adapter_dropout: float = 0.0,
                 corr_arch: str = "v2_reduced") -> None:
        super().__init__()
        self.corr_arch = corr_arch
        self.last_extractor = DINOv2LastLayerExtractor(dino_model)
        self.layer3_extractor = DINOv2Layer3Extractor(dino_model, block_index=2)
        self.adapter = BottleneckAdapter(in_ch=dino_ch, norm_first=adapter_norm_first,
                                         dropout=adapter_dropout)
        self.semantic_branch = SemanticAwarePromptBranch(in_ch=dino_ch, out_ch=transformer_dim)
        self.pcs_branch = PrototypeCosineBranch()
        if corr_arch != "v2_reduced":
            raise ValueError(f"unsupported corr_arch {corr_arch!r}; this package builds 'v2_reduced'")
        self.corr_branch = Correlation4DBranchV2(feat_hw=feat_hw, out_prompt_ch=transformer_dim)
        self.sam_full = SamDecoderHead(sam_full)
        self.feat_hw = feat_hw

    # ── forward: K supports per query, prompt-level aggregation ──────────────
    # K = 1 is the one-shot computation. For K > 1 the per-support
    # prompts are computed exactly as fs_dino_hq.modeling.kshot does at inference
    # (semantic prompt, prototype-cosine map and correlation branch per support,
    # then the MEAN over supports of the semantic prompt, the dense prompt and
    # the prior logits; ONE decoder pass on the averaged prompts). The B*K
    # support images go through the branches in one batched call, so K = 1
    # reduces to the original path bit-for-bit (BatchNorm statistics over the
    # same B samples) and K > 1 is differentiable through every support.
    # Aggregation level: PROMPT (embedding space), not feature, prototype,
    # correlation-volume or prediction level. The query enters the prompts only
    # through S_pcs (cosine to each support prototype) and the correlation
    # branch, as in 1-shot; the SAM image embedding uses the query alone.
    def forward(self, batch: Dict[str, torch.Tensor], support_feat_grad: bool = False) -> Dict[str, torch.Tensor]:
        """support_feat_grad=True (diagnostics only) makes the per-support DINO
        features a leaf that requires grad and returns it as out["Fs_k"]
        ([B*K,768,37,37]) so per-support gradient flow can be measured."""
        device = next(self.adapter.parameters()).device
        support_dino = batch["support_dino"].to(device)
        support_mask = batch["support_mask"].to(device)
        if support_dino.dim() == 4:                     # [B,3,S,S] -> K = 1
            support_dino, support_mask = support_dino[:, None], support_mask[:, None]
        query_dino = batch["query_dino"].to(device)
        B, K = support_dino.shape[:2]
        assert support_mask.shape[:2] == (B, K), (support_dino.shape, support_mask.shape)

        with torch.no_grad():
            Fs = self.last_extractor(support_dino.flatten(0, 1))   # [B*K,768,37,37]
            Fq = self.last_extractor(query_dino)                   # [B,768,37,37]
            Fq_l3 = self.layer3_extractor(query_dino)

        if support_feat_grad:
            Fs = Fs.detach().requires_grad_(True)
        mask_feat = F.interpolate(support_mask.flatten(0, 1).float(), size=Fs.shape[-2:], mode="nearest")
        Fq_k = Fq.repeat_interleave(K, dim=0) if K > 1 else Fq            # [B*K,768,37,37]
        semantic_k = self.semantic_branch(Fs, mask_feat)                 # [B*K,1,256]
        _, S_pcs_k = self.pcs_branch(Fs, mask_feat, Fq_k)                # [B*K,1,37,37]
        corr_map_k, fused_map_k, prior_logits_k, prior_mask_k, dense_k = \
            self.corr_branch(Fs, Fq_k, S_pcs_k)
        if K == 1:
            semantic_prompt, S_pcs, prior_logits, prior_mask, dense_prompt = \
                semantic_k, S_pcs_k, prior_logits_k, prior_mask_k, dense_k
            corr_map, fused_map = corr_map_k, fused_map_k
        else:   # mean over supports, as kshot.aggregate_prompts
            semantic_prompt = semantic_k.view(B, K, *semantic_k.shape[1:]).mean(1)
            dense_prompt = dense_k.view(B, K, *dense_k.shape[1:]).mean(1)
            prior_logits = prior_logits_k.view(B, K, *prior_logits_k.shape[1:]).mean(1)
            prior_mask = prior_mask_k.view(B, K, *prior_mask_k.shape[1:]).mean(1)
            S_pcs = S_pcs_k.view(B, K, *S_pcs_k.shape[1:]).mean(1)
            corr_map = corr_map_k.view(B, K, *corr_map_k.shape[1:]).mean(1)
            fused_map = fused_map_k.view(B, K, *fused_map_k.shape[1:]).mean(1)

        image_embedding = F.interpolate(self.adapter(Fq_l3), size=(64, 64),
                                        mode="bilinear", align_corners=False)
        image_pe = self.sam_full.prompt_encoder.get_dense_pe()
        final = []
        for i in range(image_embedding.shape[0]):
            low_res_masks, _ = self.sam_full.mask_decoder(
                image_embeddings=image_embedding[i:i + 1], image_pe=image_pe,
                sparse_prompt_embeddings=semantic_prompt[i:i + 1],
                dense_prompt_embeddings=dense_prompt[i:i + 1], multimask_output=False)
            final.append(low_res_masks)
        final_logits = torch.cat(final, dim=0)                           # [B,1,256,256]

        out = {"prior_logits": prior_logits, "prior_mask": prior_mask,
               "final_logits": final_logits, "dense_prompt": dense_prompt,
               "semantic_prompt": semantic_prompt, "S_pcs": S_pcs,
               "corr_map": corr_map, "fused_map": fused_map,
               "image_embedding": image_embedding, "n_shot": K}
        if support_feat_grad:
            out["Fs_k"] = Fs
        if "query_mask" in batch:
            qm = batch["query_mask"].to(device).float()
            out["query_mask_prior"] = F.interpolate(qm, size=prior_logits.shape[-2:], mode="nearest")
            out["query_mask_final"] = F.interpolate(qm, size=final_logits.shape[-2:], mode="nearest")
        return out

    # ── checkpoint view ──────────────────────────────────────────────────────
    def stage2_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict without the frozen DINOv2 copies (reloaded from the DINOv2 checkpoint)."""
        return {k: v.detach().cpu() for k, v in self.state_dict().items()
                if not k.startswith(DINO_PREFIXES)}


def stage2_loss(out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(total, prior, final): ``loss = seg(prior) + seg(final)``, seg = dice + BCE."""
    lp = seg_loss(out["prior_logits"], out["query_mask_prior"])
    lf = seg_loss(out["final_logits"], out["query_mask_final"])
    return lp + lf, lp, lf


@torch.no_grad()
def batch_iou(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Mean per-sample IoU of sigmoid(logits) > 0.5 against a {0,1} target."""
    pred = (logits > 0).float().flatten(1)
    t = target.flatten(1)
    inter = (pred * t).sum(1)
    union = pred.sum(1) + t.sum(1) - inter
    return float(((inter + eps) / (union + eps)).mean())


def apply_stage2_freeze(model: Stage2Model) -> None:
    """DINOv2 + prompt encoder frozen (eval mode); adapter, semantic branch,
    correlation branch and SAM mask decoder trainable. Re-applied after every
    ``model.train()``."""
    for name in ("last_extractor", "layer3_extractor"):
        m = getattr(model, name); m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    model.sam_full.prompt_encoder.eval()
    for p in model.sam_full.prompt_encoder.parameters():
        p.requires_grad_(False)
    for name in ("adapter", "semantic_branch", "corr_branch"):
        m = getattr(model, name); m.train()
        for p in m.parameters():
            p.requires_grad_(True)
    model.sam_full.mask_decoder.train()
    for p in model.sam_full.mask_decoder.parameters():
        p.requires_grad_(True)


def stage2_param_audit(model: Stage2Model) -> Dict:
    """Trainable/frozen parameter counts by top-level module (DINO counted once)."""
    by: Dict[str, Dict[str, int]] = {}
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        parts = name.split(".")
        top = ".".join(parts[:2]) if parts[0] == "sam_full" else parts[0]
        rec = by.setdefault(top, {"trainable": 0, "frozen": 0})
        rec["trainable" if p.requires_grad else "frozen"] += p.numel()
    trainable_top = sorted(k for k, v in by.items() if v["trainable"] > 0)
    return {"by_module": by,
            "trainable_params": sum(v["trainable"] for v in by.values()),
            "frozen_params": sum(v["frozen"] for v in by.values()),
            "trainable_top_modules": trainable_top,
            "policy": "DINOv2 + SAM prompt encoder frozen; adapter, semantic_branch, "
                      "corr_branch, SAM mask decoder trainable"}


def trainable_parameters(model: Stage2Model):
    seen, out = set(), []
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); out.append(p)
    return out
