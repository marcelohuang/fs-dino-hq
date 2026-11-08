"""K-shot prompt aggregation, shared by HQ training and inference so that both
aggregate supports identically.

Per support k:  Fs_k -> semantic_prompt_k                       [1, 1, 256]
                S_pcs_k = cosine(prototype_k, Fq)                [1, 1, 37, 37]
                corr_branch(Fs_k, Fq, S_pcs_k) -> dense_prompt_k [1, 256, 64, 64]
                                                  prior_k        [1, 1, 37, 37]
Aggregate:      semantic_prompt = mean_k semantic_prompt_k   (embedding space)
                dense_prompt    = mean_k dense_prompt_k      (embedding space)
                prior_mask      = mean_k sigmoid(prior_k)    (probability space)
                prior_logits    = mean_k prior_k
Decode once:    the mask decoder runs a single time on the averaged prompts.

Not averaged: the final-adapter image embedding, the early-adapter feature and
the HQ branch come from the query image only and are computed once. Final
mask probabilities are never averaged across supports ("prompt averaging",
not "prediction averaging"). Stage 2 training uses the same rule
(``fs_dino_hq.modeling.stage2.Stage2Model.forward``).

Decoder: two passes over the same prompts (``HQMaskDecoder.predict_dual_canonical``):

  canonical_base_logits  SAM decoder pass without the HQ token; depends only
                         on Stage 2 weights (the "Stage 2 base" output)
  hq_logits              HQ residual from a second pass that includes the HQ token
  corrected_logits       canonical_base_logits.detach() + alpha * hq_logits,
                         alpha = 1 (the model output)
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def compute_support_prompts(model, support_dino_k: torch.Tensor,
                            support_mask_k: torch.Tensor,
                            Fq_1: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-support prompts for one query, batched over the K supports.

    support_dino_k [K,3,H,W], support_mask_k [K,1,H,W], Fq_1 [1,768,37,37].
    """
    K = support_dino_k.shape[0]
    with torch.no_grad():
        Fs_k = model.last_extractor(support_dino_k)               # [K,768,37,37]
    mask_k = F.interpolate(support_mask_k.float(), size=Fs_k.shape[-2:],
                           mode="nearest")
    semantic_k = model.semantic_branch(Fs_k, mask_k)              # [K,1,256]
    Fq_k = Fq_1.expand(K, -1, -1, -1)
    _, S_pcs_k = model.pcs_branch(Fs_k, mask_k, Fq_k)             # [K,1,37,37]
    _, _, prior_logits_k, prior_mask_k, dense_k = \
        model.corr_branch(Fs_k, Fq_k, S_pcs_k)
    return {
        "semantic_k": semantic_k,          # [K,1,256]
        "dense_k": dense_k,                # [K,256,64,64]
        "prior_logits_k": prior_logits_k,  # [K,1,37,37]
        "prior_mask_k": prior_mask_k,      # [K,1,37,37]
        "S_pcs_k": S_pcs_k,                # [K,1,37,37]
    }


def aggregate_prompts(p: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Mean over the support axis (dimension 0) of the per-support prompts."""
    return {
        "semantic_prompt": p["semantic_k"].mean(dim=0, keepdim=True)[0:1],  # [1,1,256]
        "dense_prompt":    p["dense_k"].mean(dim=0, keepdim=True)[0:1],     # [1,256,64,64]
        "prior_logits":    p["prior_logits_k"].mean(dim=0, keepdim=True)[0:1],
        "prior_mask":      p["prior_mask_k"].mean(dim=0, keepdim=True)[0:1],
    }


def early_feature(model, Fq_l3: torch.Tensor) -> torch.Tensor:
    """Frozen distilled early-adapter feature at 64x64 (input of the HQ branch)."""
    if model.early_adapter is None:
        raise ValueError("the HQ branch needs model.early_adapter (distilled early adapter)")
    with torch.no_grad():
        ea = model.early_adapter(Fq_l3)
        if ea.shape[-2:] != (64, 64):
            ea = F.interpolate(ea, size=(64, 64), mode="bilinear", align_corners=False)
    return ea


def forward_kshot(model, batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    """K-shot forward pass of ``FSDinoHQModel``.

    batch:
      support_dino [B,K,3,518,518]   support_mask [B,K,1,518,518]
      query_dino   [B,3,518,518]     query_mask   [B,1,518,518] (optional)

    Returns canonical base, HQ residual and corrected logits (all [B,1,256,256]),
    the aggregated prior, and, when ``query_mask`` is given, the masks resized
    to the logit grids (nearest neighbour). ``final_logits`` is the logit map
    the HQ training loss is applied to (the corrected logits).
    """
    support_dino = batch["support_dino"].to(device)
    support_mask = batch["support_mask"].to(device)
    query_dino = batch["query_dino"].to(device)
    B, K = support_dino.shape[:2]
    assert support_mask.shape[:2] == (B, K), \
        f"support/mask K mismatch: {support_dino.shape} vs {support_mask.shape}"

    with torch.no_grad():
        Fq = model.last_extractor(query_dino)          # [B,768,37,37]
        Fq_l3 = model.layer3_extractor(query_dino)     # [B,768,37,37]

    adapter_out = model.adapter(Fq_l3)
    adapter_out_64 = F.interpolate(adapter_out, size=(64, 64),
                                   mode="bilinear", align_corners=False)
    distilled_early = early_feature(model, Fq_l3)
    image_pe = model.prompt_encoder.get_dense_pe()

    sam_l, hq_l, iou_l, prior_l, prior_m, can_l = [], [], [], [], [], []
    for b in range(B):
        prompts = aggregate_prompts(compute_support_prompts(
            model, support_dino[b], support_mask[b], Fq[b:b + 1]))
        hq_out = model.hq_mask_decoder.predict_dual_canonical(
            image_embeddings=adapter_out_64[b:b + 1],
            image_pe=image_pe,
            sparse_prompt_embeddings=prompts["semantic_prompt"],
            dense_prompt_embeddings=prompts["dense_prompt"],
            dino_l3_feat=Fq_l3[b:b + 1],
            adapter_feat=adapter_out_64[b:b + 1],
            distilled_early_feat=distilled_early[b:b + 1],
        )
        can_l.append(hq_out["canonical_base_logits"])
        sam_l.append(hq_out["sam_logits"])
        hq_l.append(hq_out["hq_logits"])
        iou_l.append(hq_out["iou_pred"])
        prior_l.append(prompts["prior_logits"])
        prior_m.append(prompts["prior_mask"])

    hq_logits = torch.cat(hq_l, dim=0)
    # residual over the frozen canonical base (HQ-token-free pass, detached)
    canonical_logits = torch.cat(can_l, dim=0)
    alpha = model.hq_mask_decoder.residual_scale
    corrected_logits = canonical_logits.detach() + alpha * hq_logits
    final_logits = corrected_logits
    prior_logits = torch.cat(prior_l, dim=0)

    out = {
        "canonical_base_logits": canonical_logits,
        "hq_logits": hq_logits,
        "corrected_logits": corrected_logits,
        "final_logits": final_logits,
        # SAM-token output of the second (HQ-token) pass; diagnostic only
        "shared_pass_sam_logits": torch.cat(sam_l, dim=0),
        "prior_logits": prior_logits,
        "prior_mask": torch.cat(prior_m, dim=0),
        "iou_pred": torch.cat(iou_l, dim=0),
        "adapter_embedding": adapter_out_64,
    }
    if "query_mask" in batch:
        qm = batch["query_mask"].to(device)
        out["query_mask_prior"] = F.interpolate(
            qm.float(), size=prior_logits.shape[-2:], mode="nearest")
        out["query_mask_final"] = F.interpolate(
            qm.float(), size=final_logits.shape[-2:], mode="nearest")
    return out
