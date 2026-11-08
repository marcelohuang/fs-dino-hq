"""Four-dimensional correlation branch of the meta visual prompt generator.

``Correlation4DBranchV2`` computes the cosine-similarity volume between the
37x37 DINOv2 last-layer token grids of a support and a query image
(37^4 entries), reduces it with learned convolutions to a 19x19 x 19x19
volume, fuses it with the prototype-cosine map and produces

  * a coarse prior (``prior_logits``, 37x37, query coordinates), and
  * a dense prompt embedding for the SAM mask decoder (256x64x64).

Parameter names and shapes are those of the Stage 2 checkpoints, which load
strictly. Axis bookkeeping, kept as trained: ``reduce_conv1`` and
``reduce_conv2`` mix the query-token axis with learned 1x1 / 3x3 kernels, and
the resulting 361 channels are later reshaped into a 19x19 "query grid". That
grid is therefore a learned ordering rather than a geometric downsampling of
the query image; only the supervision of the prior and of the final mask (in
true query coordinates, after upsampling) ties it to image positions. The
support axes remain true (strided) image coordinates throughout, and the final
mean removes exactly the support dimensions.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Correlation4DBranchV2(nn.Module):
    """4D correlation with learned query-token reduction (``v2_reduced``)."""

    def __init__(self, feat_hw: int = 37, out_prompt_ch: int = 256) -> None:
        super().__init__()
        self.feat_hw = feat_hw                       # Hq = Wq = Hs = Ws = 37
        self.reduced_hw = (feat_hw + 1) // 2         # 19
        self.in_tokens = feat_hw * feat_hw           # 1369  (query tokens)
        self.reduced_tokens = self.reduced_hw * self.reduced_hw  # 361

        # 1×1 over the SUPPORT plane; mixes the QUERY-token channel axis
        # 1369 → 512 (learned, non-geometric).
        self.reduce_conv1 = nn.Sequential(
            nn.Conv2d(self.in_tokens, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )
        # 3×3 stride-2 over the SUPPORT plane (37→19, true coords) while mixing
        # channels 512 → 361 (the learned pseudo query grid).
        self.reduce_conv2 = nn.Sequential(
            nn.Conv2d(512, self.reduced_tokens, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(self.reduced_tokens),
            nn.GELU(),
        )
        # Depthwise: one 3×3 kernel per (pseudo) query token over the SUPPORT plane.
        self.depthwise = nn.Conv2d(self.reduced_tokens, self.reduced_tokens,
                                   kernel_size=3, padding=1,
                                   groups=self.reduced_tokens, bias=False)
        # Pointwise: 1×1 mixing of SUPPORT tokens at each (pseudo) query location.
        self.pointwise = nn.Conv2d(self.reduced_tokens, self.reduced_tokens,
                                   kernel_size=1, bias=False)

        self.prior_head = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=True),
        )
        self.dense_prompt_head = nn.Sequential(
            nn.Conv2d(1, out_prompt_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_prompt_ch),
            nn.GELU(),
        )

    def forward(self, support_feat: torch.Tensor, query_feat: torch.Tensor,
                S_pcs: torch.Tensor):
        """
        support_feat [B, C, Hs0, Ws0], query_feat [B, C, Hq0, Wq0] (DINO last),
        S_pcs [B, 1, Hq0, Wq0] prototype-cosine map in QUERY coordinates.
        Returns (corr_map, fused_map, prior_logits, prior_mask, dense_prompt)
        with prior_logits [B,1,feat_hw,feat_hw] in query coords, dense_prompt
        [B,256,64,64].
        """
        fh = self.feat_hw
        Fs = F.interpolate(support_feat, size=(fh, fh), mode="bilinear",
                           align_corners=False)          # [B, C, Hs, Ws]
        Fq = F.interpolate(query_feat, size=(fh, fh), mode="bilinear",
                           align_corners=False)          # [B, C, Hq, Wq]
        Fs = F.normalize(Fs, dim=1)
        Fq = F.normalize(Fq, dim=1)

        B, C, Hq, Wq = Fq.shape
        Hs, Ws = Fs.shape[-2:]
        assert (Hq, Wq) == (fh, fh) and (Hs, Ws) == (fh, fh)

        # 4D cosine similarity. "hw" indexes QUERY, "ij" indexes SUPPORT.
        corr4d = torch.einsum("bchw,bcij->bhwij", Fq, Fs)   # [B,Hq,Wq,Hs,Ws]
        assert corr4d.shape == (B, Hq, Wq, Hs, Ws)

        # Fold query tokens into channels; spatial dims = TRUE support coords.
        x = corr4d.reshape(B, Hq * Wq, Hs, Ws)              # [B,1369,37,37]
        x = self.reduce_conv1(x)                            # [B, 512,37,37]
        x = self.reduce_conv2(x)                            # [B, 361,19,19]

        Hq_r = Wq_r = self.reduced_hw                       # pseudo query grid
        Hs_r = Ws_r = self.reduced_hw                       # true strided support
        assert x.shape == (B, Hq_r * Wq_r, Hs_r, Ws_r)

        # View 1: channels = pseudo-query tokens, spatial = support.
        view_q_chan = x                                     # [B,361,19,19]
        # View 2: swap → channels = support tokens, spatial = pseudo-query grid.
        view_s_chan = (x.reshape(B, Hq_r, Wq_r, Hs_r, Ws_r)
                        .permute(0, 3, 4, 1, 2).contiguous()
                        .reshape(B, Hs_r * Ws_r, Hq_r, Wq_r))  # [B,361,19,19]

        dw_out = self.depthwise(view_q_chan)   # per-query-token conv over SUPPORT plane
        pw_out = self.pointwise(view_s_chan)   # support-token mixing per query location

        # Recompose both views into [B, Hq_r, Wq_r, Hs_r, Ws_r].
        rec_dw = dw_out.reshape(B, Hq_r, Wq_r, Hs_r, Ws_r)
        rec_pw = (pw_out.reshape(B, Hs_r, Ws_r, Hq_r, Wq_r)
                        .permute(0, 3, 4, 1, 2))
        fused4d = rec_dw + rec_pw                           # [B,Hq_r,Wq_r,Hs_r,Ws_r]
        assert fused4d.shape == (B, Hq_r, Wq_r, Hs_r, Ws_r)

        # Average over the SUPPORT dimensions only → query-coordinate map.
        corr_map = fused4d.mean(dim=(-1, -2)).unsqueeze(1)  # [B,1,Hq_r,Wq_r]
        assert corr_map.shape == (B, 1, Hq_r, Wq_r)

        S_pcs_ds = F.interpolate(S_pcs, size=(Hq_r, Wq_r), mode="bilinear",
                                 align_corners=False)
        fused_map = corr_map + S_pcs_ds
        fused_map = (fused_map - fused_map.amin(dim=(-2, -1), keepdim=True)) / (
            fused_map.amax(dim=(-2, -1), keepdim=True)
            - fused_map.amin(dim=(-2, -1), keepdim=True) + 1e-6)

        prior_logits = F.interpolate(self.prior_head(fused_map),
                                     size=(fh, fh), mode="bilinear",
                                     align_corners=False)   # [B,1,37,37]
        prior_mask = torch.sigmoid(prior_logits)
        dense_prompt = F.interpolate(self.dense_prompt_head(fused_map),
                                     size=(64, 64), mode="bilinear",
                                     align_corners=False)   # [B,256,64,64]
        return corr_map, fused_map, prior_logits, prior_mask, dense_prompt


def detect_corr_arch(stage2_sd: Dict, prefix: str = "corr_branch.") -> Tuple[str, int]:
    """Identify the correlation branch stored in a state dict from tensor shapes.

    Returns ``("v2_reduced", feat_hw)`` for the branch implemented here. Other
    layouts found in older checkpoints (a branch without the learned
    reduction, or a depthwise support reduction) are reported by name so the
    caller can refuse them; this package only builds ``v2_reduced``.
    """
    keys = [k for k in stage2_sd if k.startswith(prefix)]
    if not keys:
        raise KeyError(f"no '{prefix}*' keys in checkpoint state dict")
    if any(k.startswith(prefix + "reduce_support.") for k in keys):
        return "v3_geometric", 37
    if any(k.startswith(prefix + "reduce_conv1.") for k in keys):
        # reduce_conv1.0.weight: [512, feat_hw**2, 1, 1]
        in_tokens = int(stage2_sd[prefix + "reduce_conv1.0.weight"].shape[1])
        feat_hw = int(round(math.sqrt(in_tokens)))
        assert feat_hw * feat_hw == in_tokens, \
            f"reduce_conv1 in_channels {in_tokens} is not a square grid"
        return "v2_reduced", feat_hw
    n_tokens = int(stage2_sd[prefix + "depthwise.weight"].shape[0])
    return "v1_downsampled", int(round(math.sqrt(n_tokens)))
