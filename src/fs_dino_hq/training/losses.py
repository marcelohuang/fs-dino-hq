"""Segmentation losses.

All logits are [B,1,H,W]; targets are float {0,1} masks of the same shape.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return 1 - ((2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def bce_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


def seg_loss(logits, target):
    return dice_loss(logits, target) + bce_loss(logits, target)


def hq_loss(out: Dict) -> torch.Tensor:
    """HQ training objective: dice + BCE on the corrected logits (256 x 256 grid).

    Equal to the weighted objective w_final * seg(final) with w_final = 1 and
    the prior and HQ-only terms disabled (weights 0), the setting of every
    HQ run reported in docs/results.md.
    """
    return 1.0 * seg_loss(out["final_logits"], out["query_mask_final"])
