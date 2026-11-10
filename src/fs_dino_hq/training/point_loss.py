"""Point-sampled BCE + Dice mask loss (the mask loss of the HQ-SAM trainer).

Adapted from HQ-SAM, ``train/utils/loss_mask.py``
(https://github.com/SysCV/sam-hq, commit e696978), distributed under the
Apache License, Version 2.0; that file follows Mask2Former and the PointRend
point sampling of Detectron2. See THIRD_PARTY_NOTICES.md.

Modifications: the random draws accept an explicit ``torch.Generator`` so the
sampling stream is separate from every other use of random numbers; the
functions are not wrapped in ``torch.jit.script``; ``num_masks`` is the batch
size of the logits. With ``generator=None`` the values equal the upstream
functions' for the same global random state.

The loss is evaluated at ``num_points`` locations per mask instead of on
every cell of the logit grid:

  * ``oversample_ratio * num_points`` candidate locations are drawn uniformly
    in [0, 1]^2; the ``importance_sample_ratio * num_points`` candidates whose
    bilinearly interpolated logit is closest to zero are kept (the least
    certain predictions, which lie mostly along mask boundaries), and the
    remaining points are drawn uniformly again;
  * logits and target are read at those locations with
    ``grid_sample(align_corners=False)``, so one normalised coordinate
    addresses the same position on both maps when both span the same canvas.
    The target may therefore have a higher resolution than the logits;
  * BCE is the mean over the points of a mask; Dice is
    ``1 - (2 * sum(p * t) + 1) / (sum(p) + sum(t) + 1)`` over the points of a
    mask; both are averaged over the masks of the batch.

Point selection runs without gradient; gradients flow through the sampled
logits only.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

NUM_POINTS = 112 * 112
OVERSAMPLE_RATIO = 3.0
IMPORTANCE_SAMPLE_RATIO = 0.75


def point_sample(values: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    """Bilinear samples of ``values`` [N,C,H,W] at ``point_coords`` [N,P,2] ((x, y) in [0, 1]^2) -> [N,C,P]."""
    return F.grid_sample(values, 2.0 * point_coords.unsqueeze(2) - 1.0, align_corners=False).squeeze(3)


def _rand(shape, device, generator: Optional[torch.Generator]) -> torch.Tensor:
    if generator is None:
        return torch.rand(*shape, device=device)
    return torch.rand(*shape, device=device, generator=generator)


def sample_uncertain_points(logits: torch.Tensor, num_points: int, oversample_ratio: float,
                            importance_sample_ratio: float,
                            generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Point coordinates [N, num_points, 2] for logits [N,1,H,W] (PointRend sampling)."""
    assert oversample_ratio >= 1 and 0 <= importance_sample_ratio <= 1
    n = logits.shape[0]
    num_candidates = int(num_points * oversample_ratio)
    candidates = _rand((n, num_candidates, 2), logits.device, generator)
    # Uncertainty of the interpolated logit at each candidate, not of the grid
    # cells: a point between cells with logits -1 and +1 is maximally uncertain.
    uncertainty = -torch.abs(point_sample(logits, candidates))
    num_uncertain = int(importance_sample_ratio * num_points)
    idx = torch.topk(uncertainty[:, 0, :], k=num_uncertain, dim=1)[1]
    idx += num_candidates * torch.arange(n, dtype=torch.long, device=logits.device)[:, None]
    points = candidates.view(-1, 2)[idx.view(-1), :].view(n, num_uncertain, 2)
    if num_points > num_uncertain:
        points = torch.cat([points, _rand((n, num_points - num_uncertain, 2), logits.device, generator)],
                           dim=1)
    return points


def point_sampled_loss(logits: torch.Tensor, target: torch.Tensor,
                       num_points: int = NUM_POINTS,
                       oversample_ratio: float = OVERSAMPLE_RATIO,
                       importance_sample_ratio: float = IMPORTANCE_SAMPLE_RATIO,
                       generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """(BCE, Dice) at sampled points. ``logits`` [N,1,h,w]; ``target`` [N,1,H,W] with values in [0, 1].

    ``target`` may have another resolution than ``logits``; both must cover the
    same canvas. The training loss is the sum of the two returned terms.
    """
    assert logits.dim() == 4 and logits.shape[1] == 1, logits.shape
    assert target.dim() == 4 and target.shape[:2] == logits.shape[:2], (logits.shape, target.shape)
    target = target.to(dtype=logits.dtype)
    n = logits.shape[0]
    with torch.no_grad():
        points = sample_uncertain_points(logits, num_points, oversample_ratio,
                                         importance_sample_ratio, generator)
        labels = point_sample(target, points).squeeze(1)
    point_logits = point_sample(logits, points).squeeze(1)
    bce = F.binary_cross_entropy_with_logits(point_logits, labels, reduction="none").mean(1).sum() / n
    p = point_logits.sigmoid().flatten(1)
    dice = (1 - (2 * (p * labels).sum(-1) + 1) / (p.sum(-1) + labels.sum(-1) + 1)).sum() / n
    return bce, dice
