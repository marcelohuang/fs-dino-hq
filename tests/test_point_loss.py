"""Point-sampled BCE + Dice loss: sampling geometry, selection of uncertain
points, reductions, random-number isolation and the high-resolution target."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from fs_dino_hq.data.letterbox import letterbox_mask
from fs_dino_hq.training.point_loss import (NUM_POINTS, point_sample, point_sampled_loss,
                                            sample_uncertain_points)


def test_defaults_are_the_hq_sam_settings():
    assert NUM_POINTS == 112 * 112 == 12544
    logits = torch.randn(2, 1, 32, 32)
    pts = sample_uncertain_points(logits, NUM_POINTS, 3.0, 0.75, torch.Generator().manual_seed(0))
    assert pts.shape == (2, NUM_POINTS, 2) and 0 <= float(pts.min()) and float(pts.max()) <= 1


def test_point_sample_reads_cell_centres_in_normalised_coordinates():
    # value = column index; the centre of column j is at x = (j + 0.5) / W on any grid size
    for w in (8, 32):
        ramp = torch.arange(w, dtype=torch.float32).expand(1, 1, w, w).contiguous()
        x = (torch.tensor([2, 5]) + 0.5) / w
        pts = torch.stack([x, torch.full_like(x, 0.5)], dim=-1)[None]          # (x, y)
        assert torch.allclose(point_sample(ramp, pts)[0, 0], torch.tensor([2.0, 5.0]), atol=1e-5)
        between = torch.tensor([[[3.0 / w, 0.5]]])                              # edge of columns 2 | 3
        assert torch.allclose(point_sample(ramp, between)[0, 0], torch.tensor([2.5]), atol=1e-5)


def test_one_coordinate_addresses_the_same_position_on_logits_and_a_finer_target():
    # left half foreground on a 16 grid and on a 64 grid: labels agree away from the mask edge
    # and from the canvas border (within half a cell of the border, grid_sample blends with
    # its zero padding, as in the upstream loss)
    coarse, fine = torch.zeros(1, 1, 16, 16), torch.zeros(1, 1, 64, 64)
    coarse[..., :8] = 1
    fine[..., :32] = 1
    pts = torch.rand(1, 4000, 2, generator=torch.Generator().manual_seed(1))
    a, b = point_sample(coarse, pts)[0, 0], point_sample(fine, pts)[0, 0]
    half = 0.5 / 16
    inside = ((pts[0] > half) & (pts[0] < 1 - half)).all(dim=1)
    far = inside & ((pts[0, :, 0] - 0.5).abs() > half)
    assert int(far.sum()) > 3000
    assert torch.allclose(a[far], b[far], atol=1e-6) and set(a[far].round().tolist()) == {0.0, 1.0}
    edge = pts[0, :, 0] < half / 2                              # blended with the zero padding
    assert float(a[edge].max()) < 1.0


def test_most_points_fall_where_the_prediction_is_uncertain():
    logits = torch.full((1, 1, 64, 64), 8.0)
    logits[..., 28:36] = 0.0                       # uncertain band: 1/8 of the width
    pts = sample_uncertain_points(logits, 4096, 3.0, 0.75, torch.Generator().manual_seed(2))
    in_band = ((pts[0, :, 0] > 28 / 64) & (pts[0, :, 0] < 36 / 64)).float().mean()
    # 3 x 4096 uniform candidates put ~1/8 of them (1536) in the band; all are kept by the
    # 3072 importance points, plus 1/8 of the 1024 uniform points
    assert 0.37 < float(in_band) < 0.45


def test_loss_values_match_a_direct_evaluation_at_the_sampled_points():
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(2, 1, 24, 24, generator=g)
    target = (torch.rand(2, 1, 96, 96, generator=g) > 0.6).float()
    bce, dice = point_sampled_loss(logits, target, num_points=500, generator=torch.Generator().manual_seed(4))
    pts = sample_uncertain_points(logits, 500, 3.0, 0.75, torch.Generator().manual_seed(4))
    lg = F.grid_sample(logits, 2 * pts[:, :, None] - 1, align_corners=False)[:, 0, :, 0]
    lb = F.grid_sample(target, 2 * pts[:, :, None] - 1, align_corners=False)[:, 0, :, 0]
    p = torch.sigmoid(lg)
    ref_bce = torch.stack([-(lb[i] * torch.log(p[i]) + (1 - lb[i]) * torch.log(1 - p[i])).mean() for i in range(2)]).mean()
    ref_dice = torch.stack([1 - (2 * (p[i] * lb[i]).sum() + 1) / (p[i].sum() + lb[i].sum() + 1) for i in range(2)]).mean()
    assert torch.allclose(bce, ref_bce, atol=1e-5) and torch.allclose(dice, ref_dice, atol=1e-6)


def test_empty_and_full_masks_give_finite_loss_and_gradient():
    for fill in (0.0, 1.0):
        logits = torch.randn(1, 1, 16, 16, requires_grad=True)
        bce, dice = point_sampled_loss(logits, torch.full((1, 1, 64, 64), fill), num_points=256,
                                       generator=torch.Generator().manual_seed(5))
        (bce + dice).backward()
        assert torch.isfinite(bce) and torch.isfinite(dice) and torch.isfinite(logits.grad).all()
        assert float(logits.grad.abs().sum()) > 0


def test_generator_isolates_the_sampling_from_the_global_random_state():
    logits, target = torch.randn(1, 1, 16, 16), torch.zeros(1, 1, 32, 32)
    torch.manual_seed(11)
    before = torch.get_rng_state()
    a = point_sampled_loss(logits, target, num_points=64, generator=torch.Generator().manual_seed(6))
    assert torch.equal(torch.get_rng_state(), before)          # global stream untouched
    b = point_sampled_loss(logits, target, num_points=64, generator=torch.Generator().manual_seed(6))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    c = point_sampled_loss(logits, target, num_points=64)       # no generator: global stream is used
    assert not torch.equal(torch.get_rng_state(), before) and torch.isfinite(c[0])


def test_thin_structures_survive_in_the_high_resolution_target():
    # a 3 px wide bar in a 2400 x 1800 mask: present in the 1024 letterbox, mostly lost at 256
    mask = np.zeros((1800, 2400), np.uint8)
    mask[:, 1000:1003] = 1
    hr = letterbox_mask(mask, 1024)
    lo = F.interpolate(letterbox_mask(mask, 518)[None], size=(256, 256), mode="nearest")[0]
    rows_hr, rows_lo = int(round(1800 * 1024 / 2400)), int(round(1800 * 256 / 2400))
    assert float(hr[0, :rows_hr].sum(1).min()) >= 1            # every content row keeps the bar
    assert float(lo[0, :rows_lo].sum(1).min()) == 0            # the 256 grid misses it in some rows
    assert float(hr[0, rows_hr:].sum()) == 0                   # padding stays background
