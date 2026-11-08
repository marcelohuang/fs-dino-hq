"""Per-image metrics and the paired bootstrap on hand-made masks."""
import math

import numpy as np
import pytest

from fs_dino_hq.evaluation.compare import paired_bootstrap
from fs_dino_hq.evaluation.metrics import boundary_of, compute_metrics, iou_dice


def _rect(shape, y0, y1, x0, x1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_iou_dice_values():
    gt = _rect((20, 20), 0, 10, 0, 10)          # 100 px
    pred = _rect((20, 20), 0, 10, 5, 15)        # 100 px, 50 overlap
    iou, dice = iou_dice(pred, gt)
    assert iou == pytest.approx(50 / 150) and dice == pytest.approx(100 / 200)
    m = compute_metrics(pred, gt)
    assert m["iou"] == pytest.approx(50 / 150) and m["dice"] == pytest.approx(0.5)
    assert iou_dice(np.zeros((4, 4)), np.zeros((4, 4))) == (0.0, 0.0)   # empty denominators
    assert iou_dice(np.ones((4, 4)), np.zeros((4, 4))) == (0.0, 0.0)
    assert iou_dice(gt.astype(np.uint8) * 255, gt) == (1.0, 1.0)        # any non-zero is fg


def test_perfect_prediction():
    gt = _rect((40, 40), 5, 30, 8, 20)
    m = compute_metrics(gt.copy(), gt)
    assert m["iou"] == m["dice"] == 1.0
    assert m["boundary_precision"] == m["boundary_recall"] == m["boundary_f"] == 1.0
    assert m["thin_recall"] == 1.0
    assert m["fragmentation_ratio"] == 1.0 and m["n_components_gt"] == m["n_components_pred"] == 1


def test_boundary_of_four_neighbour_erosion_with_image_border():
    full = np.ones((5, 6), bool)
    b = boundary_of(full)
    ring = full.copy()
    ring[1:-1, 1:-1] = False
    assert np.array_equal(b, ring)                       # image border counts as boundary
    plus = np.zeros((7, 7), bool)
    plus[1:6, 1:6] = True
    b = boundary_of(plus)
    assert b.sum() == 16 and not b[2:5, 2:5].any()
    # diagonal neighbours do not matter (4-neighbour erosion)
    diag = np.zeros((5, 5), bool)
    diag[1:4, 1:4] = True
    diag[0, 0] = True
    assert not boundary_of(diag)[2, 2]


def _brute_boundary_scores(pred, gt, tol):
    def bnd(m):
        out = np.zeros_like(m)
        h, w = m.shape
        for y in range(h):
            for x in range(w):
                if m[y, x]:
                    nb = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
                    out[y, x] = any(not (0 <= a < h and 0 <= c < w) or not m[a, c] for a, c in nb)
        return out
    pb, gb = bnd(pred), bnd(gt)
    P, G = np.argwhere(pb), np.argwhere(gb)
    d = np.sqrt(((P[:, None, :] - G[None, :, :]) ** 2).sum(-1))
    prec = float((d.min(1) <= tol).mean())
    rec = float((d.min(0) <= tol).mean())
    f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f


@pytest.mark.parametrize("tol", [0, 1, 2, 3, 5])
def test_boundary_f_matches_definition(tol):
    rng = np.random.default_rng(tol)
    gt = _rect((30, 34), 4, 22, 5, 25)
    gt[10:14, 25:31] = True
    pred = np.roll(gt, (2, 3), axis=(0, 1)) | (rng.random(gt.shape) > 0.97)
    m = compute_metrics(pred, gt, tolerance=tol)
    prec, rec, f = _brute_boundary_scores(pred, gt, tol)
    assert m["boundary_precision"] == pytest.approx(prec)
    assert m["boundary_recall"] == pytest.approx(rec)
    assert m["boundary_f"] == pytest.approx(f)


def test_boundary_f_tolerance():
    gt = _rect((40, 50), 10, 30, 10, 30)
    shifted = _rect((40, 50), 10, 30, 13, 33)            # 3 px to the right
    assert compute_metrics(shifted, gt, tolerance=3)["boundary_f"] == 1.0
    assert compute_metrics(shifted, gt, tolerance=5)["boundary_f"] == 1.0
    assert compute_metrics(shifted, gt, tolerance=2)["boundary_f"] < 1.0
    empty = compute_metrics(np.zeros_like(gt), gt)
    assert empty["boundary_precision"] == empty["boundary_recall"] == empty["boundary_f"] == 0.0


def test_thin_recall_definition():
    """thin = GT pixels with EDT <= thin_width/2: whole thin bars plus the
    outer rim of wide regions (not a width filter)."""
    shape = (60, 80)
    bar = _rect(shape, 5, 9, 5, 70)                      # 4 px wide: all thin
    square = _rect(shape, 20, 50, 30, 60)                # 30 x 30
    gt = bar | square
    m = compute_metrics(bar.copy(), gt, thin_width=12)
    rim = 30 * 30 - 18 * 18                              # outer 6 px of the square
    assert m["n_thin_px"] == bar.sum() + rim
    assert m["thin_recall"] == pytest.approx(bar.sum() / (bar.sum() + rim))
    core = _rect(shape, 26, 44, 36, 54)                  # inner 18 x 18 of the square
    m2 = compute_metrics(core, gt, thin_width=12)
    assert m2["thin_recall"] == 0.0                      # the core is not "thin"
    # thin_width=2 (EDT <= 1): one-pixel rims only; bar interior is 2 x 63 px
    assert compute_metrics(gt, gt, thin_width=2)["n_thin_px"] == (
        bar.sum() - 2 * 63 + (30 * 30 - 28 * 28))
    nan = compute_metrics(np.zeros(shape, bool), np.zeros(shape, bool))
    assert math.isnan(nan["thin_recall"]) and nan["n_thin_px"] == 0


def test_fragmentation():
    shape = (30, 30)
    gt = _rect(shape, 5, 25, 5, 25)
    pred = _rect(shape, 5, 8, 5, 8) | _rect(shape, 10, 13, 10, 13) | _rect(shape, 20, 23, 20, 23)
    m = compute_metrics(pred, gt)
    assert m["n_components_pred"] == 3 and m["n_components_gt"] == 1
    assert m["fragmentation_ratio"] == 3.0
    # 8-connectivity: diagonal neighbours form one component
    diag = np.zeros(shape, bool)
    diag[1, 1] = diag[2, 2] = diag[3, 3] = True
    assert compute_metrics(diag, gt)["n_components_pred"] == 1
    # empty prediction scores 0 (looks best: read together with IoU)
    assert compute_metrics(np.zeros(shape, bool), gt)["fragmentation_ratio"] == 0.0
    # empty ground truth: undefined
    assert math.isnan(compute_metrics(pred, np.zeros(shape, bool))["fragmentation_ratio"])
    two_gt = _rect(shape, 0, 3, 0, 3) | _rect(shape, 10, 13, 10, 13)
    assert compute_metrics(pred, two_gt)["fragmentation_ratio"] == 1.5


def test_compute_metrics_accepts_uint8():
    gt = _rect((20, 20), 2, 12, 2, 12)
    a = compute_metrics(gt.astype(np.uint8), gt.astype(np.uint8) * 255)
    b = compute_metrics(gt, gt)
    assert a == b


# ── paired bootstrap ────────────────────────────────────────────────────────

def _runs(n=30, seed=0):
    rng = np.random.default_rng(seed)
    base, new = {}, {}
    for i in range(n):
        b = {"iou": rng.random(), "dice": rng.random(), "boundary_f": rng.random(),
             "thin_recall": rng.random(), "fragmentation_ratio": float(rng.integers(1, 30))}
        base[f"q{i:02d}"] = b
        new[f"q{i:02d}"] = {k: v + rng.normal(0.02, 0.05) for k, v in b.items()}
    return base, new


def test_paired_bootstrap_deterministic_with_seed():
    base, new = _runs()
    r1 = paired_bootstrap(base, new, seed=7, n_boot=500)
    r2 = paired_bootstrap(base, new, seed=7, n_boot=500)
    r3 = paired_bootstrap(base, new, seed=8, n_boot=500)
    assert r1 == r2
    assert r1["iou"]["ci95"] != r3["iou"]["ci95"]
    assert r1["iou"]["delta"] == r3["iou"]["delta"]       # only the interval is random


def test_paired_bootstrap_reference_algorithm():
    """One default_rng(seed) drawn sequentially over the metrics; percentile CI."""
    base, new = _runs(n=12, seed=3)
    metrics = ("iou", "dice", "boundary_f", "thin_recall", "fragmentation_ratio")
    rec = paired_bootstrap(base, new, seed=42, n_boot=1000)
    rng = np.random.default_rng(42)
    ids = sorted(base)
    for m in metrics:
        d = np.array([new[q][m] for q in ids]) - np.array([base[q][m] for q in ids])
        idx = rng.integers(0, len(ids), size=(1000, len(ids)))
        lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
        assert rec[m]["ci95"] == [float(lo), float(hi)]
        assert rec[m]["delta"] == float(d.mean())
        win = int((d < 0).sum()) if m == "fragmentation_ratio" else int((d > 0).sum())
        assert rec[m]["win_tie_loss"][0] == win
        assert sum(rec[m]["win_tie_loss"]) == len(ids)
    assert rec["fragmentation_ratio"]["direction"] == "lower_is_better"
    assert rec["iou"]["direction"] == "higher_is_better"
    assert rec["n_queries"] == 12


def test_paired_bootstrap_nan_and_ties():
    base = {"a": {"iou": 0.5, "thin_recall": float("nan")},
            "b": {"iou": 0.4, "thin_recall": 0.2},
            "c": {"iou": 0.3, "thin_recall": 0.1}}
    new = {"a": {"iou": 0.5, "thin_recall": 0.9},
           "b": {"iou": 0.6, "thin_recall": 0.1},
           "c": {"iou": 0.2, "thin_recall": 0.3}}
    rec = paired_bootstrap(base, new, seed=0, n_boot=200, metrics=("iou", "thin_recall"))
    assert rec["iou"]["win_tie_loss"] == [1, 1, 1] and rec["iou"]["n_queries"] == 3
    assert rec["thin_recall"]["n_queries"] == 2                 # NaN dropped for this metric only
    assert rec["thin_recall"]["delta"] == pytest.approx(((0.1 - 0.2) + (0.3 - 0.1)) / 2)
    lo, hi = rec["iou"]["ci95"]
    assert lo <= rec["iou"]["delta"] <= hi


def test_paired_bootstrap_metric_without_finite_pairs():
    base = {"a": {"iou": 0.5, "thin_recall": float("nan")}, "b": {"iou": 0.4, "thin_recall": float("nan")}}
    new = {"a": {"iou": 0.6, "thin_recall": 0.2}, "b": {"iou": 0.5, "thin_recall": 0.1}}
    rec = paired_bootstrap(base, new, seed=0, n_boot=100, metrics=("thin_recall", "iou"))
    assert rec["thin_recall"]["n_queries"] == 0 and rec["thin_recall"]["ci95"] is None
    # the empty metric draws nothing: iou gets the generator's first draws
    ref = paired_bootstrap(base, new, seed=0, n_boot=100, metrics=("iou",))
    assert rec["iou"] == ref["iou"]
