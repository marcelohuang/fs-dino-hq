"""Per-image segmentation metrics on binary masks at ground-truth resolution.

  iou, dice            area overlap; 0 when the denominator is 0
  boundary_precision   share of predicted boundary pixels within ``tolerance``
                       pixels (Euclidean) of the ground-truth boundary
  boundary_recall      the converse; boundary_f is their harmonic mean.
                       A boundary is the mask minus its 4-neighbour erosion
                       (pixels on the image border count as boundary); all
                       three are 0 when either boundary is empty
  thin_recall          recall over ground-truth pixels whose distance to the
                       nearest background pixel is at most thin_width / 2:
                       every pixel of structures up to about ``thin_width``
                       pixels wide plus the outer rim of wider structures;
                       NaN when that set is empty
  fragmentation_ratio  number of 8-connected predicted components divided by
                       the number of ground-truth components (lower is better;
                       an empty prediction gives 0); NaN when the ground
                       truth has no component
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage


def iou_dice(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """IoU and Dice of two binary masks; 0.0 when the denominator is empty."""
    p, g = pred.astype(bool), gt.astype(bool)
    inter, union = (p & g).sum(), (p | g).sum()
    iou = inter / union if union else 0.0
    dice = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 0.0
    return float(iou), float(dice)


def boundary_of(mask: np.ndarray) -> np.ndarray:
    """1-pixel boundary: mask minus its erosion."""
    eroded = ndimage.binary_erosion(mask, iterations=1, border_value=0)
    return mask & ~eroded


def compute_metrics(pred: np.ndarray, gt: np.ndarray,
                    tolerance: int = 5, thin_width: int = 12) -> Dict[str, float]:
    """All metrics of the module docstring for boolean masks of equal shape."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    p_sum, g_sum = float(pred.sum()), float(gt.sum())
    iou = inter / union if union else 0.0
    dice = 2 * inter / (p_sum + g_sum) if (p_sum + g_sum) else 0.0

    pb, gb = boundary_of(pred), boundary_of(gt)
    if pb.any() and gb.any():
        dist_to_gb = ndimage.distance_transform_edt(~gb)
        dist_to_pb = ndimage.distance_transform_edt(~pb)
        b_prec = float((dist_to_gb[pb] <= tolerance).mean())
        b_rec = float((dist_to_pb[gb] <= tolerance).mean())
        b_f = (2 * b_prec * b_rec / (b_prec + b_rec)) if (b_prec + b_rec) else 0.0
    else:
        b_prec = b_rec = b_f = 0.0

    if gt.any():
        gt_edt = ndimage.distance_transform_edt(gt)
        thin = gt & (gt_edt <= thin_width / 2.0)
        thin_recall = float(pred[thin].mean()) if thin.any() else float("nan")
        n_thin = int(thin.sum())
    else:
        thin_recall, n_thin = float("nan"), 0

    struct = np.ones((3, 3), dtype=bool)
    n_gt = int(ndimage.label(gt, structure=struct)[1])
    n_pred = int(ndimage.label(pred, structure=struct)[1])
    frag = (n_pred / n_gt) if n_gt else float("nan")

    return {
        "iou": iou, "dice": dice,
        "boundary_precision": b_prec, "boundary_recall": b_rec, "boundary_f": b_f,
        "thin_recall": thin_recall, "n_thin_px": n_thin,
        "n_components_gt": n_gt, "n_components_pred": n_pred,
        "fragmentation_ratio": frag,
    }
