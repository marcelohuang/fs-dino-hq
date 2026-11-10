"""Evaluate binary prediction masks against LabelMe ground truth.

    fs-dino-hq evaluate --pred-dir predictions/pred_mask --gt-dir queries/ \\
        --out-csv predictions/metrics.csv --out-json predictions/metrics.json

Each ``<stem>.png`` in ``--pred-dir`` is matched with
``<stem>.json`` in ``--gt-dir`` (``--strip-suffix`` removes a suffix from the
prediction stem first). Predictions are read as 8-bit grayscale and a pixel
value above 127 is foreground, i.e. 0/255 masks as written by ``predict``
(``pred_mask/``); a mask whose non-zero values are all <= 127 (for example a
0/1 mask) is refused. Predictions whose size differs from the annotation are
resized with nearest neighbour. Predictions without an annotation file, or
whose ground truth is empty, are skipped and counted; two prediction files
that map to the same image are an error. An annotation in ``--gt-dir`` without
a prediction is an error as well, because a mean over fewer images than
intended is not comparable; ``--allow-missing`` evaluates the available
predictions and lists the missing ones in the summary. Metrics are defined in
``fs_dino_hq.evaluation.metrics``; the summary reports the unweighted mean
and median over images (NaN values dropped per metric).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from ..data.labelme import labelme_json_to_mask
from .metrics import compute_metrics

SUMMARY_METRICS = ("iou", "dice", "boundary_precision", "boundary_recall", "boundary_f",
                   "thin_recall", "fragmentation_ratio")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fs-dino-hq evaluate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", required=True, help="binary prediction PNGs, 0/255 (value > 127 = foreground)")
    p.add_argument("--gt-dir", required=True, help="LabelMe JSON files")
    p.add_argument("--strip-suffix", default="", help="suffix removed from prediction stems")
    p.add_argument("--label-filter", nargs="+", default=None,
                   help="LabelMe labels counted as foreground (default: all labels)")
    p.add_argument("--tolerance", type=int, default=5, help="boundary tolerance in pixels")
    p.add_argument("--thin-width", type=int, default=12, help="thin-structure width in pixels")
    p.add_argument("--allow-missing", action="store_true",
                   help="do not fail when an annotation in --gt-dir has no prediction")
    p.add_argument("--out-csv", default=None, help="default <pred-dir>/metrics.csv")
    p.add_argument("--out-json", default=None, help="default <pred-dir>/metrics.json")
    return p


def evaluate_dir(pred_dir: Path, gt_dir: Path, strip_suffix: str = "", label_filter=None,
                 tolerance: int = 5, thin_width: int = 12):
    """Per-image metrics (list of dicts), the skipped files with reasons, and the
    annotated images of ``gt_dir`` that have no prediction.

    ``label_filter=None`` merges all labels; a list keeps only those labels.
    """
    rows: List[Dict] = []
    skipped: List[str] = []
    seen: Dict[str, str] = {}
    for pp in sorted(Path(pred_dir).glob("*.png")):
        stem = pp.stem
        if strip_suffix and stem.endswith(strip_suffix):
            stem = stem[: -len(strip_suffix)]
        if stem in seen:
            raise ValueError(f"{pp.name} and {seen[stem]} both map to image {stem!r}")
        seen[stem] = pp.name
        jp = Path(gt_dir) / f"{stem}.json"
        if not jp.exists():
            skipped.append(f"{pp.name}: no annotation")
            continue
        gt = labelme_json_to_mask(jp, label_filter).astype(bool)
        if not gt.any():
            skipped.append(f"{pp.name}: empty ground truth")
            continue
        pred_img = Image.open(pp).convert("L")
        if pred_img.size != (gt.shape[1], gt.shape[0]):
            pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
        arr = np.array(pred_img)
        if arr.any() and arr.max() <= 127:
            raise ValueError(f"{pp.name}: maximum pixel value {int(arr.max())}; predictions must be 0/255 "
                             "masks (value > 127 = foreground), e.g. multiply a 0/1 mask by 255")
        pred = arr > 127
        rows.append({"image": stem, "pred_file": pp.name,
                     **compute_metrics(pred, gt, tolerance, thin_width)})
    missing = [jp.stem for jp in sorted(Path(gt_dir).glob("*.json"))
               if jp.stem not in seen and _is_annotation(jp)]
    return rows, skipped, missing


def _is_annotation(path: Path) -> bool:
    """True for a LabelMe file with at least one shape (other JSON files in the folder are ignored)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and bool(data.get("shapes"))


def summarise(rows: List[Dict]) -> Dict:
    agg = {}
    for key in SUMMARY_METRICS:
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
        agg[key] = ({"mean": float(np.mean(vals)), "median": float(np.median(vals)), "n": len(vals)}
                    if vals else None)
    return agg


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    pred_dir = Path(args.pred_dir)
    try:
        rows, skipped, missing = evaluate_dir(pred_dir, Path(args.gt_dir), args.strip_suffix,
                                              args.label_filter, args.tolerance, args.thin_width)
    except ValueError as e:
        raise SystemExit(str(e))
    if not rows:
        raise SystemExit(f"no prediction/annotation pairs evaluated in {pred_dir}")
    if missing and not args.allow_missing:
        raise SystemExit(f"{len(missing)} annotated image(s) in {args.gt_dir} have no prediction in {pred_dir} "
                         f"(first: {missing[0]}); pass --allow-missing to evaluate the others")
    out_csv = Path(args.out_csv) if args.out_csv else pred_dir / "metrics.csv"
    out_json = Path(args.out_json) if args.out_json else pred_dir / "metrics.json"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"pred_dir": str(pred_dir), "gt_dir": str(args.gt_dir),
               "tolerance_px": args.tolerance, "thin_width_px": args.thin_width,
               "label_filter": args.label_filter, "n_evaluated": len(rows),
               "n_skipped": len(skipped), "skipped": skipped,
               "n_missing_predictions": len(missing), "missing_predictions": missing,
               "aggregate": summarise(rows)}
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(rows)} images evaluated, {len(skipped)} skipped, {len(missing)} without prediction")
    for k, v in summary["aggregate"].items():
        if v:
            print(f"  {k:20s} mean {v['mean']:.4f}  median {v['median']:.4f}")
    print(f"per-image: {out_csv}\nsummary:   {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
