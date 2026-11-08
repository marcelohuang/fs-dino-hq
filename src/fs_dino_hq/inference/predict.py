"""Predict masks for a folder of query images from a fixed set of annotated supports.

    fs-dino-hq predict \\
        --hq-ckpt hq_best.pt --dino-ckpt dinov2_vitb14_pretrain.pth \\
        --support-dir supports/ --support-set supports/support_set.json \\
        --query-dir queries/ --output-dir predictions/

Supports: images with LabelMe JSON files of the same stem in ``--support-dir``.
``--support-set`` (JSON list of K stems, in order) selects and orders them;
without it every annotated image in ``--support-dir`` is used, sorted by name.
The same K supports are used for every query.

Queries: every image in ``--query-dir`` (``--query-pattern``). Output files
are named by the image stem, so two query images may not share a stem. A
LabelMe JSON with the same stem is optional; when present, per-query IoU and
Dice are written to ``predictions.csv`` (computed at the annotation's
resolution, as ``fs-dino-hq evaluate`` does; use that command for the full
set of metrics).

Outputs in ``--output-dir``:
  pred_prob/<stem>.png       corrected probability, 8 bit (value = floor(p * 255))
  pred_mask/<stem>.png       corrected mask (0/255), probability >= --threshold
  base_prob/, base_mask/     the Stage 2 base output (no HQ branch), with --save-base
  visualizations/            overlays and comparison panels, with --save-visualizations
  predictions.csv            one row per query
  inference_config.json      arguments, supports, checkpoint SHA-256, package version
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from .. import __version__
from ..data.labelme import IMG_EXTENSIONS, labelme_json_to_mask, open_image_rgb
from ..evaluation.metrics import iou_dice
from ..utils.provenance import file_sha256
from .predictor import Predictor
from .visualization import comparison_panel, overlay, save, support_sheet

SUPPORT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG")


def find_supports(support_dir: Path) -> List[Tuple[Path, Path]]:
    """(image, json) pairs in ``support_dir`` (non-recursive), sorted by JSON name.

    The image is found by the JSON's stem with extension preference
    .png, .jpg, .jpeg, .PNG, .JPG; the ``imagePath`` field is ignored.
    """
    out = []
    for jf in sorted(Path(support_dir).glob("*.json")):
        for ext in SUPPORT_EXTENSIONS:
            img = jf.with_suffix(ext)
            if img.exists():
                out.append((img, jf))
                break
    return out


def read_support_set(path: Path) -> List[str]:
    """A JSON list of image stems, or a list holding exactly one such list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data and isinstance(data[0], list):
        if len(data) != 1:
            raise ValueError(f"{path}: holds {len(data)} support sets; give one")
        data = data[0]
    if not data or not all(isinstance(s, str) for s in data):
        raise ValueError(f"{path}: expected a JSON list of image stems")
    if len(set(data)) != len(data):
        raise ValueError(f"{path}: duplicate supports")
    return data


def resolve_supports(support_dir: Path, support_set) -> List[Tuple[Path, Path]]:
    pool = find_supports(support_dir)
    if not pool:
        raise SystemExit(f"no annotated images (image + LabelMe JSON) in {support_dir}")
    if support_set is None:
        return pool
    by_stem = {img.stem: (img, js) for img, js in pool}
    stems = read_support_set(support_set)
    missing = [s for s in stems if s not in by_stem]
    if missing:
        raise SystemExit(f"support set entries without image + JSON in {support_dir}: {missing}")
    return [by_stem[s] for s in stems]


def find_queries(query_dir: Path, pattern: str) -> List[Path]:
    return sorted(p for p in Path(query_dir).glob(pattern)
                  if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)


def resize_nearest(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """{0,1} uint8 mask resized to ``shape`` (H, W) with nearest neighbour
    (the convention of ``fs-dino-hq evaluate`` for size mismatches)."""
    if mask.shape == tuple(shape):
        return mask
    im = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return (np.array(im.resize((shape[1], shape[0]), Image.NEAREST)) > 127).astype(np.uint8)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fs-dino-hq predict", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hq-ckpt", required=True, dest="hq_checkpoint",
                   help="HQ checkpoint (output of train-hq)")
    p.add_argument("--dino-ckpt", required=True, dest="dinov2_checkpoint",
                   help="DINOv2 ViT-B/14 weights (dinov2_vitb14_pretrain.pth)")
    p.add_argument("--dinov2-repo", default=None,
                   help="local clone of facebookresearch/dinov2 (default: torch.hub cache / GitHub)")
    p.add_argument("--support-dir", required=True)
    p.add_argument("--support-set", default=None, help="JSON list of support stems (order is kept)")
    p.add_argument("--query-dir", required=True)
    p.add_argument("--query-pattern", default="*", help="glob inside --query-dir (default: *)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--label-filter", nargs="+", default=None,
                   help="LabelMe labels to use as foreground (default: all labels)")
    p.add_argument("--save-base", action="store_true", help="also save the Stage 2 base output")
    p.add_argument("--save-visualizations", action="store_true")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument("--overwrite", action="store_true",
                   help="delete earlier outputs of this command in --output-dir first")
    return p


OUTPUT_DIRS = ("pred_prob", "pred_mask", "base_prob", "base_mask", "visualizations")
OUTPUT_FILES = ("predictions.csv", "inference_config.json")


def clear_or_refuse(out: Path, overwrite: bool) -> None:
    """Refuse an output directory holding earlier predictions (stale masks would
    be evaluated with the new ones) unless ``overwrite``, which removes them."""
    stale = [out / d for d in OUTPUT_DIRS if (out / d).is_dir() and any((out / d).iterdir())]
    stale += [out / f for f in OUTPUT_FILES if (out / f).exists()]
    if not stale:
        return
    if not overwrite:
        raise SystemExit(f"{out} already holds predictions ({', '.join(p.name for p in stale)}); "
                         "use --overwrite to replace them or choose another --output-dir")
    for p in stale:
        shutil.rmtree(p) if p.is_dir() else p.unlink()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.output_dir)
    supports = resolve_supports(Path(args.support_dir), args.support_set)
    queries = find_queries(Path(args.query_dir), args.query_pattern)
    if not queries:
        raise SystemExit(f"no query images in {args.query_dir}")
    stems = [q.stem for q in queries]
    dup = sorted({s for s in stems if stems.count(s) > 1})
    if dup:
        raise SystemExit(f"query images share a stem (outputs are named by stem): {dup[:5]}")
    support_stems = {img.stem for img, _ in supports}
    overlap = [q.name for q in queries if q.stem in support_stems]
    if overlap:
        raise SystemExit(f"query images are also supports: {overlap[:5]}")
    clear_or_refuse(out, args.overwrite)

    device = torch.device(args.device) if args.device else None
    predictor = Predictor.from_checkpoint(args.hq_checkpoint, args.dinov2_checkpoint,
                                          device=device, threshold=args.threshold,
                                          dinov2_repo=args.dinov2_repo)
    k_ckpt = predictor.model.checkpoint_meta.get("n_shot")
    if k_ckpt and k_ckpt != len(supports):
        print(f"[note] {len(supports)} supports; the checkpoint was trained with {k_ckpt}-shot episodes")

    sup_imgs = [open_image_rgb(i) for i, _ in supports]
    sup_masks = [labelme_json_to_mask(j, args.label_filter) for _, j in supports]
    for (img, _), m in zip(supports, sup_masks):
        if not m.any():
            print(f"[warning] support {img.name} has an empty mask (label filter {args.label_filter})")
    dirs = ["pred_prob", "pred_mask"] + (["base_prob", "base_mask"] if args.save_base else [])
    for d in dirs:
        (out / d).mkdir(parents=True, exist_ok=True)
    if args.save_visualizations:
        save(support_sheet(sup_imgs, sup_masks, "supports: " + " | ".join(i.stem for i, _ in supports)),
             out / "visualizations" / "supports.png")

    rows = []
    for qp in queries:
        res = predictor.predict(qp, sup_imgs, sup_masks)
        stem = qp.stem
        Image.fromarray((res.probability * 255).astype(np.uint8)).save(out / "pred_prob" / f"{stem}.png")
        Image.fromarray(res.mask * 255).save(out / "pred_mask" / f"{stem}.png")
        if args.save_base:
            Image.fromarray((res.base_probability * 255).astype(np.uint8)).save(out / "base_prob" / f"{stem}.png")
            Image.fromarray(res.base_mask * 255).save(out / "base_mask" / f"{stem}.png")
        row = {"query": stem, "query_path": str(qp), "foreground_fraction": float(res.mask.mean())}
        gt = None
        gt_path = qp.with_suffix(".json")
        if gt_path.is_file():
            gt = labelme_json_to_mask(gt_path, args.label_filter)
        if gt is not None and not gt.any():
            print(f"[note] {gt_path.name}: empty annotation (label filter {args.label_filter}); "
                  "no IoU/Dice for this query (fs-dino-hq evaluate skips it too)")
            row["empty_annotation"] = True
            gt = None
        if gt is not None:
            if gt.shape != res.mask.shape:
                print(f"[warning] {gt_path.name}: annotation size {gt.shape[::-1]} differs from "
                      f"the image {res.mask.shape[::-1]}; IoU is computed on the prediction "
                      "resized (nearest) to the annotation")
            row["iou"], row["dice"] = iou_dice(resize_nearest(res.mask, gt.shape), gt)
            row["iou_base"], row["dice_base"] = iou_dice(resize_nearest(res.base_mask, gt.shape), gt)
        rows.append(row)
        if args.save_visualizations:
            qimg = open_image_rgb(qp)
            gt_vis = None if gt is None else resize_nearest(gt, res.mask.shape)
            cap = f"{stem}" + (f"  IoU={row['iou']:.3f}" if gt is not None else "")
            save(overlay(qimg, res.mask, gt_vis, cap, max_side=2048),
                 out / "visualizations" / f"{stem}_overlay.png")
            save(comparison_panel(qimg, res.base_mask, res.mask, gt_vis, cap),
                 out / "visualizations" / f"{stem}_compare.png")
        print(f"{stem}: foreground {row['foreground_fraction']:.4f}"
              + (f"  IoU {row['iou']:.4f} (base {row['iou_base']:.4f})" if gt is not None else ""))

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    config = {
        "package_version": __version__,
        "arguments": vars(args),
        "supports": [{"image": str(i), "annotation": str(j)} for i, j in supports],
        "n_supports": len(supports),
        "n_queries": len(queries),
        "hq_checkpoint_sha256": file_sha256(args.hq_checkpoint),
        "checkpoint_meta": predictor.model.checkpoint_meta,
        "output": "corrected = canonical Stage 2 base + HQ residual; mask = probability >= threshold",
        "preprocessing": "letterbox 518 (bilinear image, nearest mask, top-left padding)",
    }
    (out / "inference_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    with_gt = [r for r in rows if "iou" in r]
    if with_gt:
        print(f"mean IoU over {len(with_gt)} annotated queries: "
              f"{np.mean([r['iou'] for r in with_gt]):.4f} (base {np.mean([r['iou_base'] for r in with_gt]):.4f})")
    print(f"wrote {len(rows)} predictions to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
