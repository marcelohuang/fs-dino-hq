"""End-to-end use of the inference API on generated images.

    python examples/synthetic_demo.py --out demo/                       # untrained model
    python examples/synthetic_demo.py --out demo/ \\
        --hq-ckpt hq_best.pt --dino-ckpt dinov2_vitb14_pretrain.pth

The script draws synthetic "bar grid" images with LabelMe annotations (five
supports and three queries), writes them in the folder layout expected by
``fs-dino-hq predict``, runs the predictor and evaluates the predictions.

Without checkpoints it builds the model with random weights. The DINOv2
architecture still comes from torch.hub, which downloads the pinned DINOv2 code
once; without network access, pass --dinov2-repo or set FS_DINO_HQ_DINOV2_REPO
to a local DINOv2 clone (as for ``fs-dino-hq predict``). The untrained model
only shows the call sequence and the output format: the predictions are
meaningless. The generated images are simple drawings, not photographs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from fs_dino_hq.data.labelme import labelme_json_to_mask
from fs_dino_hq.evaluation.evaluate import evaluate_dir, summarise
from fs_dino_hq.inference import Predictor


def draw_bar_grid(path: Path, seed: int, size=(640, 480)) -> None:
    """Grey background with a grid of dark bars; the bars are annotated as polygons."""
    rng = np.random.default_rng(seed)
    w, h = size
    img = Image.fromarray(rng.integers(150, 200, (h, w, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    shapes = []
    spacing = int(rng.integers(50, 90))
    width = int(rng.integers(6, 12))
    for x in range(int(rng.integers(20, 50)), w - width, spacing):        # vertical bars
        poly = [[x, 0], [x + width, 0], [x + width, h - 1], [x, h - 1]]
        shapes.append(poly)
    for y in range(int(rng.integers(20, 50)), h - width, spacing):        # horizontal bars
        poly = [[0, y], [w - 1, y], [w - 1, y + width], [0, y + width]]
        shapes.append(poly)
    for poly in shapes:
        draw.polygon([tuple(p) for p in poly], fill=(60, 50, 45))
    img.save(path.with_suffix(".png"))
    path.with_suffix(".json").write_text(json.dumps({
        "imageHeight": h, "imageWidth": w, "imagePath": path.with_suffix(".png").name,
        "shapes": [{"label": "bar", "points": p, "shape_type": "polygon"} for p in shapes]}))


def untrained_predictor(device, dinov2_repo=None) -> Predictor:
    from fs_dino_hq.modeling.backbones import build_dinov2_vitb14, build_sam_prompt_decoder
    from fs_dino_hq.modeling.distill_adapter import BottleneckAdapter as DistillationAdapter
    from fs_dino_hq.modeling.model import FSDinoHQModel
    dino = build_dinov2_vitb14(dinov2_repo)
    model = FSDinoHQModel(dino, build_sam_prompt_decoder(), adapter_norm_first=True, adapter_dropout=0.0)
    model.early_adapter = DistillationAdapter(in_channels=768, out_channels=256)
    return Predictor(model.eval().to(device))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--hq-ckpt")
    p.add_argument("--dino-ckpt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dinov2-repo",
                   help="local clone of facebookresearch/dinov2 (default: $FS_DINO_HQ_DINOV2_REPO, "
                        "else the pinned commit via torch.hub)")
    args = p.parse_args()

    out = Path(args.out)
    sup_dir, q_dir = out / "supports", out / "queries"
    sup_dir.mkdir(parents=True, exist_ok=True)
    q_dir.mkdir(parents=True, exist_ok=True)
    supports = [sup_dir / f"support_{i}" for i in range(5)]
    queries = [q_dir / f"query_{i}" for i in range(3)]
    for i, s in enumerate(supports):
        draw_bar_grid(s, seed=i)
    for i, q in enumerate(queries):
        draw_bar_grid(q, seed=100 + i)
    (sup_dir / "support_set.json").write_text(json.dumps([s.name for s in supports]))

    device = torch.device(args.device)
    if args.hq_ckpt and args.dino_ckpt:
        predictor = Predictor.from_checkpoint(args.hq_ckpt, args.dino_ckpt, device=device,
                                              dinov2_repo=args.dinov2_repo)
    else:
        print("No checkpoints given: using an untrained model; predictions are meaningless.")
        predictor = untrained_predictor(device, args.dinov2_repo)

    sup_imgs = [Image.open(s.with_suffix(".png")) for s in supports]
    sup_masks = [labelme_json_to_mask(s.with_suffix(".json")) for s in supports]
    (out / "pred_mask").mkdir(exist_ok=True)
    for q in queries:
        res = predictor.predict(q.with_suffix(".png"), sup_imgs, sup_masks)
        Image.fromarray(res.mask * 255).save(out / "pred_mask" / f"{q.name}.png")
        print(f"{q.name}: probability range [{res.probability.min():.3f}, {res.probability.max():.3f}], "
              f"foreground {res.mask.mean():.3f}")
    rows, skipped, _ = evaluate_dir(out / "pred_mask", q_dir)
    agg = summarise(rows)
    print(f"mean IoU {agg['iou']['mean']:.3f} over {len(rows)} queries ({len(skipped)} skipped)")


if __name__ == "__main__":
    main()
