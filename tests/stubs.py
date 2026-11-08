"""CPU stand-ins for the frozen backbones, and a tiny synthetic LabelMe dataset.

StubDino  mimics the torch.hub DINOv2 ViT-B/14 interface the extractors need
          (``blocks`` for the block-3 hook, ``forward_features`` returning
          ``x_norm_patchtokens``) with a single 14x14 conv patch embedding, so a
          518 input yields the real 37x37 token grid at a fraction of the cost.
StubSam   the real SAM ViT-H prompt encoder and mask decoder (segment_anything
          modules at ViT-H dimensions, random weights) without the image encoder.

All images and annotations created here are random synthetic test fixtures.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer


class StubDino(nn.Module):
    def __init__(self, dim: int = 768, n_blocks: int = 12, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=14, stride=14)
        with torch.no_grad():
            self.patch_embed.weight.copy_(torch.randn(self.patch_embed.weight.shape, generator=g) * 0.02)
            self.patch_embed.bias.copy_(torch.randn(self.patch_embed.bias.shape, generator=g) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(n_blocks)])
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward_features(self, x):
        t = self.patch_embed(x).flatten(2).transpose(1, 2)          # [B,1369,768]
        t = torch.cat([self.cls_token.expand(t.shape[0], -1, -1), t], 1)
        for blk in self.blocks:                                      # hook on blocks[2]
            t = blk(t)
        return {"x_norm_patchtokens": t[:, 1:]}


class StubSam(nn.Module):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(64, 64),
                                            input_image_size=(1024, 1024), mask_in_chans=16)
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256)
        self.image_encoder = None
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)


def make_tiny_photo_dataset(root: Path, n_per_class: int = 3, classes=("beams", "walls"),
                            size=(64, 48), seed: int = 0):
    """Random images with LabelMe JSON (one rectangle polygon each) plus 1-shot
    and 5-shot plan CSVs in the column layout written by ``make-episodes``."""
    rng = np.random.default_rng(seed)
    rows = {c: [] for c in classes}
    for c in classes:
        (root / c).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            w, h = size
            img = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
            stem = f"real-{100 + i}-{i}-{1000 + i}"
            img.save(root / c / f"{stem}.png")
            x0, y0 = 5 + i, 5 + i
            poly = [[x0, y0], [x0 + 20, y0], [x0 + 20, y0 + 15], [x0, y0 + 15]]
            (root / c / f"{stem}.json").write_text(json.dumps(
                {"imageHeight": h, "imageWidth": w, "imagePath": f"{stem}.png",
                 "shapes": [{"label": "spacing", "points": poly}]}))
            rows[c].append({"image": f"{c}/{stem}.png", "json": f"{c}/{stem}.json",
                            "group": f"{c}/{100 + i}"})
    plans = {}
    for k in (1, 5):
        eps = []
        for c in classes:
            items = rows[c]
            for qi, q in enumerate(items):
                # supports cycle through the other items of the class; with tiny
                # pools (k >= n_per_class) they repeat and may include the query
                sup = [items[(qi + 1 + j) % len(items)] for j in range(k)]
                ep = {"cycle": 0, "component": c, "query_image": q["image"],
                      "query_json": q["json"], "query_group": q["group"]}
                for j, s in enumerate(sup, 1):
                    ep[f"support_{j}_image"] = s["image"]; ep[f"support_{j}_json"] = s["json"]
                    ep[f"support_{j}_group"] = s["group"]
                eps.append(ep)
        for i, ep in enumerate(eps):
            ep["episode_idx"] = i
        for split in ("train", "val"):
            p = root / f"{split}_episode_plan_{k}shot.csv"
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(eps[0].keys())); w.writeheader(); w.writerows(eps)
            plans[(split, k)] = p
    return plans
