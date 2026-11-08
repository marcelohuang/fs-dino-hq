"""Overlays and comparison panels for predictions (visual inspection only;
nothing here affects predictions or metrics)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def mask_contour(mask: np.ndarray) -> np.ndarray:
    """Inner 4-neighbourhood boundary of a binary mask (no wrap-around at the borders)."""
    g = mask.astype(bool)
    core = g.copy()
    for ax in (0, 1):
        for sh in (1, -1):
            r = np.roll(g, sh, axis=ax)
            if ax == 0:
                r[0 if sh == 1 else -1, :] = False
            else:
                r[:, 0 if sh == 1 else -1] = False
            core &= r
    return g & ~core


def _caption(im: Image.Image, text: str) -> None:
    font = ImageFont.load_default()
    if not isinstance(font, ImageFont.FreeTypeFont):
        # the bitmap default font (older Pillow, or Pillow without FreeType) is Latin-1 only
        text = text.encode("latin-1", "replace").decode("latin-1")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 24], fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 0), font=font)


def overlay(image: Image.Image, pred: np.ndarray, gt: Optional[np.ndarray] = None,
            caption: Optional[str] = None, max_side: Optional[int] = None) -> Image.Image:
    """Query image with the predicted mask in translucent red and, if given,
    the ground-truth contour in green. ``max_side`` downsizes large images."""
    base = np.array(image.convert("RGB"))
    ov = base.copy()
    p = pred.astype(bool)
    ov[p] = (0.45 * ov[p] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
    if gt is not None:
        ov[mask_contour(gt)] = [60, 255, 60]
    im = Image.fromarray(ov)
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
    if caption:
        _caption(im, caption)
    return im


def comparison_panel(image: Image.Image, base_mask: np.ndarray, mask: np.ndarray,
                     gt: Optional[np.ndarray] = None, caption: str = "",
                     panel_h: int = 360) -> Image.Image:
    """Side-by-side tiles (masks white on black): query | [ground truth] |
    Stage 2 base | corrected prediction."""
    w = max(1, int(round(image.width * panel_h / image.height)))
    tiles = [np.array(image.convert("RGB").resize((w, panel_h)))]
    labels = ["query"]
    masks = []
    if gt is not None:
        masks.append(gt); labels.append("ground truth")
    masks += [base_mask, mask]
    labels += ["Stage 2 base", "corrected (output)"]
    for m in masks:
        mi = Image.fromarray(m.astype(np.uint8) * 255).resize((w, panel_h), Image.NEAREST)
        tiles.append(np.stack([np.array(mi)] * 3, axis=-1))
    sheet = Image.fromarray(np.concatenate(tiles, axis=1))
    if caption:
        _caption(sheet, caption)
    d = ImageDraw.Draw(sheet)
    for i, lab in enumerate(labels):
        d.rectangle([i * w, panel_h - 20, i * w + 7 * len(lab) + 10, panel_h], fill=(0, 0, 0))
        d.text((i * w + 4, panel_h - 17), lab, fill=(255, 255, 255), font=ImageFont.load_default())
    return sheet


def support_sheet(images: Sequence[Image.Image], masks: Sequence[np.ndarray],
                  title: str = "", thumb: int = 256) -> Image.Image:
    """Row of support thumbnails (square-resized) with their masks tinted blue."""
    tiles = []
    for img, m in zip(images, masks):
        a = np.array(img.convert("RGB").resize((thumb, thumb)))
        mm = np.array(Image.fromarray((np.asarray(m) > 0).astype(np.uint8) * 255)
                      .resize((thumb, thumb), Image.NEAREST)) > 0
        a[mm] = (0.5 * a[mm] + 0.5 * np.array([60, 120, 255])).astype(np.uint8)
        tiles.append(a)
    sheet = Image.fromarray(np.concatenate(tiles, axis=1))
    if title:
        _caption(sheet, title)
    return sheet


def save(image: Image.Image, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
