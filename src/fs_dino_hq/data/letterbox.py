"""Letterbox (aspect-preserving) preprocessing used for training and inference.

Convention (every checkpoint of this package was trained with it):
  - scale = target / max(w, h);  new_w = int(round(w*scale)), new_h = int(round(h*scale))
  - image resized with BILINEAR, mask with NEAREST
  - pasted at the TOP-LEFT (0, 0) corner of a target x target canvas
  - image padding (0, 0, 0) before ImageNet normalisation; mask padding 0

A square stretch to 518 x 518 is not equivalent: a checkpoint must be used
with the preprocessing it was trained with.

Back-mapping: model outputs live on an S x S grid (256 for mask-decoder
logits) aligned with the padded square. Only the top-left content box is
valid; probabilities are cropped to it before resizing to the original (w, h).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T


def content_box(orig_w: int, orig_h: int, target: int) -> Tuple[int, int]:
    """(content_w, content_h) after longest-side resize to `target` (pre-pad)."""
    assert orig_w > 0 and orig_h > 0, f"bad image size {orig_w}x{orig_h}"
    scale = target / max(orig_w, orig_h)
    return int(round(orig_w * scale)), int(round(orig_h * scale))


def content_box_on_grid(orig_w: int, orig_h: int, target: int, grid: int) -> Tuple[int, int]:
    """Valid (w, h) of the content box expressed on a grid×grid output map."""
    cw, ch = content_box(orig_w, orig_h, target)
    gw = max(1, min(grid, int(round(cw * grid / target))))
    gh = max(1, min(grid, int(round(ch * grid / target))))
    return gw, gh


def letterbox_image(img: Image.Image, target: int) -> Image.Image:
    """Longest-side resize + top-left pad onto a black target x target canvas."""
    cw, ch = content_box(img.width, img.height, target)
    resized = img.resize((cw, ch), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), (0, 0, 0))
    canvas.paste(resized, (0, 0))          # top-left padding
    return canvas


def letterbox_mask(mask_np: np.ndarray, target: int) -> torch.Tensor:
    """uint8/bool HxW mask → binarized float tensor [1, target, target]."""
    mask = Image.fromarray(mask_np)
    cw, ch = content_box(mask.width, mask.height, target)
    resized = mask.resize((cw, ch), resample=Image.NEAREST)
    canvas = Image.new("L", (target, target), 0)
    canvas.paste(resized, (0, 0))
    t = torch.from_numpy(np.array(canvas)).float().unsqueeze(0)
    return (t > 0).float()


def unletterbox_prob(prob_grid: np.ndarray, orig_w: int, orig_h: int,
                     target: int) -> np.ndarray:
    """
    Crop a [G, G] float prob map (aligned with the padded square) to its valid
    content box and resize (BILINEAR) back to the ORIGINAL (orig_h, orig_w).

    Rounding note: the content box on the G-grid is rounded to the nearest
    cell (≤ half a cell ≈ target/G/2 px of drift at original scale).
    """
    g = prob_grid.shape[0]
    assert prob_grid.ndim == 2 and prob_grid.shape[1] == g, \
        f"prob_grid must be square, got {prob_grid.shape}"
    gw, gh = content_box_on_grid(orig_w, orig_h, target, g)
    valid = prob_grid[:gh, :gw]
    img = Image.fromarray((np.clip(valid, 0.0, 1.0) * 255).astype(np.uint8))
    img = img.resize((orig_w, orig_h), Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


class LetterboxPreprocess:
    """Letterbox preprocessing of images (ImageNet-normalised) and masks for DINOv2."""

    def __init__(self, dino_size: int = 518):
        self.dino_size = dino_size
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    def image_for_dino(self, img: Image.Image) -> torch.Tensor:
        return self.normalize(self.to_tensor(letterbox_image(img, self.dino_size)))

    def mask_for_dino(self, mask_np: np.ndarray) -> torch.Tensor:
        return letterbox_mask(mask_np, self.dino_size)
