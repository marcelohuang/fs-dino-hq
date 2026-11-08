"""Letterbox data pipeline for Stage 1 adapter distillation.

Both views of an image preserve its aspect ratio (longest side to the target
size, top-left placement, padding bottom/right), so the student's token grid
and the teacher's 64x64 embedding grid cover the same image region at every
student resolution:

  DINO view   ``fs_dino_hq.data.letterbox.letterbox_image`` at ``dino_res``
              (BILINEAR, black padding applied BEFORE ImageNet normalisation).
              This is byte-identical to ``LetterboxPreprocess``, i.e. to what
              Stage 2, HQ training and inference feed the adapter, so the
              adapter is distilled under the geometry and padding values it
              later sees.
  SAM view    SAM's own convention (``segment_anything.modeling.Sam.preprocess``):
              longest side to 1024 (BILINEAR), SAM pixel normalisation, then
              zero padding AFTER normalisation. The teacher therefore produces
              the embeddings it was trained to produce for padded inputs; the
              padded region is not black but "mean colour".

The two padding orders are deliberately different: the student must match the
downstream convention; the teacher must match its own.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .letterbox import LetterboxPreprocess, content_box, content_box_on_grid

SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
SAM_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
SAM_INPUT_SIZE = 1024
TEACHER_GRID = 64
DINO_PATCH = 14


def sam_letterbox_native(img: Image.Image, size: int = SAM_INPUT_SIZE) -> torch.Tensor:
    """[3, size, size] float tensor in SAM's input convention (normalise, then zero-pad)."""
    cw, ch = content_box(img.width, img.height, size)
    x = torch.from_numpy(np.array(img.resize((cw, ch), Image.BILINEAR))).permute(2, 0, 1).float()
    x = (x - SAM_PIXEL_MEAN) / SAM_PIXEL_STD
    return F.pad(x, (0, size - cw, 0, size - ch))


def content_mask_on_grid(w: int, h: int, grid: int = TEACHER_GRID) -> torch.Tensor:
    """[1, grid, grid] float mask of the valid (non-padded) cells on the teacher grid."""
    gw, gh = content_box_on_grid(w, h, SAM_INPUT_SIZE, grid)
    m = torch.zeros(1, grid, grid)
    m[:, :gh, :gw] = 1.0
    return m


def read_manifest(path) -> List[str]:
    """Non-empty, stripped lines of a manifest (image paths, relative to the image root)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


class LetterboxDistillDataset(Dataset):
    """One manifest entry -> dict(dino_image, sam_image, content_mask, index).

    ``dino_image`` is [3, dino_res, dino_res], ``sam_image`` [3, 1024, 1024],
    ``content_mask`` [1, 64, 64] (valid cells of the teacher grid) and
    ``index`` the position of the entry in ``entries``.

    ``dino_res`` can be changed between coarse-to-fine stages with
    ``set_dino_res()``; the SAM view is always 1024, so the teacher target is
    the same in every stage.
    """

    def __init__(self, entries: List[str], image_root: Optional[str], dino_res: int,
                 sam_res: int = SAM_INPUT_SIZE, max_images: int = 0):
        self.entries = entries[:max_images] if max_images > 0 else list(entries)
        self.root = Path(image_root) if image_root else None
        self.sam_res = sam_res
        self.set_dino_res(dino_res)

    def set_dino_res(self, dino_res: int) -> None:
        if dino_res % DINO_PATCH != 0:
            raise ValueError(f"DINOv2 ViT-B/14 needs a multiple of {DINO_PATCH}, got {dino_res}")
        self.dino_res = dino_res
        self.prep = LetterboxPreprocess(dino_size=dino_res)

    def path(self, i: int) -> Path:
        p = Path(self.entries[i])
        return p if p.is_absolute() or self.root is None else self.root / p

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Dict:
        img = Image.open(self.path(i)).convert("RGB")
        return {
            "dino_image": self.prep.image_for_dino(img),                 # [3, R, R]
            "sam_image": sam_letterbox_native(img, self.sam_res),         # [3, 1024, 1024]
            "content_mask": content_mask_on_grid(img.width, img.height),  # [1, 64, 64]
            "index": i,
        }
