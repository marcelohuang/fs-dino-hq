"""LabelMe annotation helpers.

Masks are LabelMe polygon JSON files stored next to their images. All
polygons are merged into one foreground class unless a label filter is given.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Image file extensions recognised next to LabelMe annotations.
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def open_image_rgb(image) -> Image.Image:
    """Open an image (path or PIL image) as RGB in the frame LabelMe annotates.

    LabelMe applies the EXIF orientation tag before annotation, so the same is
    done here (``ImageOps.exif_transpose``); images without an orientation tag,
    such as PNG files, are returned unchanged.
    """
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    with Image.open(image) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def labelme_json_to_mask(json_path, label_filter=None) -> np.ndarray:
    """Rasterise all polygons of a LabelMe JSON into a uint8 {0,1} mask.

    label_filter=None keeps every shape. A list keeps only shapes whose label
    is in the list; an empty list therefore yields an empty mask.
    Only the ``points`` of each shape are used: every shape with at least
    three points is filled as a polygon with PIL's rasteriser (outline=1,
    fill=1); shapes with fewer points (for example two-point rectangles,
    circles or lines) are skipped.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    h, w = int(data["imageHeight"]), int(data["imageWidth"])
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    for shape in data.get("shapes", []):
        if label_filter is not None and shape.get("label", "") not in label_filter:
            continue
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        draw.polygon([(float(x), float(y)) for x, y in pts], outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8)
