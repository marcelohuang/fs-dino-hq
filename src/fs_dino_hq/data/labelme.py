"""LabelMe annotation helpers.

Masks are LabelMe polygon JSON files stored next to their images. All
polygons are merged into one foreground class unless a label filter is given.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

# A set, as in the original scripts: iteration order only matters when several
# image files share a stem, which never happens in the project datasets.
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


def find_image_json_pairs(folder: Path, image_relative_to: str = "folder"
                          ) -> List[Tuple[Path, Path]]:
    """(image, json) pairs for every ``*.json`` under ``folder`` (recursive, sorted).

    The image is located from the JSON's ``imagePath`` field, falling back to
    ``<stem><ext>`` for ext in IMG_EXTENSIONS. Both lookups are anchored at
    ``folder`` (``image_relative_to="folder"``, the trainer and k-shot
    inference convention) or at the JSON's own directory
    (``image_relative_to="json"``, the training-set verification convention).
    The two agree for flat directories.
    """
    folder = Path(folder)
    pairs: List[Tuple[Path, Path]] = []
    for jf in sorted(folder.rglob("*.json")):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        anchor = folder if image_relative_to == "folder" else jf.parent
        img_path: Optional[Path] = None
        ip = data.get("imagePath")
        if ip:
            c = anchor / ip
            if c.exists():
                img_path = c
        if img_path is None:
            for ext in IMG_EXTENSIONS:
                c = anchor / f"{jf.stem}{ext}"
                if c.exists():
                    img_path = c
                    break
        if img_path is not None:
            pairs.append((img_path, jf))
    return pairs


def find_pairs_flat(folder: Path) -> List[Tuple[Path, Path]]:
    """Non-recursive pairing used by the k-shot real inference protocol.

    Only the JSON's own stem is tried, with a fixed extension preference
    (.png, .jpg, .jpeg, .PNG, .JPG); ``imagePath`` is ignored.
    """
    out = []
    for jf in sorted(Path(folder).glob("*.json")):
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
            img = jf.with_suffix(ext)
            if img.exists():
                out.append((img, jf))
                break
    return out


def find_query_pairs(folder: Path) -> List[Tuple[Path, Optional[Path]]]:
    """Every image under ``folder`` with its JSON if one exists (unlabelled queries allowed)."""
    out = []
    for img_path in sorted(p for p in Path(folder).rglob("*")
                           if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS):
        jp = img_path.with_suffix(".json")
        out.append((img_path, jp if jp.exists() else None))
    return out
