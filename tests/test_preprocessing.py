"""Letterbox preprocessing, back-mapping of probabilities and LabelMe rasterisation."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from fs_dino_hq.data.labelme import labelme_json_to_mask
from fs_dino_hq.data.letterbox import (LetterboxPreprocess, content_box, content_box_on_grid,
                                       letterbox_image, letterbox_mask, unletterbox_prob)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@pytest.mark.parametrize("w,h,expected", [
    (1280, 720, (518, 291)),      # landscape: 720 * 518 / 1280 = 291.375
    (720, 1280, (291, 518)),      # portrait
    (300, 300, (518, 518)),       # square
    (5171, 3878, (518, 388)),     # 3878 * 518 / 5171 = 388.47
    (64, 48, (518, 388)),         # upscaling: 48 * 518 / 64 = 388.5 -> round half even
])
def test_content_box(w, h, expected):
    assert content_box(w, h, 518) == expected


def test_content_box_on_grid():
    assert content_box_on_grid(1280, 720, 518, 256) == (256, 144)
    assert content_box_on_grid(720, 1280, 518, 256) == (144, 256)
    assert content_box_on_grid(300, 300, 518, 256) == (256, 256)
    assert content_box_on_grid(10000, 10, 518, 256) == (256, 1)      # never empty


@pytest.mark.parametrize("w,h", [(160, 90), (90, 160), (70, 70)])
def test_letterbox_image_top_left_black_pad(w, h):
    img = Image.new("RGB", (w, h), (200, 100, 50))
    out = letterbox_image(img, 518)
    assert out.size == (518, 518) and out.mode == "RGB"
    a = np.array(out)
    cw, ch = content_box(w, h, 518)
    assert (a[:ch, :cw] == [200, 100, 50]).all()          # constant image stays constant
    assert (a[ch:, :] == 0).all() and (a[:, cw:] == 0).all()


def test_letterbox_image_uses_pil_bilinear():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (45, 80, 3), dtype=np.uint8))
    out = np.array(letterbox_image(img, 518))
    cw, ch = content_box(80, 45, 518)
    ref = np.array(img.resize((cw, ch), resample=Image.BILINEAR))
    assert np.array_equal(out[:ch, :cw], ref)


def test_image_for_dino_pads_black_before_normalisation():
    img = Image.new("RGB", (160, 90), (255, 255, 255))
    t = LetterboxPreprocess(518).image_for_dino(img)
    assert t.shape == (3, 518, 518) and t.dtype == torch.float32
    cw, ch = content_box(160, 90, 518)
    pad_value = (0.0 - MEAN) / STD                         # black, then normalised
    white_value = (1.0 - MEAN) / STD
    assert torch.allclose(t[:, ch:, :], pad_value.expand(3, 518 - ch, 518), atol=1e-6)
    assert torch.allclose(t[:, :ch, :cw], white_value.expand(3, ch, cw), atol=1e-6)


def test_letterbox_mask_nearest_and_binary():
    m = np.zeros((90, 160), np.uint8)
    m[10:40, 20:100] = 3            # any non-zero value is foreground
    m[60:62, :] = 255
    t = letterbox_mask(m, 518)
    assert t.shape == (1, 518, 518) and t.dtype == torch.float32
    assert set(torch.unique(t).tolist()) <= {0.0, 1.0}
    cw, ch = content_box(160, 90, 518)
    ref = np.array(Image.fromarray(m).resize((cw, ch), resample=Image.NEAREST)) > 0
    assert np.array_equal(t[0, :ch, :cw].numpy().astype(bool), ref)
    assert t[0, ch:, :].sum() == 0 and t[0, :, cw:].sum() == 0
    # same result through the preprocessing object
    assert torch.equal(LetterboxPreprocess(518).mask_for_dino(m), t)


@pytest.mark.parametrize("w,h", [(1280, 720), (720, 1280), (300, 300), (64, 48)])
def test_unletterbox_prob_crops_content_and_restores_size(w, h):
    grid = np.ones((256, 256), np.float32)          # padding region = 1.0 (must not leak)
    gw, gh = content_box_on_grid(w, h, 518, 256)
    grid[:gh, :gw] = 0.8
    out = unletterbox_prob(grid, w, h, 518)
    assert out.shape == (h, w) and out.dtype == np.float32
    assert np.all(out == np.float32(204) / 255)     # floor(0.8 * 255) = 204, nothing else


def test_unletterbox_prob_quantises_to_1_over_255_by_truncation():
    rng = np.random.default_rng(1)
    grid = rng.random((256, 256)).astype(np.float32)
    out = unletterbox_prob(grid, 1280, 720, 518)
    k = out * 255
    assert np.array_equal(k, np.round(k))
    assert out.min() >= 0 and out.max() <= 1
    # truncation, not rounding: 0.999 -> 254/255, 0.5 -> 127/255 (< 0.5)
    for p, expected in ((0.999, 254), (0.5, 127), (1.0, 255), (0.0, 0), (1.7, 255), (-0.2, 0)):
        g = np.full((256, 256), p, np.float32)
        assert np.all(unletterbox_prob(g, 64, 48, 518) == np.float32(expected) / 255)


def test_unletterbox_prob_matches_reference_pipeline():
    rng = np.random.default_rng(2)
    grid = rng.random((256, 256)).astype(np.float32)
    w, h = 1280, 720
    ref = Image.fromarray((np.clip(grid[:144, :256], 0, 1) * 255).astype(np.uint8))
    ref = np.array(ref.resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0
    assert np.array_equal(unletterbox_prob(grid, w, h, 518), ref)


def test_eight_bit_round_trip_and_threshold():
    """Saved PNG value floor(p * 255) recovers k exactly; p >= 0.5 <=> k >= 128."""
    k = np.arange(256, dtype=np.uint8)
    p = k.astype(np.float32) / 255.0
    assert np.array_equal((p * 255).astype(np.uint8), k)
    assert np.array_equal(p >= 0.5, k >= 128)


# ── LabelMe ──────────────────────────────────────────────────────────────────

def _write_labelme(path, shapes, w=40, h=30):
    path.write_text(json.dumps({"imageWidth": w, "imageHeight": h, "imagePath": "x.png",
                                "shapes": shapes}))
    return path


@pytest.fixture
def labelme_file(tmp_path):
    return _write_labelme(tmp_path / "a.json", [
        {"label": "spacing", "points": [[2, 2], [10, 2], [10, 8], [2, 8]], "shape_type": "polygon"},
        {"label": "intersection", "points": [[20, 15], [30, 15], [30, 25], [20, 25]]},
        {"label": "spacing", "points": [[35, 1], [38, 4]], "shape_type": "rectangle"},  # 2 points
        {"label": "spacing", "points": [[0, 20], [5, 25]], "shape_type": "line"},
    ])


def test_labelme_all_labels(labelme_file):
    m = labelme_json_to_mask(labelme_file)
    assert m.shape == (30, 40) and m.dtype == np.uint8
    assert set(np.unique(m)) == {0, 1}
    assert m[2:9, 2:11].all() and m[15:26, 20:31].all()        # outline included
    assert m.sum() == 7 * 9 + 11 * 11                          # shapes with < 3 points skipped
    assert m[1:5, 35:39].sum() == 0


def test_labelme_label_filter(labelme_file):
    sp = labelme_json_to_mask(labelme_file, ["spacing"])
    it = labelme_json_to_mask(labelme_file, ["intersection"])
    both = labelme_json_to_mask(labelme_file, ["spacing", "intersection"])
    assert sp.sum() == 63 and it.sum() == 121
    assert np.array_equal(both, labelme_json_to_mask(labelme_file, None))
    assert labelme_json_to_mask(labelme_file, []).sum() == 0   # empty list = nothing
    assert labelme_json_to_mask(labelme_file, ["missing"]).sum() == 0


def test_labelme_size_from_json_and_pil_polygon(tmp_path):
    pts = [[3.4, 1.2], [17.8, 4.6], [12.1, 19.9], [1.0, 14.5]]
    f = _write_labelme(tmp_path / "b.json", [{"label": "x", "points": pts}], w=25, h=22)
    m = labelme_json_to_mask(f)
    assert m.shape == (22, 25)
    from PIL import ImageDraw
    ref = Image.new("L", (25, 22), 0)
    ImageDraw.Draw(ref).polygon([tuple(p) for p in pts], outline=1, fill=1)
    assert np.array_equal(m, np.array(ref))
    empty = _write_labelme(tmp_path / "c.json", [], w=7, h=5)
    assert labelme_json_to_mask(empty).shape == (5, 7) and labelme_json_to_mask(empty).sum() == 0
