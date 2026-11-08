"""Tests for ``fs-dino-hq make-episodes`` (fs_dino_hq.data.episode_plans).

All data are tiny random images with LabelMe JSON written to tmp_path, named
``real-<scene>-<camera>-<uid>`` like the training dataset.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fs_dino_hq.data import episode_plans as ep

# class -> {scene id: number of images}; walls has two equal-sized scenes (tie rank).
SCENES = {
    "beams": {101: 6, 102: 3, 103: 4, 104: 9, 105: 5},
    "columns": {201: 4, 202: 4, 203: 7, 204: 2, 205: 6, 206: 3},
    "slabs": {301: 10, 302: 5, 303: 8},
    "walls": {401: 5, 402: 5, 403: 6, 404: 7},
}
ALL_FILES = ("synthetic_train_manifest.csv", "synthetic_val_manifest.csv",
             "train_episode_plan.csv", "val_episode_plan.csv", "dataset_inventory.json",
             "invalid_files.json", "episode_coverage.json", "audit_config.json",
             "split_audit.json")
QUERY_COLS = ("cycle", "component", "query_image", "query_json", "query_group", "episode_idx")


def _image(path: Path, rng, size=(8, 6)):
    w, h = size
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(path)


def _labelme(path: Path, image_name, shapes, size=(8, 6), **extra):
    w, h = size
    d = {"version": "5.0.1", "shapes": shapes, "imagePath": image_name,
         "imageData": None, "imageHeight": h, "imageWidth": w}
    d.update(extra)
    path.write_text(json.dumps(d))


def _shapes(i):
    tri = {"label": "spacing", "points": [[0, 0], [5, 0], [0, 4]], "shape_type": "polygon"}
    quad = {"label": "intersection", "points": [[1, 1], [6, 1], [6, 5], [1, 5]],
            "shape_type": "polygon"}
    line = {"label": "spacing", "points": [[0, 0], [3, 3]], "shape_type": "line"}
    return [tri, quad, line][: 1 + i % 3]


def make_dataset(root: Path, seed: int = 0) -> Path:
    """Valid pairs for SCENES plus one case of every invalid kind."""
    rng = np.random.default_rng(seed)
    uid = 1000
    for cls, scenes in SCENES.items():
        d = root / cls
        d.mkdir(parents=True)
        for scene, n in scenes.items():
            for j in range(n):
                uid += 7
                # mixed separators and case: the default pattern accepts both
                stem = (f"real-{scene}-{j % 3}-{uid}" if j % 4 else f"REAL_{scene}_{j}_{uid}")
                _image(d / f"{stem}.png", rng)
                _labelme(d / f"{stem}.json", f"{stem}.png", _shapes(j))
    b, c, s, w = (root / k for k in ("beams", "columns", "slabs", "walls"))
    (b / "real-101-9-1.json").write_text("{not json")               # malformed
    _image(b / "real-101-9-1.png", rng)                               # -> image without JSON
    _image(b / "real-101-9-2.png", rng)                               # image without JSON
    _labelme(c / "real-201-8-3.json", "real-201-8-3.png", _shapes(0))  # image not found
    first_col = sorted(p.name for p in c.glob("real-201-*.png"))[0]
    _labelme(c / "real-201-9-4.json", first_col, _shapes(0))          # duplicate image
    (s / "real-303-1-5.png").write_bytes(b"this is not an image")      # corrupt image
    _labelme(s / "real-303-1-5.json", "real-303-1-5.png", _shapes(1))
    _labelme(w / "real-401-7-6.json", "real-401-7-6.png", [])          # empty mask
    _image(w / "real-401-7-6.png", rng)
    _image(w / "real-402-7-7.png", rng)                               # missing image size
    _labelme(w / "real-402-7-7.json", "real-402-7-7.png", _shapes(0), imageHeight=None)
    return root


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("photo"))


def run(root, out, *flags):
    argv = ["--synthetic-root", str(root), "--out-dir", str(out), *map(str, flags)]
    assert ep.main(argv) == 0
    return Path(out)


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def header(path):
    with open(path, newline="") as f:
        return next(csv.reader(f))


def supports(row, k):
    return [row[f"support_{i}_image"] for i in range(1, k + 1)]


# ---------------------------------------------------------------------------------------------

def test_outputs_and_column_order(data_root, tmp_path):
    out = run(data_root, tmp_path / "o", "--n-shot", 3, "--cycles", 2, "--val-frac", 0.2)
    assert sorted(p.name for p in out.iterdir()) == sorted(ALL_FILES)
    assert header(out / "synthetic_train_manifest.csv") == [
        "component", "image_path", "json_path", "group", "n_shapes", "width", "height", "split"]
    expect = (["cycle", "component", "query_image", "query_json", "query_group"]
              + [f"support_{i}_image" for i in (1, 2, 3)]
              + [f"support_{i}_json" for i in (1, 2, 3)]
              + [f"support_{i}_group" for i in (1, 2, 3)] + ["episode_idx"])
    assert header(out / "train_episode_plan.csv") == expect
    assert header(out / "val_episode_plan.csv") == expect
    raw = (out / "train_episode_plan.csv").read_bytes()
    assert raw.count(b"\r\n") == len(rows(out / "train_episode_plan.csv")) + 1
    cfg = json.loads((out / "audit_config.json").read_text())
    assert cfg["n_shot"] == 3 and cfg["group_mode"] == "scene"
    assert "synthetic_root" not in cfg and "out_dir" not in cfg
    for name in ALL_FILES:                                  # no absolute paths anywhere
        assert str(data_root) not in (out / name).read_text()


def test_two_runs_are_byte_identical(data_root, tmp_path):
    a = run(data_root, tmp_path / "a", "--n-shot", 5, "--cycles", 3)
    b = run(data_root, tmp_path / "b", "--n-shot", 5, "--cycles", 3)
    for name in ALL_FILES:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    c = run(data_root, tmp_path / "c", "--n-shot", 5, "--cycles", 3, "--seed", 7)
    assert (c / "train_episode_plan.csv").read_bytes() != (a / "train_episode_plan.csv").read_bytes()


def test_split_is_scene_disjoint_and_complete(data_root, tmp_path):
    out = run(data_root, tmp_path / "o", "--n-shot", 5, "--val-frac", 0.2)
    tr, va = rows(out / "synthetic_train_manifest.csv"), rows(out / "synthetic_val_manifest.csv")
    assert {r["split"] for r in tr} == {"train"} and {r["split"] for r in va} == {"val"}
    for cls, scenes in SCENES.items():
        trg = {r["group"] for r in tr if r["component"] == cls}
        vag = {r["group"] for r in va if r["component"] == cls}
        assert trg and vag and not trg & vag
        assert trg | vag == {f"{cls}/{s}" for s in scenes}
        n_train = sum(r["component"] == cls for r in tr)
        assert n_train >= 5 + 1
    for r in tr + va:                                       # group = <class>/<scene from stem>
        scene = Path(r["image_path"]).stem.replace("_", "-").split("-")[1]
        assert r["group"] == f"{r['component']}/{scene}"
    audit = json.loads((out / "split_audit.json").read_text())
    assert audit["overlap_ok"] is True and audit["overlapping_scenes"] == []
    assert list(audit)[:5] == ["group_mode", "seed", "val_frac", "n_shot", "algorithm"]
    inv = json.loads((out / "dataset_inventory.json").read_text())["inventory"]
    assert inv["_total"]["valid"] == len(tr) + len(va)


@pytest.mark.parametrize("n_shot", [1, 5])
def test_plan_invariants(data_root, tmp_path, n_shot):
    cycles = 3
    out = run(data_root, tmp_path / "o", "--n-shot", n_shot, "--cycles", cycles, "--val-frac", 0.2)
    tr, va = rows(out / "synthetic_train_manifest.csv"), rows(out / "synthetic_val_manifest.csv")
    train_imgs = {r["image_path"]: r for r in tr}
    val_groups = {r["group"] for r in va}
    plan, vplan = rows(out / "train_episode_plan.csv"), rows(out / "val_episode_plan.csv")

    assert len(plan) == cycles * len(tr)
    assert [int(r["episode_idx"]) for r in plan] == list(range(len(plan)))
    for c in range(cycles):
        qs = [r["query_image"] for r in plan if int(r["cycle"]) == c]
        assert Counter(qs) == Counter(list(train_imgs))        # every train image once per cycle
    for r in plan:
        sup = supports(r, n_shot)
        assert len(set(sup)) == n_shot and r["query_image"] not in sup
        for i, s in enumerate(sup, 1):
            assert s in train_imgs and train_imgs[s]["component"] == r["component"]
            assert r[f"support_{i}_group"] == train_imgs[s]["group"] != r["query_group"]
            assert r[f"support_{i}_json"] == train_imgs[s]["json_path"]
    # validation: every val image once, supports from the TRAIN pool of the same class
    assert sorted(r["query_image"] for r in vplan) == sorted(r["image_path"] for r in va)
    assert [int(r["episode_idx"]) for r in vplan] == list(range(len(vplan)))
    assert {r["cycle"] for r in vplan} == {"0"}
    for r in vplan:
        sup = supports(r, n_shot)
        assert len(set(sup)) == n_shot and r["query_image"] not in sup
        assert all(s in train_imgs and train_imgs[s]["component"] == r["component"] for s in sup)
        assert not {r[f"support_{i}_group"] for i in range(1, n_shot + 1)} & val_groups
    # classes are interleaved, not blocked
    comps = [r["component"] for r in plan[: len(tr)]]
    assert sum(a != b for a, b in zip(comps, comps[1:])) > len(SCENES)
    cov = json.loads((out / "episode_coverage.json").read_text())
    assert cov["episodes_total"] == len(plan) and cov["val_episodes"] == len(vplan)


def test_one_and_five_shot_plans_share_queries(data_root, tmp_path):
    p1 = run(data_root, tmp_path / "k1", "--n-shot", 1, "--cycles", 4)
    p5 = run(data_root, tmp_path / "k5", "--n-shot", 5, "--cycles", 4)
    for name in ("synthetic_train_manifest.csv", "synthetic_val_manifest.csv"):
        assert (p1 / name).read_bytes() == (p5 / name).read_bytes()
    for name in ("train_episode_plan.csv", "val_episode_plan.csv"):
        a, b = rows(p1 / name), rows(p5 / name)
        assert [[r[c] for c in QUERY_COLS] for r in a] == [[r[c] for c in QUERY_COLS] for r in b]
        assert any(r["support_1_image"] != s["support_1_image"] for r, s in zip(a, b))


def test_fewer_cycles_are_a_prefix(data_root, tmp_path):
    short = run(data_root, tmp_path / "c2", "--n-shot", 5, "--cycles", 2)
    long = run(data_root, tmp_path / "c5", "--n-shot", 5, "--cycles", 5)
    s = (short / "train_episode_plan.csv").read_bytes()
    assert (long / "train_episode_plan.csv").read_bytes().startswith(s)
    assert (short / "val_episode_plan.csv").read_bytes() == (long / "val_episode_plan.csv").read_bytes()


def test_suffix_renames_without_changing_contents(data_root, tmp_path):
    plain = run(data_root, tmp_path / "plain", "--n-shot", 5)
    named = run(data_root, tmp_path / "named", "--n-shot", 5, "--suffix", "_5shot")
    renamed = {"train_episode_plan.csv": "train_episode_plan_5shot.csv",
               "val_episode_plan.csv": "val_episode_plan_5shot.csv",
               "episode_coverage.json": "episode_coverage_5shot.json",
               "audit_config.json": "audit_config_5shot.json",
               "split_audit.json": "split_audit_5shot.json"}
    assert sorted(p.name for p in named.iterdir()) == sorted(renamed.get(n, n) for n in ALL_FILES)
    for name in ALL_FILES:
        assert (plain / name).read_bytes() == (named / renamed.get(name, name)).read_bytes(), name


def test_standard_set_in_one_directory(data_root, tmp_path):
    out = tmp_path / "manifests"
    run(data_root, out, "--n-shot", 1, "--cycles", 4, "--suffix", "_1shot")
    run(data_root, out, "--n-shot", 5, "--cycles", 2, "--suffix", "_5shot")
    run(data_root, out, "--n-shot", 5, "--cycles", 4, "--suffix", "_5shot_4cyc")
    for name in ("train_episode_plan_1shot.csv", "val_episode_plan_1shot.csv",
                 "train_episode_plan_5shot.csv", "val_episode_plan_5shot.csv",
                 "train_episode_plan_5shot_4cyc.csv", "split_audit_1shot.json"):
        assert (out / name).is_file()
    assert json.loads((out / "split_audit_1shot.json").read_text())["n_shot"] == 1
    assert ((out / "val_episode_plan_5shot.csv").read_bytes()
            == (out / "val_episode_plan_5shot_4cyc.csv").read_bytes())


def test_changed_split_in_existing_directory_is_refused(tmp_path):
    root = make_dataset(tmp_path / "photo")
    out = run(root, tmp_path / "o", "--n-shot", 5, "--no-verify-images")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(SystemExit, match="different split"):
        ep.main(["--synthetic-root", str(root), "--out-dir", str(out), "--n-shot", "1",
                 "--val-frac", "0.4", "--suffix", "_other"])
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_invalid_files_and_inventory(data_root, tmp_path):
    out = run(data_root, tmp_path / "o", "--expected", "beams=29,columns=29,slabs=25,walls=25")
    inv = json.load(open(out / "invalid_files.json"))
    reasons = {r["json"]: r["reason"] for rs in inv.values() for r in rs}
    assert reasons["beams/real-101-9-1.json"].startswith("malformed json")
    assert reasons["columns/real-201-8-3.json"] == "image not found"
    assert reasons["columns/real-201-9-4.json"].startswith("duplicate: image already paired with columns/")
    assert reasons["slabs/real-303-1-5.json"].startswith("corrupt image")
    assert "slabs/real-303-1-5.png" in reasons["slabs/real-303-1-5.json"]
    assert reasons["walls/real-401-7-6.json"] == "empty mask after label filter"
    assert reasons["walls/real-402-7-7.json"] == "missing imageHeight/imageWidth"
    d = json.load(open(out / "dataset_inventory.json"))
    inv_c = d["inventory"]
    assert inv_c["beams"]["images_without_json"] == 2
    assert inv_c["beams"]["valid_pairs"] == sum(SCENES["beams"].values())
    assert inv_c["walls"]["delta_vs_expected"] == sum(SCENES["walls"].values()) + 2 - 25
    assert inv_c["_total"]["expected"] == 108
    assert set(d["label_distribution"]["slabs"]) == {"spacing", "intersection"}
    assert inv_c["beams"]["grouping_mode"] == "scene" and inv_c["beams"]["n_groups"] == 5


def test_verify_images_flag(data_root, tmp_path):
    a = run(data_root, tmp_path / "v", "--verify-images")
    b = run(data_root, tmp_path / "nv", "--no-verify-images")
    def imgs(o):
        return {r["image_path"] for n in ("synthetic_train_manifest.csv", "synthetic_val_manifest.csv")
                for r in rows(o / n)}
    assert imgs(b) - imgs(a) == {"slabs/real-303-1-5.png"}


def test_label_filter(data_root, tmp_path):
    out = run(data_root, tmp_path / "o", "--label-filter", "intersection")
    inv = json.load(open(out / "invalid_files.json"))
    empty = [r for rs in inv.values() for r in rs if r["reason"] == "empty mask after label filter"]
    assert len(empty) > 10                                    # every spacing-only file
    cfg = json.load(open(out / "audit_config.json"))
    assert cfg["label_filter"] == ["intersection"]


def test_unparsed_stem_aborts_before_writing(tmp_path):
    root = make_dataset(tmp_path / "photo")
    rng = np.random.default_rng(1)
    _image(root / "beams" / "img_0001.png", rng)
    _labelme(root / "beams" / "img_0001.json", "img_0001.png", _shapes(0))
    out = tmp_path / "o"
    with pytest.raises(SystemExit, match="scene parsing failed"):
        ep.main(["--synthetic-root", str(root), "--out-dir", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["scene_parse_failures.json"]
    rep = json.load(open(out / "scene_parse_failures.json"))
    assert rep["failures"]["beams"] == ["beams/img_0001.png"]
    assert rep["regex"] == ep.DEFAULT_SCENE_REGEX


def test_custom_scene_regex(tmp_path):
    rng = np.random.default_rng(2)
    root = tmp_path / "data"
    for cls in ("a", "b"):
        (root / cls).mkdir(parents=True)
        for site in range(4):
            for view in range(3 + site):
                stem = f"site{site}_view{view}"
                _image(root / cls / f"{stem}.jpg", rng)
                _labelme(root / cls / f"{stem}.json", f"{stem}.jpg", _shapes(0))
    out = run(root, tmp_path / "o", "--classes", "a", "b", "--n-shot", 2,
              "--scene-regex", r"^site(?P<scene>\d+)_view\d+$", "--val-frac", 0.2)
    tr, va = rows(out / "synthetic_train_manifest.csv"), rows(out / "synthetic_val_manifest.csv")
    assert {r["group"] for r in tr + va} == {f"{c}/{s}" for c in "ab" for s in range(4)}
    assert not {r["group"] for r in tr} & {r["group"] for r in va}
    assert json.load(open(out / "audit_config.json"))["scene_regex"] == r"^site(?P<scene>\d+)_view\d+$"


def test_bad_options_are_rejected(data_root, tmp_path):
    base = ["--synthetic-root", str(data_root), "--out-dir", str(tmp_path / "o")]
    for flags in (["--scene-regex", r"^real-(\d+)$"], ["--scene-regex", "("], ["--n-shot", "0"],
                  ["--val-frac", "1.5"], ["--suffix", "a/b"], ["--expected", "beams:3"],
                  ["--classes", "beams", "beams"], ["--group-mode", "auto"]):
        with pytest.raises(SystemExit) as e:
            ep.main(base + flags)
        assert e.value.code == 2, flags
    assert not (tmp_path / "o").exists()


def test_image_group_mode(data_root, tmp_path):
    out = run(data_root, tmp_path / "o", "--group-mode", "image", "--n-shot", 3, "--val-frac", 0.2)
    tr, va = rows(out / "synthetic_train_manifest.csv"), rows(out / "synthetic_val_manifest.csv")
    allr = tr + va
    assert len({r["group"] for r in allr}) == len(allr)      # one group per image
    assert all(r["group"] == f"{r['component']}/{Path(r['image_path']).stem}" for r in allr)
    for cls in SCENES:
        n = sum(r["component"] == cls for r in allr)
        assert sum(r["component"] == cls for r in va) == max(1, round(n * 0.2))
    audit = json.load(open(out / "split_audit.json"))
    assert audit["group_mode"] == "image" and "NOT scene-disjoint" in audit["note"]
    assert "scene_parse" not in audit
    plan = rows(out / "train_episode_plan.csv")
    assert all(r["query_image"] not in supports(r, 3) for r in plan)


def test_cli_dispatch(data_root, tmp_path):
    from fs_dino_hq import cli
    out = tmp_path / "o"
    assert cli.main(["make-episodes", "--synthetic-root", str(data_root), "--out-dir", str(out),
                     "--n-shot", "1", "--cycles", "1"]) == 0
    assert (out / "train_episode_plan.csv").is_file()
    r = subprocess.run([sys.executable, "-m", "fs_dino_hq.data.episode_plans", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "--scene-regex" in r.stdout
