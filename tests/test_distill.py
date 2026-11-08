"""Stage 1 distillation: letterbox views, split tool, CLI/config handling and CPU runs with stub backbones."""
from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from stubs import StubDino

import fs_dino_hq.training.distill as distill
from fs_dino_hq.data import distill_split
from fs_dino_hq.data.distill_data import (LetterboxDistillDataset, SAM_PIXEL_MEAN, SAM_PIXEL_STD,
                                          content_mask_on_grid, read_manifest, sam_letterbox_native)
from fs_dino_hq.data.letterbox import LetterboxPreprocess, content_box
from fs_dino_hq.modeling.distill_adapter import BottleneckAdapter
from fs_dino_hq.utils.seeding import seed_everything

HISTORY_COLUMNS = ["stage", "res", "epoch_in_stage", "global_epoch", "event", "adapter_sha16",
                   "diag_mse", "diag_normalized_mse", "diag_cosine", "diag_mse_content",
                   "diag_normalized_mse_content", "diag_cosine_content", "diag_student_norm",
                   "diag_teacher_norm", "diag_loss", "lr", "train_loss", "val_loss", "val_cosine",
                   "val_cosine_content", "seconds"]
CKPT_KEYS = {"config", "epoch", "stage", "res", "epoch_in_stage", "val_mse", "val_cosine", "diag", "stage_val"}


# ── stub teacher ─────────────────────────────────────────────────────────────

class TinySamEncoder(nn.Module):
    """Stand-in for the SAM ViT-H image encoder: [B,3,1024,1024] -> BHWC tokens on the 64x64 grid
    -> ``blocks`` (BHWC in and out, hookable) -> neck [B,256,64,64]."""

    def __init__(self, dim: int = 320, n_blocks: int = 8) -> None:
        super().__init__()
        self.embed = nn.Linear(48, dim)
        self.blocks = nn.ModuleList([nn.Tanh() for _ in range(n_blocks)])
        self.neck = nn.Conv2d(dim, 256, kernel_size=1)

    def forward(self, x):
        x = F.pixel_unshuffle(F.avg_pool2d(x, 4), 4)          # [B,48,64,64]
        x = self.embed(x.permute(0, 2, 3, 1))                  # [B,64,64,dim]
        for blk in self.blocks:
            x = blk(x)
        return self.neck(x.permute(0, 3, 1, 2))


class TinySam(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = TinySamEncoder()
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)


@pytest.fixture
def stub_backbones(monkeypatch):
    """Replace the DINOv2 / SAM loaders of the trainer; records the construction order."""
    events = []

    def fake_dino(ckpt, device=None, *a, **k):
        events.append("dino")
        return StubDino().to(device)

    def fake_sam(ckpt, device=None, *a, **k):
        events.append("sam")
        return TinySam().to(device)

    class RecordingAdapter(BottleneckAdapter):
        def __init__(self, *a, **k):
            events.append("adapter")
            super().__init__(*a, **k)

    real_fit = distill.fit_pca_teacher

    def recording_fit(*a, **k):
        events.append("pca_fit")
        return real_fit(*a, **k)

    monkeypatch.setattr(distill, "load_dinov2_vitb14", fake_dino)
    monkeypatch.setattr(distill, "load_sam_vit_h", fake_sam)
    monkeypatch.setattr(distill, "BottleneckAdapter", RecordingAdapter)
    monkeypatch.setattr(distill, "fit_pca_teacher", recording_fit)
    return events


def make_pool(root, n_train=8, n_val=4, seed=0):
    """Random images of several aspect ratios, manifests relative to root/images, dummy checkpoint files."""
    rng = np.random.default_rng(seed)
    img_dir = root / "images"
    img_dir.mkdir(parents=True)
    sizes = [(64, 48), (40, 60), (56, 56), (72, 40)]
    names = []
    for i in range(n_train + n_val):
        w, h = sizes[i % len(sizes)]
        name = f"seq{i // 3}_{i:04d}.png"
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(img_dir / name)
        names.append(name)
    (root / "train.txt").write_text("\n".join(names[:n_train]) + "\n")
    (root / "val.txt").write_text("\n".join(names[n_train:]) + "\n")
    (root / "dino.pth").write_bytes(b"dino")
    (root / "sam.pth").write_bytes(b"sam")
    return root


def base_args(pool, save_dir, adapter="final", stages="28:3,42:2", extra=()):
    return ["--adapter", adapter, "--train-manifest", str(pool / "train.txt"),
            "--val-manifest", str(pool / "val.txt"), "--image-root", str(pool / "images"),
            "--dino-ckpt", str(pool / "dino.pth"), "--sam-ckpt", str(pool / "sam.pth"),
            "--save-dir", str(save_dir), "--stages", stages, "--diag-res", "42",
            "--batch-size", "2", "--num-workers", "0", "--print-every", "1000", *extra]


def read_history(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def without_seconds(rows):
    return [{k: v for k, v in r.items() if k != "seconds"} for r in rows]


def states_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ── letterbox views ──────────────────────────────────────────────────────────

def test_sam_view_follows_sam_preprocess_convention():
    img = Image.new("RGB", (400, 300), (200, 100, 50))
    x = sam_letterbox_native(img)
    assert x.shape == (3, 1024, 1024)
    assert content_box(400, 300, 1024) == (1024, 768)
    expected = (torch.tensor([200.0, 100.0, 50.0]).view(3, 1, 1) - SAM_PIXEL_MEAN) / SAM_PIXEL_STD
    assert torch.allclose(x[:, :768, :], expected.expand(3, 768, 1024), atol=1e-5)
    assert torch.equal(x[:, 768:, :], torch.zeros(3, 256, 1024))   # zero AFTER normalisation


def test_dino_view_matches_downstream_preprocess_and_pad_is_black_before_norm(tmp_path):
    img = Image.new("RGB", (400, 300), (200, 100, 50)); img.save(tmp_path / "a.png")
    x = LetterboxDistillDataset([str(tmp_path / "a.png")], None, 252)[0]["dino_image"]
    assert torch.equal(x, LetterboxPreprocess(252).image_for_dino(img))
    pad_value = (0 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(x[:, 200:, 0], pad_value.view(3, 1).expand(3, 52), atol=1e-5)


def test_teacher_and_student_cover_the_same_region_at_every_resolution(tmp_path):
    # a 16:9 image: content occupies 36 of 64 teacher rows at every student resolution
    Image.new("RGB", (1280, 720)).save(tmp_path / "a.png")
    (tmp_path / "m.txt").write_text("a.png\n\n")
    ds = LetterboxDistillDataset(read_manifest(tmp_path / "m.txt"), str(tmp_path), 126)
    for res in (126, 252, 518):
        ds.set_dino_res(res)
        item = ds[0]
        assert set(item) == {"dino_image", "sam_image", "content_mask", "index"} and item["index"] == 0
        assert item["dino_image"].shape == (3, res, res)
        assert item["sam_image"].shape == (3, 1024, 1024)
        assert item["content_mask"].shape == (1, 64, 64) and item["content_mask"].sum() == 64 * 36
        gw, gh = content_box(1280, 720, res)
        assert gw == res and abs(gh / res - 36 / 64) < 1 / res
    m = content_mask_on_grid(720, 1280)
    assert m[0, :, :36].all() and not m[0, :, 36:].any()
    with pytest.raises(ValueError):
        ds.set_dino_res(128)


# ── CLI and config ───────────────────────────────────────────────────────────

def test_parse_stages():
    assert distill.parse_stages("126:8,252:8,518:14") == [(126, 8), (252, 8), (518, 14)]
    for bad in ("128:1", "126:0"):
        with pytest.raises(ValueError):
            distill.parse_stages(bad)


def _cfg(tmp_path, **kw):
    cfg = {"adapter": "early", "train_manifest": "t.txt", "val_manifest": "v.txt", "dino_ckpt": "d.pth",
           "sam_ckpt": "s.pth", "save_dir": str(tmp_path / "run"), "stages": "126:2,252:2", "lr": 1e-3, **kw}
    p = tmp_path / "c.json"; p.write_text(json.dumps(cfg))
    return p


def test_config_fills_values_and_command_line_wins(tmp_path, monkeypatch):
    p = _cfg(tmp_path)
    a = distill.parse_args(["--config", str(p), "--stages", "126:1"])
    assert a.adapter == "early" and a.lr == 1e-3 and a.stages == "126:1"
    assert distill.STATE_KEY[a.adapter] == "early_adapter_state"
    # an explicit command-line value wins even when it equals the parser default
    assert distill.parse_args(["--config", str(p), "--lr", "5e-4"]).lr == 5e-4
    # ${VAR} expansion in string values
    monkeypatch.setenv("RUN_ROOT", str(tmp_path / "runs"))
    p = _cfg(tmp_path, save_dir="${RUN_ROOT}/early")
    assert distill.parse_args(["--config", str(p)]).save_dir == str(tmp_path / "runs" / "early")
    # removed options are unknown keys
    for removed in ("compare_ckpt", "loss", "grad_clip", "init_adapter_ckpt"):
        with pytest.raises(ValueError):
            distill.parse_args(["--config", str(_cfg(tmp_path, **{removed: 1}))])


def test_required_values_and_smoke(tmp_path):
    with pytest.raises(SystemExit):
        distill.parse_args(["--adapter", "final"])
    p = _cfg(tmp_path, adapter="final", stages="126:8,252:8,518:14")
    a = distill.parse_args(["--config", str(p), "--smoke"])
    assert a.stages == "126:1,252:1,518:1" and a.max_train_images == 8 and a.max_val_images == 4
    assert a.batch_size == 2 and a.pca_fit_images == 4
    with pytest.raises(SystemExit):
        distill.parse_args(["--config", str(p), "--stages", "128:2"])


def test_command_is_registered():
    from fs_dino_hq import cli
    for cmd in ("distill", "make-distill-split"):
        with pytest.raises(SystemExit) as e:
            cli.main([cmd, "--help"])
        assert e.value.code == 0


def test_warmup_cosine_schedule():
    opt = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=5e-4)
    sched, warmup = distill.build_warmup_cosine(opt, 8, 1)
    lrs = []
    for _ in range(8):
        lrs.append(opt.param_groups[0]["lr"]); opt.step(); sched.step()
    assert warmup == 1 and math.isclose(lrs[0], 5e-6) and math.isclose(lrs[1], 5e-4)
    assert math.isclose(lrs[-1], 2.48e-5, rel_tol=1e-2)


# ── stage-best bookkeeping on resume ─────────────────────────────────────────

def test_restore_stage_best(tmp_path):
    assert distill.restore_stage_best({"best_stage": 0.123}, tmp_path, "final_adapter", 518) == 0.123
    assert distill.restore_stage_best({}, tmp_path, "early_adapter", 252) == math.inf
    # a checkpoint without the key: the stage-best file on disk carries the loss to beat
    torch.save({"adapter_state": {}, "stage_val": {"loss": 0.0034}, "stage": 2, "res": 518, "epoch_in_stage": 5},
               tmp_path / "final_adapter_stage518_best.pt")
    assert distill.restore_stage_best({"stage_idx": 2}, tmp_path, "final_adapter", 518) == 0.0034


def test_checkpoint_provenance_fields():
    ck = {"stage": 1, "res": 252, "epoch_in_stage": 3, "epoch": 11, "selected_by": "min diag@518 mse", "x": 1}
    assert distill.checkpoint_provenance(ck) == {"stage": 1, "res": 252, "epoch_in_stage": 3, "epoch": 11,
                                                 "selected_by": "min diag@518 mse"}


# ── end-to-end runs with stub backbones ──────────────────────────────────────

def test_smoke_run_final_adapter(tmp_path, stub_backbones):
    pool = make_pool(tmp_path / "pool")
    run = tmp_path / "run"
    assert distill.main(base_args(pool, run, extra=["--smoke"])) == 0
    assert stub_backbones == ["dino", "sam", "adapter"]

    # the first history row records the adapter right after seeded construction
    seed_everything(42, deterministic_cudnn=True)
    StubDino(); TinySam()
    expected_sha = distill.adapter_sha256(BottleneckAdapter(in_channels=768, out_channels=256))[:16]

    cols, rows = read_history(run / "history.csv")
    assert cols == HISTORY_COLUMNS
    assert [(r["event"], r["res"]) for r in rows] == [("stage_start", "28"), ("epoch", "28"),
                                                      ("stage_start", "42"), ("epoch", "42")]
    assert rows[0]["adapter_sha16"] == expected_sha
    assert rows[1]["lr"] == str(5e-4)               # 1-epoch stages: no warm-up epoch

    names = {p.name for p in run.iterdir()}
    assert {"config.json", "history.csv", "c2f_last.pt", "results.json", "final_adapter_best.pt",
            "final_adapter_stage28_best.pt", "final_adapter_stage28_last.pt",
            "final_adapter_stage42_best.pt", "final_adapter_stage42_last.pt"} <= names
    assert "pca_teacher_1280to256.pt" not in names
    best = torch.load(run / "final_adapter_best.pt", map_location="cpu", weights_only=False)
    assert set(best) == CKPT_KEYS | {"adapter_state", "selected_by"}
    assert best["selected_by"] == "min diag@42 mse" and best["config"]["adapter"] == "final"
    assert sum(".layer." in k for k in best["adapter_state"]) == 24 and len(best["adapter_state"]) == 44
    stage_ck = torch.load(run / "final_adapter_stage28_last.pt", map_location="cpu", weights_only=False)
    assert set(stage_ck) == CKPT_KEYS | {"adapter_state"}
    last = torch.load(run / "c2f_last.pt", map_location="cpu", weights_only=False)
    assert set(last) == {"adapter_state", "optimizer_state", "scheduler_state", "stage_idx", "epoch_in_stage",
                         "history", "best_diag", "best_stage", "config", "rng"}
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["loss"] == "mse" and cfg["state_key"] == "adapter_state" and cfg["n_train"] == 8
    assert set(cfg["sha256"]) == {"dino_ckpt", "sam_ckpt", "train_manifest", "val_manifest"}
    res = json.loads((run / "results.json").read_text())
    assert {"diag_res", "n_diag_images", "selected_best", "stage_final_best", "stage_final_last",
            "stage_last_checkpoints", "selection_note", "checkpoints", "adapter", "stages", "loss",
            "best_diag_mse_during_training", "history_rows"} == set(res)
    assert res["history_rows"] == 4 and set(res["stage_last_checkpoints"]) == {"28", "42"}
    assert math.isclose(res["selected_best"]["diag"]["mse"], res["best_diag_mse_during_training"], rel_tol=1e-6)


def test_early_adapter_fits_and_reuses_the_pca_teacher(tmp_path, stub_backbones, monkeypatch):
    pool = make_pool(tmp_path / "pool", n_train=4, n_val=2)
    run = tmp_path / "early"
    assert distill.main(base_args(pool, run, adapter="early", extra=["--smoke"])) == 0
    assert stub_backbones == ["dino", "sam", "adapter", "pca_fit"]     # PCA fit after the adapter
    pca = torch.load(run / "pca_teacher_1280to256.pt", map_location="cpu", weights_only=True)
    assert pca["W"].shape == (256, 320) and pca["mean"].shape == (320,)
    assert pca["out_dim"] == 256 and len(pca["fit_images"]) == 4
    assert pca["geometry"] == "letterbox_1024_sam_native_pad" and 0.0 < pca["evr"] <= 1.0
    best = torch.load(run / "early_adapter_best.pt", map_location="cpu", weights_only=False)
    assert "early_adapter_state" in best and "adapter_state" not in best
    assert json.loads((run / "config.json").read_text())["pca_evr"] == pytest.approx(pca["evr"])

    # --pca-cache: no refit, the basis is copied into the new run folder, same teacher -> same run
    def no_fit(*a, **k):
        raise AssertionError("PCA refitted although --pca-cache was given")
    monkeypatch.setattr(distill, "fit_pca_teacher", no_fit)
    run2 = tmp_path / "early_cached"
    assert distill.main(base_args(pool, run2, adapter="early",
                                  extra=["--smoke", "--pca-cache", str(run / "pca_teacher_1280to256.pt")])) == 0
    assert (run2 / "pca_teacher_1280to256.pt").read_bytes() == (run / "pca_teacher_1280to256.pt").read_bytes()
    assert without_seconds(read_history(run2 / "history.csv")[1]) == without_seconds(read_history(run / "history.csv")[1])


def test_resume_reproduces_the_uninterrupted_run(tmp_path, stub_backbones, monkeypatch):
    pool = make_pool(tmp_path / "pool", n_train=4, n_val=2)

    full = tmp_path / "full"
    assert distill.main(base_args(pool, full)) == 0
    ref_rows = without_seconds(read_history(full / "history.csv")[1])
    assert len(ref_rows) == 2 + 5
    ref_best = torch.load(full / "final_adapter_best.pt", map_location="cpu", weights_only=False)
    ref_last = torch.load(full / "c2f_last.pt", map_location="cpu", weights_only=False)

    def check(run):
        assert without_seconds(read_history(run / "history.csv")[1]) == ref_rows
        best = torch.load(run / "final_adapter_best.pt", map_location="cpu", weights_only=False)
        assert checkpoint_fields(best) == checkpoint_fields(ref_best)
        assert states_equal(best["adapter_state"], ref_best["adapter_state"])
        last = torch.load(run / "c2f_last.pt", map_location="cpu", weights_only=False)
        assert states_equal(last["adapter_state"], ref_last["adapter_state"])
        res = json.loads((run / "results.json").read_text())
        ref = json.loads((full / "results.json").read_text())
        assert res["selected_best"]["diag"] == ref["selected_best"]["diag"]
        assert res["selection_note"] == ref["selection_note"]

    def checkpoint_fields(ck):
        return {k: ck[k] for k in ("epoch", "stage", "res", "epoch_in_stage", "val_mse", "val_cosine",
                                   "diag", "stage_val", "selected_by")}

    # stop at a stage boundary, then resume
    split = tmp_path / "split"
    assert distill.main(base_args(pool, split, extra=["--stop-after-stage", "1"])) == 0
    assert not (split / "results.json").exists()
    assert len(read_history(split / "history.csv")[1]) == 1 + 3
    assert distill.main(base_args(pool, split, extra=["--resume"])) == 0
    check(split)

    # interrupted inside a stage (during the second training epoch), then resume
    class Interrupted(Exception):
        pass

    real_run_epoch = distill.run_epoch
    calls = {"train": 0, "armed": True}

    def interrupting(*a, **k):
        if (len(a) > 5 and a[5] is not None) or k.get("optimizer") is not None:
            calls["train"] += 1
            if calls["armed"] and calls["train"] == 2:
                raise Interrupted
        return real_run_epoch(*a, **k)

    monkeypatch.setattr(distill, "run_epoch", interrupting)
    mid = tmp_path / "mid"
    with pytest.raises(Interrupted):
        distill.main(base_args(pool, mid))
    assert torch.load(mid / "c2f_last.pt", map_location="cpu", weights_only=False)["epoch_in_stage"] == 0
    calls["armed"] = False
    assert distill.main(base_args(pool, mid, extra=["--resume"])) == 0
    check(mid)

    # a different schedule is refused
    with pytest.raises(SystemExit):
        distill.main(base_args(pool, mid, stages="28:2,42:2", extra=["--resume"]))


# ── make-distill-split ───────────────────────────────────────────────────────

def _split_pool(tmp_path):
    imgs = tmp_path / "imgs"; imgs.mkdir()
    names = ([f"siteA_{i:04d}.jpg" for i in range(6)] + [f"siteB-{i:02d}.jpg" for i in range(4)]
             + [f"siteC_{i:04d}.jpg" for i in range(5)] + [f"vid{j}_{i:03d}.png" for j in range(8) for i in range(3)])
    for n in names:
        (imgs / n).write_bytes(b"x")
    (tmp_path / "manifest.txt").write_text("\n".join(names) + "\n")
    queries = tmp_path / "queries"; queries.mkdir()
    (queries / "siteA_0003.json").write_text("{}")
    (queries / "unrelated_0001.json").write_text("{}")
    (tmp_path / "eval_list.txt").write_text("some/dir/siteC_0002.jpg\n")
    return names


def _run_split(tmp_path, out, *extra):
    return distill_split.main(["--manifest", "manifest.txt", "--image-root", "imgs", "--exclude-dir", "queries",
                               "--exclude-list", "eval_list.txt", "--val-frac", "0.2", "--out-dir", out, *extra])


def _lines(p):
    return [line for line in p.read_text().splitlines() if line]


def test_distill_split_groups_sequences_and_excludes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = _split_pool(tmp_path)
    group = distill_split.group_of
    assert group("siteB-03") == "siteB" and group("vid1_002") == "vid1" and group("siteA_0003") == "siteA"

    assert _run_split(tmp_path, "plain") == 0
    train, val = _lines(tmp_path / "plain/train.txt"), _lines(tmp_path / "plain/val.txt")
    excluded = _lines(tmp_path / "plain/excluded.txt")
    assert sorted(excluded) == ["siteA_0003.jpg", "siteC_0002.jpg"]
    assert sorted(train + val + excluded) == sorted(names)
    assert not {group(n.rsplit(".", 1)[0]) for n in train} & {group(n.rsplit(".", 1)[0]) for n in val}
    assert val and train
    rep = json.loads((tmp_path / "plain/split_report.json").read_text())
    assert rep["n_excluded_by_stem"] == 2 and rep["n_excluded_by_sequence"] == 0
    assert rep["excluded_per_source"] == {"queries": 1, "eval_list.txt": 1}
    assert rep["manifest"] == "manifest.txt" and rep["image_root"] == "imgs"
    assert str(tmp_path) not in (tmp_path / "plain/split_report.json").read_text()

    # the stricter isolation removes the whole sequences of the excluded images; on this
    # small pool no group is then small enough for the 5 % validation target, which is
    # refused (nothing written), so a larger target is used
    assert _run_split(tmp_path, "strict_empty", "--exclude-sequences-of-excluded") == 1
    assert not (tmp_path / "strict_empty" / "train.txt").exists()
    assert _run_split(tmp_path, "strict", "--exclude-sequences-of-excluded", "--val-frac", "0.3") == 0
    train2, val2 = _lines(tmp_path / "strict/train.txt"), _lines(tmp_path / "strict/val.txt")
    assert not any(n.startswith(("siteA_", "siteC_")) for n in train2 + val2)
    rep2 = json.loads((tmp_path / "strict/split_report.json").read_text())
    assert rep2["n_excluded_by_sequence"] == 6 + 5 - 2
    assert rep2["excluded_sequences"] == ["siteA", "siteC", "unrelated"]

    # deterministic for a fixed seed
    assert _run_split(tmp_path, "again") == 0
    assert (tmp_path / "again/val.txt").read_text() == (tmp_path / "plain/val.txt").read_text()


def test_distill_split_rejects_bad_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _split_pool(tmp_path)
    (tmp_path / "imgs/siteA_0000.jpg").unlink()
    with pytest.raises(SystemExit):
        _run_split(tmp_path, "out")
    (tmp_path / "imgs/siteA_0000.jpg").write_bytes(b"x")
    (tmp_path / "manifest.txt").write_text("siteA_0000.jpg\nother/siteA_0000.jpg\n")
    with pytest.raises(SystemExit):
        _run_split(tmp_path, "out")
