"""Post-training Stage 2 checkpoint selection: candidate deduplication, cache
reuse and invalidation, selection of an epoch that periodic full validation
skipped, interrupted and resumed selection, deterministic ties, the FAILED path,
path overrides, and the real evaluator end to end. CPU only, stub backbones."""
import json
import shutil
from pathlib import Path

import pytest
import torch

from stubs import StubDino, StubSam, make_tiny_photo_dataset

from fs_dino_hq.modeling.distill_adapter import BottleneckAdapter as StudentAdapter
from fs_dino_hq.training import select_stage2 as sel
from fs_dino_hq.training import train_stage2 as trainer

torch.set_num_threads(2)

LOSSES = {1: (0.5, 0.2), 2: (0.9, 0.3), 3: (0.8, 0.25)}
SELECTION_FILES = ("stage2_selection_cache.json", "stage2_selection_report.json",
                   "stage2_selected.pt", "stage2_selected.json")


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """A 3-epoch 1-shot run with a snapshot every epoch and full validation only at
    epochs 2 and 3, so epoch 1 is a saved candidate that training-time full
    validation never scored."""
    tmp = tmp_path_factory.mktemp("sel")
    root = tmp / "PHOTO"; plans = make_tiny_photo_dataset(root)
    torch.manual_seed(1); student = StudentAdapter(in_channels=768, out_channels=256)
    ck = tmp / "final_adapter_best.pt"
    torch.save({"adapter_state": student.state_dict(), "config": {"adapter": "final", "save_dir": "runs/x"},
                "stage": 2, "res": 518, "epoch_in_stage": 14, "epoch": 30}, ck)
    backbones = (StubDino(), StubSam())   # shared by training and selection (frozen DINOv2 stand-in)
    mp = pytest.MonkeyPatch()
    mp.setattr(trainer, "load_backbones", lambda d, s: backbones)
    run = tmp / "run"
    trainer.main(["--synthetic-root", str(root), "--train-plan", str(plans[("train", 1)]), "--val-plan", str(plans[("val", 1)]),
                  "--adapter-ckpt", str(ck), "--dino-ckpt", "unused", "--sam-ckpt", "unused", "--save-dir", str(run),
                  "--num-workers", "0", "--batch-size", "2", "--episodes-per-epoch", "4", "--val-episodes", "4",
                  "--full-val-every", "2", "--epochs", "3", "--save-every", "1", "--seed", "3"])
    mp.undo()
    return run, backbones, root, plans


def _copy_run(run: Path, dst: Path) -> Path:
    shutil.copytree(run, dst)
    for f in SELECTION_FILES:
        (dst / f).unlink(missing_ok=True)
    return dst


def _fake_scorer(losses_by_epoch, fail_on_call=None, calls=None):
    """Deterministic stand-in for evaluate_checkpoint: scores depend on the epoch only;
    fail_on_call=n raises on the n-th call (1-based) to simulate an interruption."""
    def scorer(model, path, loader, device, load=True):
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        ep = int(ck["epoch"])
        if calls is not None:
            calls.append(ep)
        if fail_on_call is not None and len(calls) == fail_on_call:
            raise RuntimeError("simulated interruption")
        total, final = losses_by_epoch[ep]
        return {"loss": total, "loss_prior": total - final, "loss_final": final, "iou_final": 1 - final,
                "iou_prior": 0.1, "episodes": len(loader.dataset), "seconds": 0.0, "epoch": ep}
    return scorer


def test_dedup_reuse_invalidation_and_skipped_epoch_can_win(tiny_run, tmp_path, monkeypatch):
    run0, backbones, _, _ = tiny_run
    run = _copy_run(run0, tmp_path / "run")
    files = sel.enumerate_candidates(run, "stage2")
    assert {f.name for f in files} >= {"stage2_epoch_001.pt", "stage2_epoch_002.pt", "stage2_epoch_003.pt",
                                        "stage2_best_full_val.pt", "stage2_best.pt", "stage2_last.pt"}
    # epoch 1 was never full-validated during training (full_val_every=2) but is a saved candidate
    hist = (run / "history_full.csv").read_text()
    assert "\n1," not in hist and "\n2," in hist and "\n3," in hist
    calls = []
    rep = sel.select_stage2_checkpoint(run, scorer=_fake_scorer(LOSSES, calls=calls),
                                       load_backbones=lambda d, s: backbones, num_workers=0)
    assert rep["status"] == "OK"
    assert rep["n_distinct_states"] == 3 and rep["n_checkpoint_files"] >= 6          # dedup by model-state hash
    assert sorted(calls) == [1, 2, 3], calls                                         # one evaluation per distinct state
    assert rep["winner"]["epoch"] == 1 and "stage2_epoch_001.pt" in rep["winner"]["paths"]
    assert rep["winner"]["artifact_updated"] and (run / "stage2_selected.pt").is_file()
    assert "best among the SAVED candidate" in rep["scope"]
    side = json.loads((run / "stage2_selected.json").read_text())
    assert side["state_sha256"] == rep["winner"]["state_sha256"] and side["epoch"] == 1
    assert side["file_sha256"] == sel.file_sha256(run / "stage2_selected.pt")
    assert (run / "stage2_selected.pt").read_bytes() == (run / "stage2_epoch_001.pt").read_bytes()
    assert "stage2_best_full_val.pt" in sum((r["paths"] for r in rep["candidates"]), []) and \
        torch.load(run / "stage2_best.pt", map_location="cpu", weights_only=False)["epoch"] != 1, "training-time best untouched"
    assert all(r["reused"] is False for r in rep["candidates"])
    assert rep["preferred_by_decoder_loss"]["epoch"] == 1 and rep["training_time_best"]["full_val"]
    # second pass: everything compatible is reused, artifact unchanged
    mtime = (run / "stage2_selected.pt").stat().st_mtime_ns
    calls2 = []
    rep2 = sel.select_stage2_checkpoint(run, scorer=_fake_scorer(LOSSES, calls=calls2),
                                        load_backbones=lambda d, s: backbones, num_workers=0)
    assert calls2 == [] and rep2["n_fresh_evaluations"] == 0 and all(r["reused"] for r in rep2["candidates"])
    assert rep2["winner"]["artifact_updated"] is False and (run / "stage2_selected.pt").stat().st_mtime_ns == mtime
    # an incompatible setting (batch size) invalidates the cache -> recomputed, same winner
    calls3 = []
    rep3 = sel.select_stage2_checkpoint(run, scorer=_fake_scorer(LOSSES, calls=calls3),
                                        load_backbones=lambda d, s: backbones, num_workers=0, batch_size=1)
    assert sorted(calls3) == [1, 2, 3] and rep3["winner"]["epoch"] == 1
    # so does a different evaluation implementation (code hash)
    calls4 = []
    with monkeypatch.context() as m:
        m.setattr(sel, "code_sha256", lambda: "0" * 64)
        sel.select_stage2_checkpoint(run, scorer=_fake_scorer(LOSSES, calls=calls4),
                                     load_backbones=lambda d, s: backbones, num_workers=0)
    assert sorted(calls4) == [1, 2, 3]
    # a changed winner rewrites the artifact and the sidecar
    rep5 = sel.select_stage2_checkpoint(run, scorer=_fake_scorer({1: (0.9, 0.3), 2: (0.95, 0.3), 3: (0.4, 0.2)}),
                                        load_backbones=lambda d, s: backbones, num_workers=0, batch_size=2)
    assert rep5["winner"]["epoch"] == 3 and rep5["winner"]["artifact_updated"]
    assert json.loads((run / "stage2_selected.json").read_text())["epoch"] == 3


def test_identical_weights_with_different_epochs_are_refused(tiny_run, tmp_path):
    run0, backbones, _, _ = tiny_run
    run = _copy_run(run0, tmp_path / "run")
    ck = torch.load(run / "stage2_epoch_002.pt", map_location="cpu", weights_only=False)
    ck["epoch"] = 7
    torch.save(ck, run / "stage2_epoch_007.pt")
    with pytest.raises(ValueError, match="different epochs"):
        sel.select_stage2_checkpoint(run, scorer=_fake_scorer(LOSSES), load_backbones=lambda d, s: backbones,
                                     num_workers=0)


def test_interrupted_selection_resumes_from_cache_and_failed_cli(tiny_run, tmp_path):
    run0, backbones, _, _ = tiny_run
    run2 = _copy_run(run0, tmp_path / "run2")
    losses = {1: (0.9, 0.3), 2: (0.7, 0.3), 3: (0.6, 0.2)}
    calls = []
    with pytest.raises(RuntimeError, match="simulated interruption"):
        sel.select_stage2_checkpoint(run2, scorer=_fake_scorer(losses, fail_on_call=3, calls=calls),
                                     load_backbones=lambda d, s: backbones, num_workers=0)
    assert not (run2 / "stage2_selected.pt").exists() and not (run2 / "stage2_selection_report.json").exists()
    cache = json.loads((run2 / "stage2_selection_cache.json").read_text())
    assert len(cache) == 2                                   # the two completed evaluations survived
    calls2 = []
    rep = sel.select_stage2_checkpoint(run2, scorer=_fake_scorer(losses, calls=calls2),
                                       load_backbones=lambda d, s: backbones, num_workers=0)
    assert len(calls2) == 1 and calls2[0] == calls[2] and rep["n_reused_evaluations"] == 2 and rep["winner"]["epoch"] == 3
    # CLI: a candidate that cannot be read fails the step with a FAILED report and no artifact
    run3 = _copy_run(run0, tmp_path / "run3")
    (run3 / "stage2_epoch_002.pt").write_bytes(b"corrupt")
    before = trainer.load_backbones
    rc = sel.main(["--run-dir", str(run3), "--num-workers", "0"])
    assert rc == 1
    failed = json.loads((run3 / "stage2_selection_report.json").read_text())
    assert failed["status"] == "FAILED" and "error" in failed
    assert not (run3 / "stage2_selected.pt").exists()
    assert trainer.load_backbones is before
    # an earlier successful artifact is kept, but the report says FAILED
    run4 = _copy_run(run0, tmp_path / "run4")
    sel.select_stage2_checkpoint(run4, scorer=_fake_scorer(LOSSES), load_backbones=lambda d, s: backbones, num_workers=0)
    (run4 / "stage2_epoch_003.pt").write_bytes(b"corrupt")
    assert sel.main(["--run-dir", str(run4), "--num-workers", "0"]) == 1
    assert json.loads((run4 / "stage2_selection_report.json").read_text())["status"] == "FAILED"
    assert (run4 / "stage2_selected.pt").is_file()


def test_tie_break_is_deterministic():
    rows = [{"loss": 1.0, "loss_final": 0.3, "iou_final": 0.7, "epoch": 20},
            {"loss": 1.0, "loss_final": 0.2, "iou_final": 0.7, "epoch": 30},
            {"loss": 1.0, "loss_final": 0.2, "iou_final": 0.7, "epoch": 10},
            {"loss": 1.0 + 1e-12, "loss_final": 0.1, "iou_final": 0.9, "epoch": 5}]   # equal at 10 decimals
    assert [r["epoch"] for r in sorted(rows, key=sel.rank_key)] == [5, 10, 30, 20]
    rows2 = [{"loss": 1.0, "loss_final": 0.2, "iou_final": 0.6, "epoch": 1},
             {"loss": 1.0, "loss_final": 0.2, "iou_final": 0.8, "epoch": 2}]
    assert sorted(rows2, key=sel.rank_key)[0]["epoch"] == 2          # higher IoU before lower epoch


def test_real_scorer_end_to_end_with_path_overrides(tiny_run, tmp_path):
    """Real evaluate_checkpoint (stub backbones) through the CLI, with the dataset
    moved: every distinct state is scored on the whole plan, the re-scored
    training-time full-val best reproduces its recorded full-val loss, and the
    overrides are recorded in the report."""
    run0, backbones, root, plans = tiny_run
    run = _copy_run(run0, tmp_path / "run")
    moved = tmp_path / "moved_photo"
    shutil.copytree(root, moved)
    cfg = json.loads((run / "config.json").read_text())
    cfg["synthetic_root"] = str(tmp_path / "missing")                # recorded root no longer exists
    (run / "config.json").write_text(json.dumps(cfg))
    mp = pytest.MonkeyPatch()
    seen = []
    mp.setattr(trainer, "load_backbones", lambda d, s: seen.append((d, s)) or backbones)
    try:
        rc = sel.main(["--run-dir", str(run), "--num-workers", "0", "--batch-size", "2",
                       "--synthetic-root", str(moved), "--dino-ckpt", "dino.pth", "--sam-ckpt", "sam.pth"])
    finally:
        mp.undo()
    assert rc == 0 and seen == [("dino.pth", "sam.pth")]
    rep = json.loads((run / "stage2_selection_report.json").read_text())
    assert rep["status"] == "OK" and rep["n_fresh_evaluations"] == rep["n_distinct_states"]
    assert rep["inputs"]["synthetic_root"] == str(moved) and rep["inputs"]["dino_ckpt"] == "dino.pth"
    assert rep["inputs"]["overridden"] == ["dino_ckpt", "sam_ckpt", "synthetic_root"]
    n_plan = len(Path(rep["selection_plan"]["path"]).read_text().strip().splitlines()) - 1
    assert all(r["episodes"] == n_plan for r in rep["candidates"])
    best = next(r for r in rep["candidates"] if "best_full_val" in r["selection_tags"])
    ck = torch.load(run / "stage2_best_full_val.pt", map_location="cpu", weights_only=False)
    assert abs(best["loss"] - ck["val"]["full"]["loss"]) < 1e-4
    winner = min(rep["candidates"], key=sel.rank_key)
    assert rep["winner"]["state_sha256"] == winner["state_sha256"]
    sel_ck = torch.load(run / "stage2_selected.pt", map_location="cpu", weights_only=False)
    assert sel_ck["format"] == "stage2_model" and sel_ck["epoch"] == rep["winner"]["epoch"]
    assert sel.model_state_sha256(sel_ck["stage2_model"]) == rep["winner"]["state_sha256"]
    # without the override the recorded (missing) root is used and the step fails explicitly
    run_b = _copy_run(run, tmp_path / "run_b")
    mp = pytest.MonkeyPatch()
    mp.setattr(trainer, "load_backbones", lambda d, s: backbones)
    try:
        assert sel.main(["--run-dir", str(run_b), "--num-workers", "0", "--batch-size", "2"]) == 1
    finally:
        mp.undo()
    assert json.loads((run_b / "stage2_selection_report.json").read_text())["status"] == "FAILED"


def test_selected_artifact_passes_the_gate_and_loads_strictly_downstream(tiny_run, tmp_path):
    from fs_dino_hq.modeling.model import FSDinoHQModel
    from fs_dino_hq.training.checkpoints import (inspect_stage2_checkpoint, load_stage2_strict,
                                                 verify_selected_stage2)
    run0, backbones, _, _ = tiny_run
    run = _copy_run(run0, tmp_path / "run")
    rep = sel.select_stage2_checkpoint(run, scorer=_fake_scorer(LOSSES), load_backbones=lambda d, s: backbones,
                                       num_workers=0)
    gate = verify_selected_stage2(run)
    assert gate["selected_epoch"] == rep["winner"]["epoch"] and gate["state_sha256"] == rep["winner"]["state_sha256"]
    info = inspect_stage2_checkpoint(run / "stage2_selected.pt")
    assert info["format"] == "stage2_model" and info["adapter_arch"]["recorded_in_checkpoint"]
    assert info["adapter_arch"]["norm_first"] is True and info["epoch"] == rep["winner"]["epoch"]
    assert info["corr_arch"] == {"arch": "v2_reduced", "grid": 37}
    hq = FSDinoHQModel(StubDino(), StubSam(), corr_arch="v2_reduced", feat_hw=37,
                       adapter_norm_first=True, adapter_dropout=0.0)
    r = load_stage2_strict(hq, run / "stage2_selected.pt", tmp_path / "load_report.json")
    assert r["status"] == "OK" and r["adapter_arch"]["checkpoint_norm_first"] is True
