"""plot-history: run-type detection, selected epochs, series CSV and figures on
synthetic histories. The headers are the exact ones the three trainers write;
all values are synthetic."""
import csv
import json
import sys

import pytest

from fs_dino_hq.training import history as H

STAGE1_HEADER = ("stage,res,epoch_in_stage,global_epoch,event,adapter_sha16,diag_mse,diag_normalized_mse,diag_cosine,"
                 "diag_mse_content,diag_normalized_mse_content,diag_cosine_content,diag_student_norm,"
                 "diag_teacher_norm,diag_loss,lr,train_loss,val_loss,val_cosine,val_cosine_content,seconds")
STAGE2_FAST_HEADER = ("epoch,plan_offset,train_loss,train_loss_prior,train_loss_final,train_iou_final,train_iou_prior,"
                      "train_episodes,train_seconds,fast_val_loss,fast_val_loss_prior,fast_val_loss_final,"
                      "fast_val_iou_final,fast_val_iou_prior,fast_val_episodes,fast_val_seconds")
STAGE2_FULL_HEADER = ("epoch,full_val_loss,full_val_loss_prior,full_val_loss_final,full_val_iou_final,"
                      "full_val_iou_prior,full_val_episodes,full_val_seconds,full_val_val5_loss,"
                      "full_val_val5_loss_prior,full_val_val5_loss_final,full_val_val5_iou_final,"
                      "full_val_val5_iou_prior,full_val_val5_episodes,full_val_val5_seconds,full_val_val1_loss,"
                      "full_val_val1_loss_prior,full_val_val1_loss_final,full_val_val1_iou_final,"
                      "full_val_val1_iou_prior,full_val_val1_episodes,full_val_val1_seconds")
HQ_FAST_HEADER = "epoch,updates,train_loss,train_loss_bce,train_loss_dice,fast_val_loss,plan_offset"
HQ_FULL_HEADER = "epoch,updates,full_val_loss,n_episodes"
SERIES_COLUMNS = ["panel", "series", "epoch", "value"]


def _write_csv(path, header, rows):
    cols = header.split(",") if isinstance(header, str) else header
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in cols} for r in rows])


def _series_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def make_stage1(run):
    """Stages 126:2, 252:2, 518:3; diag_mse has a tie (epochs 4 and 6) and lower
    values on stage_start rows, which must not be selected."""
    run.mkdir(parents=True, exist_ok=True)
    diag = [0.5, 0.3, 0.2, 0.1, 0.15, 0.1, 0.12]
    rows, g = [], 0
    for si, (res, n) in enumerate(((126, 2), (252, 2), (518, 3))):
        rows.append({"stage": si, "res": res, "epoch_in_stage": 0, "global_epoch": g, "event": "stage_start",
                     "diag_mse": 0.01, "diag_cosine": 0.1, "lr": 5e-4, "seconds": 0.0})
        for e in range(1, n + 1):
            g += 1
            rows.append({"stage": si, "res": res, "epoch_in_stage": e, "global_epoch": g, "event": "epoch",
                         "adapter_sha16": "0123456789abcdef", "diag_mse": diag[g - 1], "diag_mse_content": 2 * diag[g - 1],
                         "diag_cosine": 1 - diag[g - 1], "diag_cosine_content": 0.9 - diag[g - 1], "lr": 1e-4,
                         "train_loss": 0.2 / g, "val_loss": 0.25 / g, "val_cosine": 0.5, "val_cosine_content": 0.4,
                         "seconds": 1.0})
    _write_csv(run / "history.csv", STAGE1_HEADER, rows)
    (run / "config.json").write_text(json.dumps({"adapter": "final", "diag_res": 518}))
    return run


def make_stage2(run, report=True, extra_fast_cols=()):
    run.mkdir(parents=True, exist_ok=True)
    fast = [{"epoch": e, "plan_offset": 2000 * (e - 1), "train_loss": 2 - 0.1 * e, "train_loss_prior": 1 - 0.05 * e,
             "train_loss_final": 1 - 0.05 * e, "train_iou_final": 0.05 * e, "train_iou_prior": 0.02 * e,
             "train_episodes": 2000, "train_seconds": 10.0, "fast_val_loss": 2.1 - 0.1 * e,
             "fast_val_loss_prior": 1.05 - 0.05 * e, "fast_val_loss_final": 1.05 - 0.05 * e,
             "fast_val_iou_final": 0.04 * e, "fast_val_iou_prior": 0.01 * e, "fast_val_episodes": 300,
             "fast_val_seconds": 2.0} for e in range(1, 11)]
    fast[3]["fast_val_loss"] = ""                      # empty cell: point skipped
    full = []
    for e in (5, 10):
        r = {"epoch": e}
        for pfx, off in (("full_val", 0.0), ("full_val_val5", 0.0), ("full_val_val1", 0.03)):
            r.update({f"{pfx}_loss": 1.6 - 0.02 * e + off, f"{pfx}_loss_prior": 1.0 + off,
                      f"{pfx}_loss_final": 0.6 - 0.02 * e, f"{pfx}_iou_final": 0.5 + 0.01 * e - off,
                      f"{pfx}_iou_prior": 0.2, f"{pfx}_episodes": 1421, f"{pfx}_seconds": 30.0})
        full.append(r)
    _write_csv(run / "history_fast.csv", STAGE2_FAST_HEADER.split(",") + list(extra_fast_cols), fast)
    _write_csv(run / "history_full.csv", STAGE2_FULL_HEADER, full)
    (run / "checkpoint_selection_report.json").write_text(json.dumps({"primary_epoch": 5}))
    if report:
        (run / "stage2_selection_report.json").write_text(json.dumps(
            {"status": "OK", "winner": {"epoch": 7, "loss": 1.45, "loss_prior": 0.99, "loss_final": 0.46,
                                        "iou_final": 0.57, "artifact": "stage2_selected.pt"}}))
    return run


def make_hq(run):
    run.mkdir(parents=True, exist_ok=True)
    _write_csv(run / "loss_history_fast.csv", HQ_FAST_HEADER,
               [{"epoch": e, "updates": 1000 * e, "train_loss": 0.70 - 0.002 * e,
                 "train_loss_bce": 0.55 - 0.001 * e, "train_loss_dice": 0.15 - 0.001 * e,
                 "fast_val_loss": 0.24 - 0.001 * e, "plan_offset": 2000 * (e - 1)} for e in range(1, 11)])
    _write_csv(run / "loss_history_full.csv", HQ_FULL_HEADER,
               [{"epoch": e, "updates": 1000 * e, "full_val_loss": 0.23 - 0.001 * e, "n_episodes": 1421}
                for e in (5, 10)])
    (run / "config.json").write_text(json.dumps({"loss": "point_sampled"}))
    (run / "checkpoint_selection_report.json").write_text(json.dumps({"primary_epoch": 10}))
    # an HQ run directory holds a copy of the Stage 2 report; it must not be used
    (run / "stage2_selection_report.json").write_text(json.dumps({"status": "OK", "winner": {"epoch": 183}}))
    return run


def _names(hist):
    return {s.name for p in hist.panels for s in p.series}


def test_stage1_detection_selection_and_series(tmp_path):
    run = make_stage1(tmp_path / "c2f")
    assert H.detect_run_type(run) == "stage1"
    hist = H.load_history(run)
    assert hist.selected_epoch == 4                    # first minimum over epoch rows only
    assert [p.key for p in hist.panels] == ["stage_loss", "diag_mse", "diag_cosine"]
    assert [(a, b, t) for a, b, t in hist.stages] == [(0, 2, "126 px"), (2, 4, "252 px"), (4, 7, "518 px")]
    assert {"train_loss@126", "val_loss@252", "val_loss@518", "diag_mse", "diag_cosine_content"} <= _names(hist)
    loss = {s.name: s for s in hist.panels[0].series}
    assert loss["train_loss@252"].x == [3.0, 4.0]     # stage_start rows (empty losses) are not plotted
    assert hist.panels[1].series[0].x == [float(e) for e in range(1, 8)]
    cols, rows = _series_rows(H.write_series_csv(hist, tmp_path / "s1.csv"))
    assert cols == SERIES_COLUMNS
    assert {"train_loss@126", "diag_mse"} <= {r["series"] for r in rows}
    assert all(r["epoch"].isdigit() for r in rows)


def test_stage2_detection_selection_and_plan_columns(tmp_path):
    run = make_stage2(tmp_path / "s2", extra_fast_cols=("train_peak_mem_gb", "fast_val_peak_mem_gb"))
    assert H.detect_run_type(run) == "stage2"
    hist = H.load_history(run)
    assert hist.selected_epoch == 7 and "winner.epoch" in hist.selected_source
    assert [p.key for p in hist.panels] == ["loss", "loss_prior", "loss_final", "iou_final"]
    names = _names(hist)
    assert {"train_loss", "fast_val_loss", "full_val_loss", "full_val_val1_loss", "train_loss_prior",
            "fast_val_loss_final", "full_val_iou_final", "selected_full_val_loss"} <= names
    assert "full_val_val5_loss" not in names           # identical to the plain (selection-plan) columns
    total = {s.name: s for s in hist.panels[0].series}
    assert total["full_val_loss"].label == "full val (val5)"
    assert 4.0 not in total["fast_val_loss"].x and len(total["fast_val_loss"].x) == 9
    assert total["selected_full_val_loss"].x == [7.0] and total["selected_full_val_loss"].y == [1.45]
    cols, rows = _series_rows(H.write_series_csv(hist, tmp_path / "s2.csv"))
    assert cols == SERIES_COLUMNS
    assert {r["panel"] for r in rows} == {"loss", "loss_prior", "loss_final", "iou_final"}


def test_stage2_falls_back_to_trainer_primary_epoch(tmp_path):
    run = make_stage2(tmp_path / "s2", report=False)
    hist = H.load_history(run)
    assert hist.selected_epoch == 5 and "primary_epoch" in hist.selected_source
    assert "selected_full_val_loss" not in _names(hist)
    (run / "stage2_selection_report.json").write_text(json.dumps({"status": "FAILED", "error": "x"}))
    assert H.load_history(run).selected_epoch == 5


def test_hq_detection_ignores_stage2_report_copy(tmp_path):
    run = make_hq(tmp_path / "hq")
    assert H.detect_run_type(run) == "hq"
    hist = H.load_history(run)
    assert hist.selected_epoch == 10 and hist.selected_source.startswith("checkpoint_selection_report.json")
    assert _names(hist) == {"train_loss", "train_loss_bce", "train_loss_dice", "fast_val_loss", "full_val_loss"}
    # training (point-sampled) and validation (full grid) are on different scales: separate panels
    assert [p.key for p in hist.panels] == ["train", "loss"] and "sampled points" in hist.panels[0].title
    assert {s.name for s in hist.panels[1].series} == {"fast_val_loss", "full_val_loss"}
    assert hist.updates_per_epoch == 1000
    cols, rows = _series_rows(H.write_series_csv(hist, tmp_path / "hq.csv"))
    assert cols == SERIES_COLUMNS and len(rows) == 3 * 10 + 10 + 2


def test_hq_history_without_loss_terms_or_update_counts(tmp_path):
    """Histories written before the loss terms were recorded still load."""
    run = tmp_path / "hq_old"
    run.mkdir()
    _write_csv(run / "loss_history_fast.csv", "epoch,train_loss,fast_val_loss,plan_offset",
               [{"epoch": e, "train_loss": 0.2, "fast_val_loss": 0.24, "plan_offset": 0} for e in (1, 2)])
    hist = H.load_history(run)
    assert _names(hist) == {"train_loss", "fast_val_loss"} and hist.updates_per_epoch is None
    assert "256 grid" in hist.panels[0].title


def test_unknown_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        H.detect_run_type(tmp_path)
    assert H.main([str(tmp_path), "--out", str(tmp_path / "out")]) == 1


@pytest.mark.parametrize("maker,kind", [(make_stage1, "stage1"), (make_stage2, "stage2"), (make_hq, "hq")])
def test_cli_writes_figures_and_series(tmp_path, maker, kind, capsys):
    pytest.importorskip("matplotlib")
    from fs_dino_hq import cli

    run = maker(tmp_path / "run")
    before = sorted(p.name for p in run.iterdir())
    out = tmp_path / "plots"
    assert cli.main(["plot-history", str(run), "--out", str(out), "--format", "png", "pdf"]) == 0
    assert f": {kind};" in capsys.readouterr().out
    assert (out / "run.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (out / "run.pdf").read_bytes()[:5] == b"%PDF-"
    assert _series_rows(out / "run_series.csv")[0] == SERIES_COLUMNS
    assert sorted(p.name for p in run.iterdir()) == before          # run directory untouched


def test_default_output_is_the_run_directory_and_no_report_is_allowed(tmp_path):
    pytest.importorskip("matplotlib")
    run = make_hq(tmp_path / "hq")
    (run / "checkpoint_selection_report.json").unlink()
    assert H.load_history(run).selected_epoch is None
    assert H.main([str(run), "--name", "curves"]) == 0
    assert (run / "curves.png").is_file() and (run / "curves_series.csv").is_file()
    assert not (run / "curves.pdf").exists()


def test_missing_matplotlib_gives_install_hint(tmp_path, monkeypatch, capsys):
    run = make_hq(tmp_path / "hq")
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    monkeypatch.setitem(sys.modules, "matplotlib.backends.backend_agg", None)
    assert H.main([str(run), "--out", str(tmp_path / "out")]) == 1
    assert "pip install fs-dino-hq[plot]" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
