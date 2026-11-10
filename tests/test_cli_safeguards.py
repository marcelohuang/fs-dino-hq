"""Command-line safeguards: option precedence, refusals and input checks."""
import json

import numpy as np
import pytest
from PIL import Image

from fs_dino_hq.data import distill_split
from fs_dino_hq.data.labelme import open_image_rgb
from fs_dino_hq.evaluation.evaluate import evaluate_dir
from fs_dino_hq.inference.predict import clear_or_refuse
from fs_dino_hq.training import distill, train_hq, train_stage2
from fs_dino_hq.training.checkpoints import run_ckpt_prefix


def _write_config(path, cfg):
    path.write_text(json.dumps(cfg))
    return str(path)


HQ_BASE = {"synthetic_root": "d", "train_plan": "t.csv", "val_plan": "v.csv", "dino_ckpt": "a",
           "sam_ckpt": "b", "sam_hq_ckpt": "c", "early_adapter_ckpt": "e", "save_dir": "o"}


def test_hq_command_line_stage2_input_replaces_config(tmp_path):
    cfg = _write_config(tmp_path / "hq.json", {**HQ_BASE, "stage2_run": "runs/stage2"})
    args = train_hq.parse_args(["--config", cfg, "--stage2-ckpt", "x.pt", "--allow-unselected-stage2"])
    assert args.stage2_ckpt == "x.pt" and args.stage2_run is None
    args = train_hq.parse_args(["--config", cfg])
    assert args.stage2_run == "runs/stage2" and args.stage2_ckpt is None
    with pytest.raises(SystemExit):     # both on the command line
        train_hq.parse_args(["--config", cfg, "--stage2-run", "r", "--stage2-ckpt", "x.pt",
                             "--allow-unselected-stage2"])


S2_BASE = {"synthetic_root": "d", "train_plan": "t.csv", "adapter_ckpt": "a", "dino_ckpt": "b",
           "sam_ckpt": "c", "save_dir": "o", "n_shot": 5, "select_on": "val5", "epochs": 200,
           "val_plans": ["val5=v5.csv:5", "val1=v1.csv:1"]}


def test_stage2_command_line_val_plan_replaces_config_plans(tmp_path):
    cfg = _write_config(tmp_path / "s2.json", S2_BASE)
    args = train_stage2.parse_args(["--config", cfg, "--val-plan", "other.csv"])
    assert args.val_plans == ["val1=other.csv:1"] and args.select_on == "val1"
    args = train_stage2.parse_args(["--config", cfg])
    assert args.select_on == "val5" and len(args.val_plans) == 2


def test_stage2_smoke_runs_one_epoch_unless_epochs_given(tmp_path):
    cfg = _write_config(tmp_path / "s2.json", S2_BASE)
    assert train_stage2.parse_args(["--config", cfg, "--smoke"]).epochs == 1
    assert train_stage2.parse_args(["--config", cfg, "--smoke", "--epochs", "3"]).epochs == 3


def test_selection_prefix_read_from_run_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"ckpt_prefix": "s2"}))
    assert run_ckpt_prefix(tmp_path) == "s2"
    assert run_ckpt_prefix(tmp_path / "missing") == "stage2"


def test_distill_refuses_existing_run_without_resume(tmp_path):
    (tmp_path / distill.LAST_FILE).write_bytes(b"x")
    with pytest.raises(SystemExit, match="--resume"):
        distill.main(["--adapter", "final", "--train-manifest", "t.txt", "--val-manifest", "v.txt",
                      "--dino-ckpt", "a", "--sam-ckpt", "b", "--save-dir", str(tmp_path)])


def test_distill_split_refuses_empty_validation(tmp_path, capsys):
    names = [f"IMG_{i:04d}.jpg" for i in range(50)]          # one sequence under the default rule
    (tmp_path / "img").mkdir()
    for n in names:
        (tmp_path / "img" / n).write_bytes(b"")
    (tmp_path / "all.txt").write_text("\n".join(names) + "\n")
    common = ["--manifest", str(tmp_path / "all.txt"), "--image-root", str(tmp_path / "img")]
    rc = distill_split.main(common + ["--out-dir", str(tmp_path / "split")])
    assert rc == 1 and not (tmp_path / "split" / "train.txt").exists()
    assert "would be empty" in capsys.readouterr().err
    rc = distill_split.main(common + ["--out-dir", str(tmp_path / "split"), "--group-regex", "(?!)"])
    assert rc == 0
    report = json.loads((tmp_path / "split" / "split_report.json").read_text())
    assert report["n_val"] > 0 and report["group_regex"] == "(?!)"


def test_evaluate_refuses_zero_one_masks(tmp_path):
    (tmp_path / "q.json").write_text(json.dumps({"imageHeight": 8, "imageWidth": 8, "shapes": [
        {"label": "x", "points": [[1, 1], [6, 1], [6, 6], [1, 6]]}]}))
    pred = np.zeros((8, 8), np.uint8); pred[2:6, 2:6] = 1
    (tmp_path / "pred").mkdir()
    Image.fromarray(pred).save(tmp_path / "pred" / "q.png")
    with pytest.raises(ValueError, match="0/255"):
        evaluate_dir(tmp_path / "pred", tmp_path)
    Image.fromarray(pred * 255).save(tmp_path / "pred" / "q.png")
    rows, skipped, missing = evaluate_dir(tmp_path / "pred", tmp_path)
    assert len(rows) == 1 and rows[0]["iou"] > 0 and not missing


def test_evaluate_fails_when_an_annotated_image_has_no_prediction(tmp_path):
    """A mean over fewer images than annotated must not be returned silently."""
    from fs_dino_hq.evaluation import evaluate as evaluate_cmd
    gt, pred = tmp_path / "gt", tmp_path / "pred"
    gt.mkdir(), pred.mkdir()
    for name in ("a", "b"):
        (gt / f"{name}.json").write_text(json.dumps({"imageHeight": 8, "imageWidth": 8, "shapes": [
            {"label": "x", "points": [[1, 1], [6, 1], [6, 6], [1, 6]]}]}))
    (gt / "support_set.json").write_text(json.dumps(["a", "b"]))       # not an annotation: ignored
    (gt / "empty.json").write_text(json.dumps({"imageHeight": 8, "imageWidth": 8, "shapes": []}))
    m = np.zeros((8, 8), np.uint8); m[1:7, 1:7] = 255
    Image.fromarray(m).save(pred / "a.png")
    rows, skipped, missing = evaluate_dir(pred, gt)
    assert [r["image"] for r in rows] == ["a"] and missing == ["b"]
    argv = ["--pred-dir", str(pred), "--gt-dir", str(gt), "--out-json", str(tmp_path / "m.json"),
            "--out-csv", str(tmp_path / "m.csv")]
    with pytest.raises(SystemExit, match="have no prediction"):
        evaluate_cmd.main(argv)
    assert not (tmp_path / "m.json").exists()
    assert evaluate_cmd.main(argv + ["--allow-missing"]) == 0
    summary = json.loads((tmp_path / "m.json").read_text())
    assert summary["n_evaluated"] == 1 and summary["missing_predictions"] == ["b"]


def test_predict_output_dir_refused_unless_overwrite(tmp_path):
    (tmp_path / "pred_mask").mkdir()
    (tmp_path / "pred_mask" / "old.png").write_bytes(b"x")
    with pytest.raises(SystemExit, match="--overwrite"):
        clear_or_refuse(tmp_path, overwrite=False)
    clear_or_refuse(tmp_path, overwrite=True)
    assert not (tmp_path / "pred_mask").exists()
    clear_or_refuse(tmp_path, overwrite=False)      # empty directory: nothing to refuse


def test_exif_orientation_applied(tmp_path):
    img = Image.fromarray(np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3))   # 6 wide, 4 high
    exif = Image.Exif()
    exif[0x0112] = 6                                    # rotate 90 degrees clockwise on display
    img.save(tmp_path / "r.jpg", exif=exif, quality=100)
    assert open_image_rgb(tmp_path / "r.jpg").size == (4, 6)
    img.save(tmp_path / "p.png")                        # no orientation tag: unchanged
    assert np.array_equal(np.array(open_image_rgb(tmp_path / "p.png")), np.array(img))
