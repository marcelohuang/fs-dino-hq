"""JSON configurations (${VAR} expansion, key and value checks) and the fs-dino-hq dispatcher."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fs_dino_hq import cli
from fs_dino_hq.config import check_config_values, expand_env, load_config, parse_with_config

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _parser():
    p = argparse.ArgumentParser(prog="t")
    p.add_argument("--config", default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--n-shot", type=int, default=1, choices=[1, 5])
    p.add_argument("--decoder-mode", default="dual_canonical", choices=["dual_canonical"])
    p.add_argument("--label-filter", nargs="+", default=None)
    p.add_argument("--milestones", nargs="+", type=int, default=[])
    p.add_argument("--amp", action="store_true")
    return p


def _write(tmp_path, cfg, name="cfg.json"):
    f = tmp_path / name
    f.write_text(json.dumps(cfg))
    return str(f)


def test_expand_env(monkeypatch):
    monkeypatch.setenv("FSDHQ_DATA", "/data/x")
    assert expand_env("${FSDHQ_DATA}/photo") == "/data/x/photo"
    assert expand_env({"a": ["${FSDHQ_DATA}", 3], "b": {"c": "$FSDHQ_DATA"}}) == \
        {"a": ["/data/x", 3], "b": {"c": "$FSDHQ_DATA"}}            # only ${NAME} is expanded
    assert expand_env(5) == 5 and expand_env(None) is None
    monkeypatch.delenv("FSDHQ_UNSET", raising=False)
    with pytest.raises(KeyError, match="FSDHQ_UNSET"):
        expand_env("${FSDHQ_UNSET}/x")


def test_load_config_keys_and_comments(tmp_path, monkeypatch):
    monkeypatch.setenv("FSDHQ_OUT", "/out")
    f = _write(tmp_path, {"_comment": "ignored", "save_dir": "${FSDHQ_OUT}/run", "lr": 0.001})
    assert load_config(f, {"save_dir", "lr"}) == {"save_dir": "/out/run", "lr": 0.001}
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(f, {"save_dir"})
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2]")
    with pytest.raises(ValueError):
        load_config(bad)


def test_parse_with_config_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("FSDHQ_OUT", "/out")
    f = _write(tmp_path, {"save_dir": "${FSDHQ_OUT}/a", "lr": 0.01, "epochs": 3, "amp": True,
                          "label_filter": ["spacing"], "milestones": [5, "7"], "n_shot": 5})
    a = parse_with_config(_parser(), ["--config", f])
    assert (a.save_dir, a.lr, a.epochs, a.amp, a.label_filter, a.milestones, a.n_shot) == \
        ("/out/a", 0.01, 3, True, ["spacing"], [5, 7], 5)
    b = parse_with_config(_parser(), ["--config", f, "--lr", "0.5", "--save-dir", "/cli"])
    assert (b.lr, b.save_dir, b.epochs) == (0.5, "/cli", 3)           # CLI wins
    c = parse_with_config(_parser(), ["--lr", "0.2"])                 # no config
    assert (c.lr, c.epochs, c.save_dir) == (0.2, 10, None)


def test_parse_with_config_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("FSDHQ_UNSET", raising=False)
    with pytest.raises(ValueError, match="unknown keys"):
        parse_with_config(_parser(), ["--config", _write(tmp_path, {"learning_rate": 1})])
    with pytest.raises(KeyError, match="FSDHQ_UNSET"):
        parse_with_config(_parser(), ["--config", _write(tmp_path, {"save_dir": "${FSDHQ_UNSET}"})])
    with pytest.raises(ValueError, match="not one of"):
        parse_with_config(_parser(), ["--config", _write(tmp_path, {"decoder_mode": "shared_pass"})])
    with pytest.raises(ValueError, match="not one of"):
        parse_with_config(_parser(), ["--config", _write(tmp_path, {"n_shot": 3})])
    with pytest.raises(ValueError, match="true or false"):
        parse_with_config(_parser(), ["--config", _write(tmp_path, {"amp": "false"})])
    with pytest.raises(ValueError, match="takes a list"):
        parse_with_config(_parser(), ["--config", _write(tmp_path, {"milestones": 5})])


def test_check_config_values_normalises():
    p = _parser()
    out = check_config_values(p, {"label_filter": "spacing", "n_shot": "5", "save_dir": None})
    assert out["label_filter"] == ["spacing"]       # one string -> one-element list
    assert out["n_shot"] == "5"                      # argparse converts string defaults itself
    p.set_defaults(**out)
    a = p.parse_args([])
    assert a.n_shot == 5 and a.save_dir is None


# ── dispatcher ──────────────────────────────────────────────────────────────

def test_cli_help_lists_commands(capsys):
    assert cli.main(["--help"]) == 0
    text = capsys.readouterr().out
    for name in ("make-distill-split", "distill", "make-episodes", "train-stage2", "select-stage2",
                 "train-hq", "predict", "evaluate", "compare", "plot-history"):
        assert f"  {name} " in text
    assert cli.main([]) == 0


def test_cli_unknown_command(capsys):
    assert cli.main(["bogus"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_cli_version(capsys):
    from fs_dino_hq import __version__
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_dispatches_to_command_help(capsys):
    for name in ("predict", "evaluate", "compare"):
        with pytest.raises(SystemExit) as e:
            cli.main([name, "--help"])
        assert e.value.code == 0
        assert f"fs-dino-hq {name}" in capsys.readouterr().out


def _run(*args):
    env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env, timeout=300)


def test_cli_module_subprocess():
    r = _run("-m", "fs_dino_hq.cli", "--help")
    assert r.returncode == 0 and "predict" in r.stdout
    r = _run("-m", "fs_dino_hq.cli", "no-such-command")
    assert r.returncode == 2 and "unknown command" in r.stderr


def test_command_module_runs_as_main():
    r = _run("-m", "fs_dino_hq.evaluation.compare", "--help")
    assert r.returncode == 0 and "--n-boot" in r.stdout
