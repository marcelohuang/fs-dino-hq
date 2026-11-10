"""Tests for the DINOv2 hub-source resolution in fs_dino_hq.modeling.backbones.

``torch.hub.load`` is replaced by a recorder that returns a tiny module, so no
network access, hub cache or weights are needed.
"""
from __future__ import annotations

import re

import pytest
import torch
import torch.nn as nn

from fs_dino_hq.modeling import backbones as bb


class _Recorder:
    """Stands in for torch.hub.load; records the call and returns a small module."""

    def __init__(self):
        self.calls = []

    def __call__(self, repo_or_dir, model, *args, **kwargs):
        self.calls.append((repo_or_dir, model, args, kwargs))
        torch.manual_seed(1)
        return nn.Sequential(nn.Linear(4, 3), nn.Dropout(0.5))


@pytest.fixture
def hub(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(torch.hub, "load", rec)
    monkeypatch.delenv(bb.DINOV2_REPO_ENV, raising=False)
    return rec


def test_default_is_pinned_github_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", bb.DINOV2_HUB_COMMIT)
    assert bb.DINOV2_HUB_REPO == f"facebookresearch/dinov2:{bb.DINOV2_HUB_COMMIT}"


def test_build_uses_pinned_ref_by_default(hub):
    model = bb.build_dinov2_vitb14()
    assert hub.calls == [(bb.DINOV2_HUB_REPO, "dinov2_vitb14", (),
                          {"pretrained": False, "source": "github", "skip_validation": True})]
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_empty_env_var_falls_back_to_pin(hub, monkeypatch):
    monkeypatch.setenv(bb.DINOV2_REPO_ENV, "")
    assert bb.dinov2_hub_source(None) == (bb.DINOV2_HUB_REPO, "github")


def test_env_var_local_directory(hub, monkeypatch, tmp_path):
    clone = tmp_path / "dinov2"
    clone.mkdir()
    monkeypatch.setenv(bb.DINOV2_REPO_ENV, str(clone))
    bb.build_dinov2_vitb14()
    repo, name, _, kwargs = hub.calls[-1]
    assert (repo, name) == (str(clone), "dinov2_vitb14")
    assert kwargs == {"pretrained": False, "source": "local"}       # no skip_validation for local


def test_argument_overrides_env_var(hub, monkeypatch, tmp_path):
    env_dir, arg_dir = tmp_path / "env", tmp_path / "arg"
    env_dir.mkdir()
    arg_dir.mkdir()
    monkeypatch.setenv(bb.DINOV2_REPO_ENV, str(env_dir))
    bb.build_dinov2_vitb14(str(arg_dir))
    assert hub.calls[-1][0] == str(arg_dir) and hub.calls[-1][3]["source"] == "local"


def test_home_directory_is_expanded(hub, monkeypatch, tmp_path):
    (tmp_path / "dinov2").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    assert bb.dinov2_hub_source("~/dinov2") == (str(tmp_path / "dinov2"), "local")


@pytest.mark.parametrize("ref", ["facebookresearch/dinov2", "facebookresearch/dinov2:main"])
def test_other_github_refs(hub, monkeypatch, ref):
    monkeypatch.setenv(bb.DINOV2_REPO_ENV, ref)
    bb.build_dinov2_vitb14()
    repo, _, _, kwargs = hub.calls[-1]
    assert repo == ref and kwargs["source"] == "github" and kwargs["skip_validation"] is True


@pytest.mark.parametrize("missing", ["/nonexistent/dinov2", "./no_such_dinov2", "~/no_such_dinov2"])
def test_missing_local_path_is_an_error(hub, monkeypatch, tmp_path, missing):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="DINOv2 hub directory not found"):
        bb.build_dinov2_vitb14(missing)
    assert hub.calls == []


def test_load_reads_weights_and_passes_hub_repo(hub, tmp_path):
    clone = tmp_path / "dinov2"
    clone.mkdir()
    torch.manual_seed(2)
    ref = nn.Sequential(nn.Linear(4, 3), nn.Dropout(0.5))
    ckpt = tmp_path / "dino.pth"
    torch.save(ref.state_dict(), ckpt)
    model = bb.load_dinov2_vitb14(ckpt, device=torch.device("cpu"), hub_repo=str(clone))
    assert hub.calls[-1][0] == str(clone)
    for k, v in ref.state_dict().items():
        assert torch.equal(model.state_dict()[k], v)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
