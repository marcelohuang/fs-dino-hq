"""Checkpoint loading for HQ training: strict Stage 2 contract, the Stage 2
selection gate, the SAM-HQ partial-load mapping and the early adapter.

CPU only, stub backbones, fixtures written to tmp_path. The helpers at the top
(Stage 2 checkpoint, selected run directory, fake SAM-HQ state dict) are also
used by test_hq_training.py.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from segment_anything.modeling import TwoWayTransformer

from stubs import StubDino, StubSam

from fs_dino_hq.modeling.distill_adapter import BottleneckAdapter as DistillationAdapter
from fs_dino_hq.modeling.hq_decoder import HQMaskDecoder
from fs_dino_hq.modeling.model import FSDinoHQModel
from fs_dino_hq.modeling.stage2 import Stage2Model
from fs_dino_hq.training.checkpoints import (POST_TRAINING_SELECTION, Stage2SelectionError,
                                             inspect_stage2_checkpoint, load_early_adapter,
                                             load_sam_hq_partial, load_stage2_strict,
                                             verify_selected_stage2)
from fs_dino_hq.utils.provenance import file_sha256

torch.set_num_threads(4)


# ── helpers (shared with test_hq_training.py) ────────────────────────────────

def model_state_sha256(sd) -> str:
    """Same definition as the selection step: sorted keys; name, dtype, shape, bytes."""
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def make_stage2_model(seed: int = 3) -> Stage2Model:
    """Stage 2 model on stub backbones whose trainable weights differ from any
    fresh initialisation (so a successful load is visible)."""
    m = Stage2Model(StubDino(), StubSam(), adapter_norm_first=True, adapter_dropout=0.0)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for mod in (m.adapter, m.semantic_branch, m.corr_branch, m.sam_full.mask_decoder):
            for p in mod.parameters():
                p.add_(0.01 * torch.randn(p.shape, generator=g))
    return m


def save_stage2_checkpoint(path: Path, model: Stage2Model, epoch: int = 7,
                           tag: str = "best_full_val", state=None, **overrides) -> Path:
    """Stage 2 checkpoint with the key layout written by fs-dino-hq train-stage2."""
    obj = {"format": "stage2_model", "trainer": "fs_dino_hq.training.train_stage2",
           "epoch": epoch, "stage2_model": state if state is not None else model.stage2_state_dict(),
           "stats": {"loss": 1.0, "loss_prior": 0.6, "loss_final": 0.4},
           "val": {"fast": None, "full": None}, "selection_tag": tag,
           "adapter_norm_first": True, "adapter_dropout": 0.0, "corr_arch": "v2_reduced",
           "feat_hw": 37, "dino_excluded": True, "sam_image_encoder_excluded": True,
           "config": {"ckpt_prefix": "stage2"}}
    obj.update(overrides)
    torch.save(obj, path)
    return path


def make_selected_run(run: Path, model: Stage2Model, status: str = "OK") -> Path:
    """Stage 2 run directory as left by fs-dino-hq select-stage2."""
    run.mkdir(parents=True, exist_ok=True)
    src = save_stage2_checkpoint(run / "stage2_best_full_val.pt", model)
    shutil.copyfile(src, run / "stage2_selected.pt")
    state = model_state_sha256(model.stage2_state_dict())
    (run / "stage2_selected.json").write_text(json.dumps(
        {"source_file": src.name, "state_sha256": state,
         "file_sha256": file_sha256(run / "stage2_selected.pt"), "epoch": 7,
         "selection_plan": "val5", "loss": 1.2, "rule": "min full-val total loss"}))
    (run / "stage2_selection_report.json").write_text(json.dumps(
        {"status": status, "prefix": "stage2",
         "selection_plan": {"label": "val5", "n_shot": 5},
         "winner": {"paths": [src.name], "state_sha256": state, "epoch": 7, "loss": 1.2,
                    "artifact": "stage2_selected.pt"}}))
    return run


def make_hq_model(norm_first: bool = True) -> FSDinoHQModel:
    return FSDinoHQModel(StubDino(), StubSam(), corr_arch="v2_reduced", feat_hw=37, ds_hw=18,
                         adapter_norm_first=norm_first, adapter_dropout=0.0 if norm_first else 0.1)


def new_hq_decoder(seed: int = 0) -> HQMaskDecoder:
    torch.manual_seed(seed)
    return HQMaskDecoder(transformer_dim=256,
                         transformer=TwoWayTransformer(depth=2, embedding_dim=256,
                                                       mlp_dim=2048, num_heads=8),
                         num_multimask_outputs=3)


def fake_sam_hq_state(dec: HQMaskDecoder, seed: int = 11):
    """SAM-HQ ViT-H key layout with the shapes of our HQ modules (small
    stand-ins for the tensors that are not loaded)."""
    g = torch.Generator().manual_seed(seed)

    def rnd(shape):
        return torch.randn(tuple(shape), generator=g)

    sd = {}
    for src, mod in (("hf_token", dec.hf_token), ("hf_mlp", dec.hf_mlp),
                     ("embedding_encoder", dec.adapter_proj),
                     ("embedding_maskfeature", dec.decoder_mask_proj)):
        for k, v in mod.state_dict().items():
            sd[f"mask_decoder.{src}.{k}"] = rnd(v.shape)
    for k in ("0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias"):
        sd[f"mask_decoder.compress_vit_feat.{k}"] = rnd((4, 2))       # 1280-channel in reality
    sd["mask_decoder.iou_token.weight"] = rnd((1, 256))
    sd["mask_decoder.mask_tokens.weight"] = rnd((4, 256))
    sd["image_encoder.pos_embed"] = rnd((1, 2, 2, 8))
    sd["prompt_encoder.no_mask_embed.weight"] = rnd((1, 256))
    return sd


@pytest.fixture(scope="module")
def stage2_model():
    return make_stage2_model()


@pytest.fixture(scope="module")
def stage2_ckpt(stage2_model, tmp_path_factory):
    return save_stage2_checkpoint(tmp_path_factory.mktemp("s2") / "stage2_best_full_val.pt",
                                  stage2_model)


# ── inspection and strict Stage 2 load ───────────────────────────────────────

def test_inspect_reports_convention_and_architecture(stage2_ckpt, tmp_path):
    info = inspect_stage2_checkpoint(stage2_ckpt, out_json=tmp_path / "inspect.json")
    assert info["format"] == "stage2_model"
    assert info["adapter_arch"] == {"norm_first": True, "dropout": 0.0, "recorded_in_checkpoint": True}
    assert info["corr_arch"] == {"arch": "v2_reduced", "grid": 37}
    assert info["prefixes"]["adapter"]["tensors"] == 44
    assert not any(p.startswith(("last_extractor", "layer3_extractor")) for p in info["prefixes"])
    assert json.loads((tmp_path / "inspect.json").read_text())["epoch"] == 7


def test_strict_load_checks_the_recorded_dinov2_hash(stage2_model, tmp_path):
    """Stage 2 checkpoints hold no DINOv2 tensors; the weights file is identified by its recorded hash."""
    ck = save_stage2_checkpoint(tmp_path / "s2.pt", stage2_model,
                                config={"ckpt_prefix": "stage2", "input_sha256": {"dino_ckpt": "a" * 64}})
    rep = load_stage2_strict(make_hq_model(), ck, tmp_path / "ok.json", dino_ckpt_sha256="a" * 64)
    assert rep["status"] == "OK" and rep["consistency_checks"]["dino_ckpt_sha256_equal"] is True
    with pytest.raises(RuntimeError, match="DINOv2 weights differ"):
        load_stage2_strict(make_hq_model(), ck, tmp_path / "bad.json", dino_ckpt_sha256="b" * 64)
    assert json.loads((tmp_path / "bad.json").read_text())["consistency_checks"]["dino_ckpt_sha256_equal"] is False
    rep = load_stage2_strict(make_hq_model(), ck, tmp_path / "none.json")      # hash not given: not checked
    assert rep["consistency_checks"]["dino_ckpt_sha256_equal"] is None


def test_strict_load_copies_every_stage2_weight(stage2_model, stage2_ckpt, tmp_path):
    hq = make_hq_model()
    hq_before = {k: v.clone() for k, v in hq.hq_mask_decoder.hq_state_dict().items()}
    rep = load_stage2_strict(hq, stage2_ckpt, tmp_path / "report.json")
    assert rep["status"] == "OK" and rep["errors"] == []
    assert rep["loaded"]["adapter"] == 44
    assert rep["consistency_checks"]["prompt_encoder_equal"] is True
    for name in ("adapter", "semantic_branch", "corr_branch"):
        a, b = getattr(hq, name).state_dict(), getattr(stage2_model, name).state_dict()
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    dec, hq_dec = stage2_model.sam_full.mask_decoder.state_dict(), hq.hq_mask_decoder.state_dict()
    assert all(torch.equal(hq_dec[k], dec[k]) for k in dec)
    # HQ modules are not part of Stage 2 and keep their initialisation
    assert all(torch.equal(v, hq_before[k]) for k, v in hq.hq_mask_decoder.hq_state_dict().items())
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "OK"


def test_strict_load_refuses_adapter_convention_mismatch(stage2_ckpt, tmp_path):
    wrong = make_hq_model(norm_first=False)
    with pytest.raises(RuntimeError, match="adapter convention mismatch"):
        load_stage2_strict(wrong, stage2_ckpt, tmp_path / "report.json")
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "FAILED"


def _mutate(sd, kind):
    sd = dict(sd)
    if kind == "missing_component_key":
        del sd["adapter.proj_out.bias"]
    elif kind == "missing_decoder_key":
        del sd["sam_full.mask_decoder.iou_token.weight"]
    elif kind == "unexpected_component_key":
        sd["semantic_branch.extra.weight"] = torch.zeros(3)
    elif kind == "unexpected_prefix":
        sd["fusion_gate.weight"] = torch.zeros(3)
    elif kind == "unexpected_decoder_key":
        sd["sam_full.mask_decoder.hf_token.weight"] = torch.zeros(1, 256)
    elif kind == "shape_mismatch":
        sd["corr_branch.prior_head.0.weight"] = torch.zeros(2, 1, 3, 3)
    elif kind == "prompt_encoder_drift":
        k = "sam_full.prompt_encoder.no_mask_embed.weight"
        sd[k] = sd[k] + 1.0
    return sd


@pytest.mark.parametrize("kind", ["missing_component_key", "missing_decoder_key",
                                  "unexpected_component_key", "unexpected_prefix",
                                  "unexpected_decoder_key", "shape_mismatch",
                                  "prompt_encoder_drift"])
def test_strict_load_refuses_contract_violations(stage2_model, tmp_path, kind):
    ck = save_stage2_checkpoint(tmp_path / "bad.pt", stage2_model,
                                state=_mutate(stage2_model.stage2_state_dict(), kind))
    with pytest.raises(RuntimeError, match="stage-2 strict load FAILED"):
        load_stage2_strict(make_hq_model(), ck, tmp_path / "report.json")
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["status"] == "FAILED" and rep["errors"]


def test_strict_load_refuses_recorded_corr_arch_mismatch(stage2_model, tmp_path):
    ck = save_stage2_checkpoint(tmp_path / "bad.pt", stage2_model, corr_arch="v3_geometric")
    with pytest.raises(RuntimeError, match="corr_arch=v3_geometric"):
        load_stage2_strict(make_hq_model(), ck, tmp_path / "report.json")


# ── Stage 2 selection gate ───────────────────────────────────────────────────

def test_gate_accepts_the_published_selection(stage2_model, tmp_path):
    run = make_selected_run(tmp_path / "run", stage2_model)
    for arg in (run, run / "stage2_selected.pt", str(run)):
        prov = verify_selected_stage2(arg)
        assert prov["provenance"] == POST_TRAINING_SELECTION
        assert Path(prov["stage2_ckpt"]) == (run / "stage2_selected.pt").resolve()
        assert prov["file_sha256"] == file_sha256(run / "stage2_selected.pt")
        assert prov["selected_epoch"] == 7 and prov["selection_plan"] == "val5"
        assert prov["state_sha256"] == model_state_sha256(stage2_model.stage2_state_dict())


def test_gate_refuses_a_failed_selection(stage2_model, tmp_path):
    run = make_selected_run(tmp_path / "run", stage2_model, status="FAILED")
    with pytest.raises(Stage2SelectionError, match="status is 'FAILED'"):
        verify_selected_stage2(run)


def test_gate_refuses_a_changed_artifact(stage2_model, tmp_path):
    run = make_selected_run(tmp_path / "run", stage2_model)
    with open(run / "stage2_selected.pt", "ab") as f:
        f.write(b"\0")
    with pytest.raises(Stage2SelectionError, match="changed since selection"):
        verify_selected_stage2(run)


def test_gate_refuses_sidecar_report_disagreement(stage2_model, tmp_path):
    run = make_selected_run(tmp_path / "run", stage2_model)
    side = json.loads((run / "stage2_selected.json").read_text())
    side["state_sha256"] = "0" * 64
    (run / "stage2_selected.json").write_text(json.dumps(side))
    with pytest.raises(Stage2SelectionError, match="does not match the report's winner"):
        verify_selected_stage2(run)


@pytest.mark.parametrize("name", ["stage2_selected.json", "stage2_selection_report.json",
                                  "stage2_selected.pt"])
def test_gate_refuses_a_missing_file(stage2_model, tmp_path, name):
    run = make_selected_run(tmp_path / "run", stage2_model)
    (run / name).unlink()
    with pytest.raises(Stage2SelectionError, match="no completed Stage 2 selection"):
        verify_selected_stage2(run)


def test_gate_refuses_other_checkpoints_of_the_run(stage2_model, tmp_path):
    run = make_selected_run(tmp_path / "run", stage2_model)
    with pytest.raises(Stage2SelectionError, match="neither a Stage 2 run directory"):
        verify_selected_stage2(run / "stage2_best_full_val.pt")
    with pytest.raises(Stage2SelectionError, match="does not exist"):
        verify_selected_stage2(tmp_path / "no_such_run")


# ── SAM-HQ partial load ──────────────────────────────────────────────────────

def test_sam_hq_partial_load_mapping(tmp_path):
    dec = new_hq_decoder()
    sd = fake_sam_hq_state(dec)
    ck = tmp_path / "sam_hq_vit_h.pth"
    torch.save(sd, ck)
    before = {k: v.clone() for k, v in dec.state_dict().items()}
    rep = load_sam_hq_partial(SimpleNamespace(hq_mask_decoder=dec), ck, save_dir=tmp_path / "out")

    after = dec.state_dict()
    mapping = {"hf_token.": "mask_decoder.hf_token.", "hf_mlp.": "mask_decoder.hf_mlp.",
               "adapter_proj.": "mask_decoder.embedding_encoder.",
               "decoder_mask_proj.": "mask_decoder.embedding_maskfeature."}
    loaded = 0
    for k, v in after.items():
        dst = next((d for d in mapping if k.startswith(d)), None)
        if dst is not None:
            assert torch.equal(v, sd[mapping[dst] + k[len(dst):]]), k
            loaded += 1
        else:                                # base decoder, dino_l3_proj, early_proj
            assert torch.equal(v, before[k]), k
    assert loaded == rep["loaded_count"] == 19
    assert sorted({k.split(".")[0] for k in rep["random_init_keys"]}) == ["dino_l3_proj", "early_proj"]
    assert len(rep["random_init_keys"]) == 12
    assert json.loads((tmp_path / "out" / "sam_hq_loading_report.json").read_text())["loaded_count"] == 19


def test_sam_hq_partial_load_refuses_a_file_that_would_leave_hq_modules_random(tmp_path):
    dec = new_hq_decoder()
    sd = fake_sam_hq_state(dec)
    before = {k: v.clone() for k, v in dec.state_dict().items()}
    wrong_shape = dict(sd)
    wrong_shape["mask_decoder.hf_token.weight"] = torch.zeros(1, 128)
    torch.save(wrong_shape, tmp_path / "shape.pth")
    with pytest.raises(ValueError, match="hf_token.weight has shape"):
        load_sam_hq_partial(SimpleNamespace(hq_mask_decoder=dec), tmp_path / "shape.pth")
    plain_sam = {k: v for k, v in sd.items() if "hf_" not in k and "embedding_" not in k}
    torch.save(plain_sam, tmp_path / "sam.pth")                 # e.g. a SAM file passed by mistake
    with pytest.raises(ValueError, match="not a usable SAM-HQ"):
        load_sam_hq_partial(SimpleNamespace(hq_mask_decoder=dec), tmp_path / "sam.pth")
    assert all(torch.equal(v, before[k]) for k, v in dec.state_dict().items())    # nothing was loaded


# ── early adapter ────────────────────────────────────────────────────────────

def test_early_adapter_loads_strictly_and_frozen(tmp_path):
    torch.manual_seed(4)
    src = DistillationAdapter(in_channels=768, out_channels=256)
    ck = tmp_path / "early_adapter_best.pt"
    torch.save({"early_adapter_state": src.state_dict(), "config": {"adapter": "early"},
                "epoch": 30, "val_mse": 8.5, "val_cosine": 0.85}, ck)
    ea, meta = load_early_adapter(ck, torch.device("cpu"))
    assert not ea.training and not any(p.requires_grad for p in ea.parameters())
    assert all(torch.equal(v, src.state_dict()[k]) for k, v in ea.state_dict().items())
    assert meta == {"early_adapter_epoch": 30, "early_adapter_val_mse": 8.5,
                    "early_adapter_val_cosine": 0.85}

    torch.save({"adapter_state": src.state_dict(), "config": {"adapter": "final"}}, tmp_path / "final.pt")
    with pytest.raises(KeyError, match="early_adapter_state"):
        load_early_adapter(tmp_path / "final.pt", torch.device("cpu"))
    torch.save({"early_adapter_state": src.state_dict(), "config": {"adapter": "final"}},
               tmp_path / "mislabelled.pt")
    with pytest.raises(ValueError, match="not an early adapter"):
        load_early_adapter(tmp_path / "mislabelled.pt", torch.device("cpu"))
    bad = dict(src.state_dict()); bad.pop("proj_out.bias")
    torch.save({"early_adapter_state": bad}, tmp_path / "incomplete.pt")
    with pytest.raises(RuntimeError, match="proj_out.bias"):
        load_early_adapter(tmp_path / "incomplete.pt", torch.device("cpu"))
