"""HQ-branch training: dual_canonical decoder semantics, the HQ-only freeze
policy and an end-to-end run of ``fs-dino-hq train-hq`` on stub backbones.

CPU only. DINOv2 and SAM ViT-H are replaced by the stubs of tests/stubs.py
(the SAM stub has the real prompt encoder and mask decoder at ViT-H sizes, so
the HQ parameter count is the real one).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from stubs import StubDino, StubSam, make_tiny_photo_dataset
from test_checkpoints import (fake_sam_hq_state, make_hq_model, make_selected_run,
                              make_stage2_model, new_hq_decoder, save_stage2_checkpoint)

from fs_dino_hq.modeling.distill_adapter import BottleneckAdapter as DistillationAdapter
from fs_dino_hq.modeling.hq_decoder import HQMaskDecoder
from fs_dino_hq.modeling.kshot import forward_kshot
from fs_dino_hq.training import train_hq
from fs_dino_hq.training.freeze import (apply_hq_freeze, build_hq_optimizer,
                                        check_frozen_unchanged, snapshot_frozen_params,
                                        verify_freeze)
from fs_dino_hq.training.losses import hq_loss
from fs_dino_hq.utils.provenance import hq_state_sha256

torch.set_num_threads(4)

EXPECTED_HQ_PARAMS = 1_145_056
HQ_MODULES = ["hq_mask_decoder.adapter_proj", "hq_mask_decoder.decoder_mask_proj",
              "hq_mask_decoder.dino_l3_proj", "hq_mask_decoder.early_proj",
              "hq_mask_decoder.hf_mlp", "hq_mask_decoder.hf_token"]


def decoder_inputs(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)          # noqa: E731
    return dict(image_embeddings=r(1, 256, 64, 64), image_pe=r(1, 256, 64, 64),
                sparse_prompt_embeddings=r(1, 1, 256), dense_prompt_embeddings=r(1, 256, 64, 64),
                dino_l3_feat=r(1, 768, 37, 37), adapter_feat=r(1, 256, 64, 64),
                distilled_early_feat=r(1, 256, 64, 64))


# ── decoder semantics (HQMaskDecoder alone) ──────────────────────────────────

def test_corrected_is_canonical_base_plus_hq_residual():
    out = new_hq_decoder().predict_dual_canonical(**decoder_inputs())
    assert torch.equal(out["corrected_logits"],
                       out["canonical_base_logits"] + 1.0 * out["hq_logits"])
    assert torch.equal(out["corrected_logits"], out["canonical_base_logits"] + out["hq_logits"])
    assert not out["canonical_base_logits"].requires_grad          # detached base
    assert out["corrected_logits"].shape == (1, 1, 256, 256)


def test_canonical_base_is_unaffected_by_hq_parameters():
    dec, kw = new_hq_decoder(), decoder_inputs()
    before = dec.predict_dual_canonical(**kw)
    with torch.no_grad():
        for _, p in dec.hq_named_parameters():
            p.add_(torch.randn_like(p))
    after = dec.predict_dual_canonical(**kw)
    assert torch.equal(after["canonical_base_logits"], before["canonical_base_logits"])
    assert torch.equal(after["iou_pred"], before["iou_pred"])
    assert not torch.equal(after["hq_logits"], before["hq_logits"])


def test_one_step_changes_hq_tensors_only():
    dec, kw = new_hq_decoder(), decoder_inputs()
    hq_names = {n for n, _ in dec.hq_named_parameters()}
    for n, p in dec.named_parameters():
        p.requires_grad_(n in hq_names)
    base_before = {k: v.clone() for k, v in dec.state_dict().items() if not HQMaskDecoder.is_hq_key(k)}
    hq_before = {k: v.clone() for k, v in dec.hq_state_dict().items()}
    opt = torch.optim.AdamW([p for p in dec.parameters() if p.requires_grad], lr=1e-3)
    out = dec.predict_dual_canonical(**kw)
    out["corrected_logits"].pow(2).mean().backward()
    g = dec.hf_token.weight.grad
    assert g is not None and torch.isfinite(g).all() and float(g.norm()) > 0
    assert dec.dino_l3_proj.convs[0].weight.grad is None   # early path uses early_proj
    opt.step()
    assert all(torch.equal(v, dec.state_dict()[k]) for k, v in base_before.items())
    changed = {k.split(".")[0] for k, v in dec.hq_state_dict().items() if not torch.equal(v, hq_before[k])}
    assert changed == {"hf_token", "hf_mlp", "adapter_proj", "early_proj", "decoder_mask_proj"}


def test_hq_parameter_count_and_deterministic_initial_hash():
    dec = new_hq_decoder()
    assert sum(p.numel() for _, p in dec.hq_named_parameters()) == EXPECTED_HQ_PARAMS
    assert hq_state_sha256(new_hq_decoder(5)) == hq_state_sha256(new_hq_decoder(5))
    assert hq_state_sha256(new_hq_decoder(5)) != hq_state_sha256(new_hq_decoder(6))


# ── freeze policy on the full model ──────────────────────────────────────────

@pytest.fixture()
def hq_model():
    torch.manual_seed(0)
    m = make_hq_model()
    m.early_adapter = DistillationAdapter(in_channels=768, out_channels=256)
    return m


def test_freeze_policy_trains_hq_modules_only(hq_model):
    apply_hq_freeze(hq_model)
    audit = verify_freeze(hq_model)
    assert audit["trainable_params"] == EXPECTED_HQ_PARAMS
    assert audit["trainable_top_modules"] == HQ_MODULES

    hq_model.train()
    apply_hq_freeze(hq_model)               # as after every model.train() in the trainer
    verify_freeze(hq_model)
    for name, mod in hq_model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            assert not mod.training, name   # adapters, MVPG: running statistics fixed
    assert all(m.training for n, m in hq_model.hq_mask_decoder.named_modules()
               if n and n.split(".")[0] in HQMaskDecoder.HQ_MODULE_NAMES)

    opt = build_hq_optimizer(hq_model, lr=1e-4, weight_decay=1e-4)
    hq_params = [p for n, p in hq_model.named_parameters() if n.startswith(tuple(HQ_MODULES))]
    assert len(opt.param_groups) == 1
    assert [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in hq_params]
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["lr"] == 1e-4 and opt.defaults["weight_decay"] == 1e-4
    assert opt.defaults["betas"] == (0.9, 0.999) and opt.defaults["eps"] == 1e-8


def test_verify_freeze_detects_violations(hq_model):
    apply_hq_freeze(hq_model)
    hq_model.adapter.proj_out.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="non-HQ parameters trainable"):
        verify_freeze(hq_model)
    apply_hq_freeze(hq_model)
    hq_model.semantic_branch.train()
    with pytest.raises(RuntimeError, match="frozen modules in train mode"):
        verify_freeze(hq_model)


def test_kshot_forward_loss_reaches_hq_modules_only(hq_model):
    apply_hq_freeze(hq_model)
    hq_model.train(); apply_hq_freeze(hq_model)
    g = torch.Generator().manual_seed(2)
    batch = {"support_dino": torch.randn(1, 2, 3, 518, 518, generator=g),
             "support_mask": (torch.rand(1, 2, 1, 518, 518, generator=g) > 0.5).float(),
             "query_dino": torch.randn(1, 3, 518, 518, generator=g),
             "query_mask": (torch.rand(1, 1, 518, 518, generator=g) > 0.5).float()}
    snap = snapshot_frozen_params(hq_model)
    opt = build_hq_optimizer(hq_model)
    out = forward_kshot(hq_model, batch, torch.device("cpu"))
    assert torch.equal(out["corrected_logits"], out["canonical_base_logits"] + out["hq_logits"])
    assert out["final_logits"] is out["corrected_logits"]
    hq_loss(out).backward()
    with_grad = {".".join(n.split(".")[:2]) for n, p in hq_model.named_parameters()
                 if p.grad is not None and float(p.grad.abs().sum()) > 0}
    # dino_l3_proj is trainable but not on the computation path (distilled early feature)
    assert with_grad == set(HQ_MODULES) - {"hq_mask_decoder.dino_l3_proj"}
    opt.step()
    assert check_frozen_unchanged(hq_model, snap) == (True, [])


# ── end-to-end trainer ───────────────────────────────────────────────────────

@pytest.fixture()
def hq_inputs(tmp_path, monkeypatch):
    """Tiny dataset, a selected Stage 2 run, a fake SAM-HQ file, an early adapter;
    DINOv2 and SAM ViT-H loaders replaced by stubs."""
    root = tmp_path / "PHOTO"
    plans = make_tiny_photo_dataset(root)
    s2 = make_stage2_model()
    run = make_selected_run(tmp_path / "stage2_run", s2)
    torch.manual_seed(9)
    early = DistillationAdapter(in_channels=768, out_channels=256)
    early_ck = tmp_path / "early_adapter_best.pt"
    torch.save({"early_adapter_state": early.state_dict(), "config": {"adapter": "early"},
                "epoch": 30, "val_mse": 8.5, "val_cosine": 0.85}, early_ck)
    sam_hq = tmp_path / "sam_hq_vit_h.pth"
    torch.save(fake_sam_hq_state(new_hq_decoder()), sam_hq)
    for n in ("dino.pth", "sam.pth"):                 # only hashed by the trainer
        (tmp_path / n).write_bytes(n.encode())
    monkeypatch.setattr(train_hq, "load_dinov2_vitb14", lambda p, device=None, hub_repo=None: StubDino())
    monkeypatch.setattr(train_hq, "load_sam_vit_h", lambda p, device=None: StubSam())
    args = ["--synthetic-root", str(root), "--train-plan", str(plans[("train", 5)]),
            "--val-plan", str(plans[("val", 5)]), "--dino-ckpt", str(tmp_path / "dino.pth"),
            "--sam-ckpt", str(tmp_path / "sam.pth"), "--sam-hq-ckpt", str(sam_hq),
            "--early-adapter-ckpt", str(early_ck), "--num-workers", "0", "--batch-size", "2"]
    return {"args": args, "run": run, "stage2": s2, "early": early, "sam_hq": sam_hq,
            "tmp": tmp_path}


EXPECTED_CKPT_KEYS = {
    "epoch", "model_state", "hq_state", "hq_module_names", "decoder_mode", "corrected_formula",
    "train_logits_source", "residual_scale_mode", "residual_scale_value",
    "residual_scale_initial_value", "n_shot", "early_feature_source", "adapter_norm_first",
    "adapter_dropout", "hq_arch_meta", "dino_excluded", "train_loss", "fast_val_loss",
    "full_val_loss", "selection_tag", "config", "loss", "updates"}


def test_train_hq_end_to_end(hq_inputs, monkeypatch):
    out = hq_inputs["tmp"] / "hq"
    ret = train_hq.main(hq_inputs["args"] + [
        "--stage2-run", str(hq_inputs["run"]), "--save-dir", str(out), "--epochs", "2",
        "--episodes-per-epoch", "4", "--val-episodes", "2", "--full-val-every", "1"])
    assert ret == 0
    for name in ("config.json", "freeze_audit.json", "stage2_input_provenance.json",
                 "stage2_selection_report.json", "stage2_ckpt_inspect.json",
                 "stage2_loading_report.json", "sam_hq_loading_report.json",
                 "checkpoint_selection_report.json", "hq_last.pt", "hq_best_fast_val.pt",
                 "hq_best_full_val.pt", "hq_best.pt", "hq_epoch_1.pt", "hq_epoch_2.pt"):
        assert (out / name).is_file(), name

    prov = json.loads((out / "stage2_input_provenance.json").read_text())
    assert prov["provenance"].startswith("post-training selection") and prov["selection_report_copied"]
    assert json.loads((out / "stage2_loading_report.json").read_text())["status"] == "OK"
    assert json.loads((out / "sam_hq_loading_report.json").read_text())["loaded_count"] == 19
    fz = json.loads((out / "freeze_audit.json").read_text())
    assert fz["trainable_params"] == EXPECTED_HQ_PARAMS and fz["trainable_top_modules"] == HQ_MODULES

    fast = (out / "loss_history_fast.csv").read_text().splitlines()
    full = (out / "loss_history_full.csv").read_text().splitlines()
    assert fast[0] == "epoch,updates,train_loss,train_loss_bce,train_loss_dice,fast_val_loss,plan_offset"
    assert full[0] == "epoch,updates,full_val_loss,n_episodes"
    assert [r.split(",")[-1] for r in fast[1:]] == ["0", "4"]          # 6-row plan, 4 per epoch
    assert [r.split(",")[-1] for r in full[1:]] == ["6", "6"]
    assert [r.split(",")[1] for r in fast[1:]] == ["2", "4"]           # 4 episodes at batch 2 = 2 updates
    # default loss: point-sampled BCE + Dice; one row per optimizer update with both terms
    steps = [r.split(",") for r in (out / "train_steps.csv").read_text().splitlines()]
    assert steps[0] == ["update", "epoch", "loss", "loss_bce", "loss_dice"]
    assert [(r[0], r[1]) for r in steps[1:]] == [("1", "1"), ("2", "1"), ("3", "2"), ("4", "2")]
    assert all(abs(float(r[2]) - float(r[3]) - float(r[4])) < 1e-3 * abs(float(r[2])) for r in steps[1:])
    e1 = fast[1].split(",")
    assert abs(float(e1[2]) - float(e1[3]) - float(e1[4])) < 1e-3 * abs(float(e1[2]))

    cfg = json.loads((out / "config.json").read_text())
    mp = cfg["matched_provenance"]
    assert mp["trainable_params"] == EXPECTED_HQ_PARAMS and mp["n_shot"] == 5
    assert mp["stage2_ckpt_sha256"] == prov["file_sha256"] == cfg["stage2_ckpt_sha256"]
    assert len(mp["initial_hq_state_sha256"]) == 64 and mp["best_selection"] == "full_val"
    assert (mp["w_final"], mp["w_prior"], mp["w_hq"]) == (1.0, 0.0, 0.0)
    assert mp["loss"] == cfg["loss"] == "point_sampled"
    assert cfg["loss_config"] == {"training": "point_sampled", "validation": "full_grid",
                                  "supervised_logits": "corrected", "num_points": 12544,
                                  "oversample_ratio": 3.0, "importance_sample_ratio": 0.75,
                                  "target_size": 1024, "point_seed": 42 + 7919}
    assert not [k for k in list(cfg) + list(mp) if "cldice" in k or "photometric" in k]
    assert cfg["decoder_mode"] == "dual_canonical" and cfg["residual_scale_mode"] == "fixed"
    assert cfg["adapter_arch"] == {"norm_first": True, "dropout": 0.0}

    sel = json.loads((out / "checkpoint_selection_report.json").read_text())
    assert sel["best_selection_rule"] == "full_val" and sel["primary_checkpoint"] == "hq_best_full_val.pt"
    assert (out / "hq_best.pt").read_bytes() == (out / "hq_best_full_val.pt").read_bytes()

    best = torch.load(out / "hq_best.pt", map_location="cpu", weights_only=True)
    assert set(best) == EXPECTED_CKPT_KEYS
    assert best["decoder_mode"] == "dual_canonical" and best["train_logits_source"] == "corrected"
    assert best["residual_scale_mode"] == "fixed" and best["residual_scale_value"] == 1.0
    assert best["early_feature_source"] == "distilled_adapter" and best["n_shot"] == 5
    assert best["adapter_norm_first"] is True and best["adapter_dropout"] == 0.0
    assert best["selection_tag"] == "best_full_val" and best["epoch"] == sel["primary_epoch"]
    assert best["loss"] == "point_sampled" and best["updates"] == 2 * best["epoch"]
    assert best["hq_module_names"] == sorted(HQMaskDecoder.HQ_MODULE_NAMES)
    ms = best["model_state"]
    assert not any(k.startswith(("last_extractor.", "layer3_extractor.")) for k in ms)
    assert all(torch.equal(v, ms[k]) for k, v in best["hq_state"].items())

    # Stage 2 weights unchanged by HQ training; early adapter embedded unchanged
    for k, v in hq_inputs["stage2"].stage2_state_dict().items():
        dst = (k.replace("sam_full.mask_decoder.", "hq_mask_decoder.")
                .replace("sam_full.prompt_encoder.", "prompt_encoder."))
        assert torch.equal(ms[dst], v), k
    for k, v in hq_inputs["early"].state_dict().items():
        assert torch.equal(ms["early_adapter." + k], v), k

    # dino_l3_proj never changes; the trained HQ modules do
    e1 = torch.load(out / "hq_epoch_1.pt", map_location="cpu", weights_only=True)["hq_state"]
    e2 = torch.load(out / "hq_epoch_2.pt", map_location="cpu", weights_only=True)["hq_state"]
    for k in e1:
        same = torch.equal(e1[k], e2[k])
        assert same == (".dino_l3_proj." in k), k

    # the checkpoint loads strictly with the inference loader
    from fs_dino_hq.inference import predictor
    monkeypatch.setattr(predictor, "load_dinov2_vitb14", lambda p, device=None, hub_repo=None: StubDino())
    model = predictor.load_model(out / "hq_best.pt", "unused", device=torch.device("cpu"))
    sd = model.state_dict()
    assert all(torch.equal(sd[k], v) for k, v in ms.items())


def test_train_hq_refuses_to_overwrite_a_trained_run(hq_inputs):
    out = hq_inputs["tmp"] / "hq"
    out.mkdir()
    (out / "hq_best_full_val.pt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="REFUSING"):
        train_hq.main(hq_inputs["args"] + ["--stage2-run", str(hq_inputs["run"]),
                                           "--save-dir", str(out)])


def test_train_hq_smoke_with_config_and_explicit_stage2(hq_inputs, monkeypatch):
    tmp = hq_inputs["tmp"]
    explicit = save_stage2_checkpoint(tmp / "stage2_epoch_200.pt", hq_inputs["stage2"])
    monkeypatch.setenv("HQ_TEST_OUT", str(tmp))
    cfg = tmp / "train_hq.json"
    cfg.write_text(json.dumps({"_comment": "test", "save_dir": "${HQ_TEST_OUT}/smoke",
                               "stage2_ckpt": str(explicit), "allow_unselected_stage2": True,
                               "n_shot": 1, "epochs": 3}))
    args = hq_inputs["args"]
    args[args.index("--train-plan") + 1] = args[args.index("--train-plan") + 1].replace("5shot", "1shot")
    args[args.index("--val-plan") + 1] = args[args.index("--val-plan") + 1].replace("5shot", "1shot")
    assert train_hq.main(args + ["--config", str(cfg), "--smoke", "--epochs", "7"]) == 0
    out = tmp / "smoke"
    rep = json.loads((out / "smoke_report.json").read_text())
    assert rep["ok"] and rep["frozen_unchanged_after_step"] and rep["grads_finite"]
    assert rep["modules_with_gradient"] == sorted(set(HQ_MODULES) - {"hq_mask_decoder.dino_l3_proj"})
    assert rep["hq_modules_changed"] == ["adapter_proj", "decoder_mask_proj", "early_proj",
                                         "hf_mlp", "hf_token"]
    prov = json.loads((out / "stage2_input_provenance.json").read_text())
    assert prov["provenance"].startswith("explicit checkpoint") and not prov["selection_report_copied"]
    assert not (out / "stage2_selection_report.json").exists()
    c = json.loads((out / "config.json").read_text())
    assert c["n_shot"] == 1 and c["epochs"] == 7            # config file value, CLI wins
    assert c["stage2_input"]["provenance"] == prov["provenance"]
    assert not list(out.glob("hq_*.pt"))


@pytest.mark.parametrize("case", ["no_allow_flag", "both_inputs", "no_input", "failed_selection",
                                  "changed_artifact"])
def test_train_hq_stage2_gate_refusals(hq_inputs, case):
    run, tmp = hq_inputs["run"], hq_inputs["tmp"]
    args = hq_inputs["args"] + ["--save-dir", str(tmp / "hq"), "--smoke"]
    if case == "no_allow_flag":
        args += ["--stage2-ckpt", str(run / "stage2_best_full_val.pt")]
    elif case == "both_inputs":
        args += ["--stage2-run", str(run), "--stage2-ckpt", str(run / "stage2_best_full_val.pt"),
                 "--allow-unselected-stage2"]
    elif case == "failed_selection":
        rep = json.loads((run / "stage2_selection_report.json").read_text())
        rep["status"] = "FAILED"
        (run / "stage2_selection_report.json").write_text(json.dumps(rep))
        args += ["--stage2-run", str(run)]
    elif case == "changed_artifact":
        with open(run / "stage2_selected.pt", "ab") as f:
            f.write(b"\0")
        args += ["--stage2-run", str(run)]
    with pytest.raises(SystemExit) as e:
        train_hq.main(args)
    assert e.value.code != 0
    assert not (tmp / "hq" / "config.json").exists()


def test_full_grid_loss_and_point_loss_differ_only_in_the_training_objective(hq_inputs):
    """--loss full_grid: the training loss is the validation loss function; same model
    initialisation, data order and checkpoint layout as the default."""
    runs = {}
    for loss in ("point_sampled", "full_grid"):
        out = hq_inputs["tmp"] / f"hq_{loss}"
        assert train_hq.main(hq_inputs["args"] + [
            "--stage2-run", str(hq_inputs["run"]), "--save-dir", str(out), "--epochs", "1",
            "--episodes-per-epoch", "2", "--val-episodes", "2", "--full-val-every", "1",
            "--loss", loss]) == 0
        runs[loss] = (json.loads((out / "config.json").read_text()),
                      (out / "train_steps.csv").read_text().splitlines()[1].split(","),
                      torch.load(out / "hq_epoch_1.pt", map_location="cpu", weights_only=True))
    (cp, sp, kp), (cg, sg, kg) = runs["point_sampled"], runs["full_grid"]
    assert cp["matched_provenance"]["initial_hq_state_sha256"] == cg["matched_provenance"]["initial_hq_state_sha256"]
    assert cg["loss_config"] == {"training": "full_grid", "validation": "full_grid", "supervised_logits": "corrected"}
    assert kg["loss"] == "full_grid" and set(kg) == set(kp) == EXPECTED_CKPT_KEYS
    assert sp[2] != sg[2]                                   # different objective on the same first batch
    assert abs(float(sg[2]) - float(sg[3]) - float(sg[4])) < 1e-3 * abs(float(sg[2]))
    assert not all(torch.equal(v, kg["hq_state"][k]) for k, v in kp["hq_state"].items())
