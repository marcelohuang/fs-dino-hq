"""Stage 2 training: adapter initialisation, freeze/gradient policy, one-shot and
K-shot forward, checkpoint format, configuration handling and the trainer end
to end (one-shot with --resume, five-shot with two validation plans).

CPU only: stub DINOv2, real SAM prompt encoder / mask decoder modules with
random weights, tiny synthetic LabelMe dataset."""
import json
import os

import pytest
import torch
import torch.nn.functional as F

from stubs import StubDino, StubSam, make_tiny_photo_dataset

from fs_dino_hq.modeling.distill_adapter import BottleneckAdapter as StudentAdapter
from fs_dino_hq.modeling.model import BottleneckAdapter
from fs_dino_hq.modeling.stage2 import (Stage2Model, apply_stage2_freeze, stage2_loss,
                                        stage2_param_audit)
from fs_dino_hq.training import train_stage2 as trainer
from fs_dino_hq.training.train_stage2 import (EXPECTED_TRAINABLE_PARAMS, gradient_path_check,
                                              init_adapter_from_distilled, kshot_checks,
                                              save_stage2_checkpoint)

torch.set_num_threads(2)

SLIM_KEYS = {"format", "trainer", "epoch", "stage2_model", "stats", "val", "selection_tag",
             "adapter_norm_first", "adapter_dropout", "corr_arch", "feat_hw", "dino_excluded",
             "sam_image_encoder_excluded", "config"}
LAST_EXTRA_KEYS = {"optimizer_state_dict", "rng", "trackers", "histories"}


# ── adapter initialisation ──────────────────────────────────────────────────

def _student():
    torch.manual_seed(1)
    student = StudentAdapter(in_channels=768, out_channels=256).eval()
    with torch.no_grad():                   # non-trivial BN statistics
        for m in student.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(); m.running_var.uniform_(0.5, 2.0)
    return student


def _final_adapter_file(path, student, adapter_kind="final"):
    torch.save({"adapter_state": student.state_dict(),
                "config": {"adapter": adapter_kind, "save_dir": "runs/distill_final"},
                "stage": 2, "res": 518, "epoch_in_stage": 14, "epoch": 30,
                "selected_by": "min diag@518 mse", "val_mse": 0.1, "val_cosine": 0.9}, path)
    return path


def test_init_remaps_layer_infix_strictly_and_carries_the_student_function(tmp_path):
    student = _student()
    assert any(".layer." in k for k in student.state_dict())
    ck = _final_adapter_file(tmp_path / "final.pt", student)
    a = BottleneckAdapter(norm_first=True, dropout=0.0).eval()
    rep = init_adapter_from_distilled(a, ck)
    assert rep["n_tensors"] == 44 and rep["remap"] == ".layer. -> ." and rep["state_key"] == "adapter_state"
    assert rep["provenance"]["res"] == 518 and rep["provenance"]["epoch_in_stage"] == 14
    assert rep["loaded_into"] == {"norm_first": True, "dropout": 0.0}
    x = torch.randn(1, 768, 37, 37)
    with torch.no_grad():
        assert torch.allclose(student(x), a(x), atol=1e-5), "pre-norm adapter must equal the distilled student"


def test_init_refuses_early_adapter_files_and_non_strict_states(tmp_path):
    student = _student()
    a = BottleneckAdapter(norm_first=True, dropout=0.0)
    bad = tmp_path / "early.pt"
    torch.save({"early_adapter_state": student.state_dict(), "config": {"adapter": "early"}}, bad)
    with pytest.raises(KeyError):
        init_adapter_from_distilled(a, bad)
    _final_adapter_file(bad, student, adapter_kind="early")
    with pytest.raises(ValueError):
        init_adapter_from_distilled(a, bad)
    sd = student.state_dict(); sd.pop(next(iter(sd)))
    torch.save({"adapter_state": sd, "config": {"adapter": "final"}}, bad)
    with pytest.raises(RuntimeError):            # strict load: a missing tensor is an error
        init_adapter_from_distilled(a, bad)


# ── model: freeze policy, gradient paths, forward equivalences ──────────────

@pytest.fixture(scope="module")
def stage2_model():
    torch.manual_seed(0)
    m = Stage2Model(StubDino(), StubSam(), adapter_norm_first=True, adapter_dropout=0.0)
    apply_stage2_freeze(m)
    return m


def _kbatch(B=2, K=5, S=518):
    torch.manual_seed(11)
    qm = torch.zeros(B, 1, S, S); qm[:, :, 100:300, 50:400] = 1
    sm = torch.zeros(B, K, 1, S, S)
    for k in range(K):
        sm[:, k, :, 60 + 20 * k:250, 40 + 10 * k:300] = 1
    return {"support_dino": torch.randn(B, K, 3, S, S), "support_mask": sm,
            "query_dino": torch.randn(B, 3, S, S), "query_mask": qm}


def test_trainable_set_and_count(stage2_model):
    audit = stage2_param_audit(stage2_model)
    assert audit["trainable_top_modules"] == ["adapter", "corr_branch", "sam_full.mask_decoder", "semantic_branch"]
    assert audit["trainable_params"] == EXPECTED_TRAINABLE_PARAMS == 8_596_985
    assert audit["by_module"]["sam_full.prompt_encoder"]["trainable"] == 0
    assert audit["by_module"]["last_extractor"]["trainable"] == 0
    assert "layer3_extractor" not in audit["by_module"], "shared DINOv2 must be counted once"
    assert stage2_model.last_extractor.model.training is False
    assert stage2_model.sam_full.prompt_encoder.training is False
    assert stage2_model.adapter.training and stage2_model.sam_full.mask_decoder.training


def test_gradient_paths_reach_all_trainable_modules_and_no_frozen_one(stage2_model):
    apply_stage2_freeze(stage2_model)
    out = stage2_model(_kbatch(K=1))
    assert out["final_logits"].shape == (2, 1, 256, 256) and out["prior_logits"].shape == (2, 1, 37, 37)
    loss, lp, lf = stage2_loss(out)
    assert torch.allclose(loss, lp + lf)
    stage2_model.zero_grad(set_to_none=True)
    loss.backward()
    rep = gradient_path_check(stage2_model)
    assert rep["ok"], rep
    assert all(v["grad_norm"] > 0 for v in rep["modules"].values()) and not rep["frozen_with_grad"]
    assert all(p.grad is None for p in stage2_model.last_extractor.model.parameters())
    assert all(p.grad is None for p in stage2_model.sam_full.prompt_encoder.parameters())
    stage2_model.zero_grad(set_to_none=True)


def _manual_one_shot(model, batch):
    """The one-shot computation of the original Stage 2 training script, written out."""
    sd, sm, q = batch["support_dino"][:, 0], batch["support_mask"][:, 0], batch["query_dino"]
    with torch.no_grad():
        Fs, Fq, Fq_l3 = model.last_extractor(sd), model.last_extractor(q), model.layer3_extractor(q)
    mf = F.interpolate(sm.float(), size=Fs.shape[-2:], mode="nearest")
    sem = model.semantic_branch(Fs, mf); _, S = model.pcs_branch(Fs, mf, Fq)
    _, _, prior, _, dense = model.corr_branch(Fs, Fq, S)
    emb = F.interpolate(model.adapter(Fq_l3), size=(64, 64), mode="bilinear", align_corners=False)
    pe = model.sam_full.prompt_encoder.get_dense_pe()
    fin = torch.cat([model.sam_full.mask_decoder(image_embeddings=emb[i:i + 1], image_pe=pe,
                                                 sparse_prompt_embeddings=sem[i:i + 1],
                                                 dense_prompt_embeddings=dense[i:i + 1], multimask_output=False)[0]
                     for i in range(emb.shape[0])])
    return prior, fin


def test_one_shot_forward_equals_the_original_one_shot_computation(stage2_model):
    stage2_model.eval()
    b = _kbatch(K=1)
    with torch.no_grad():
        out = stage2_model(b); prior, fin = _manual_one_shot(stage2_model, b)
        out4 = stage2_model({**b, "support_dino": b["support_dino"][:, 0], "support_mask": b["support_mask"][:, 0]})
    assert out["n_shot"] == 1
    assert torch.allclose(out["prior_logits"], prior, atol=1e-6) and torch.allclose(out["final_logits"], fin, atol=1e-5)
    assert torch.equal(out4["final_logits"], out["final_logits"])      # [B,3,S,S] input is K = 1
    apply_stage2_freeze(stage2_model)


def test_five_shot_prompt_mean_equals_kshot_aggregation_and_every_support_matters(stage2_model):
    from fs_dino_hq.modeling.kshot import aggregate_prompts, compute_support_prompts
    stage2_model.eval()
    b = _kbatch(K=5)
    with torch.no_grad():
        out = stage2_model(b)
        for i in range(2):                    # per episode, inference path of fs_dino_hq.modeling.kshot
            Fq = stage2_model.last_extractor(b["query_dino"][i:i + 1])
            agg = aggregate_prompts(compute_support_prompts(stage2_model, b["support_dino"][i], b["support_mask"][i], Fq))
            for k in ("semantic_prompt", "dense_prompt", "prior_logits"):
                assert torch.allclose(agg[k], out[k][i:i + 1], atol=1e-5), k
    apply_stage2_freeze(stage2_model)
    rep = kshot_checks(stage2_model, b, torch.device("cpu"))
    assert rep["ok"], rep
    c = rep["checks"]
    assert c["image_embedding_ignores_supports"] == 0.0 and len(c["per_support_feature_grad_norm"]) == 5
    assert c["multishot_dense_max_abs_diff"] < 1e-4 and c["order_invariant_final_logits"] < 1e-4
    assert c["order_invariant_dense_prompt"] < 1e-6 and c["order_invariant_final_mask_iou"] > 0.999

    class OrderDependent(torch.nn.Module):     # prompts that change with support order must be rejected
        def __init__(self, m):
            super().__init__(); self.m = m

        def __getattr__(self, n):
            try:
                return super().__getattr__(n)
            except AttributeError:
                return getattr(self.m, n)

        def forward(self, batch, **kw):
            out = dict(self.m(batch, **kw))
            w = torch.linspace(0.5, 1.5, batch["support_dino"].shape[1]).view(1, -1, 1, 1, 1)
            out["dense_prompt"] = out["dense_prompt"] + (batch["support_mask"].float() * w).mean((1, 2, 3, 4)).view(-1, 1, 1, 1)
            return out

    bad = kshot_checks(OrderDependent(stage2_model), b, torch.device("cpu"))
    assert not bad["ok"] and bad["checks"]["order_invariant_dense_prompt"] > 1e-5
    apply_stage2_freeze(stage2_model)
    out = stage2_model(b); loss, _, _ = stage2_loss(out)
    stage2_model.zero_grad(set_to_none=True); loss.backward()
    g = gradient_path_check(stage2_model); assert g["ok"], g
    stage2_model.zero_grad(set_to_none=True)


def test_slim_checkpoint_format(stage2_model, tmp_path):
    ck = tmp_path / "stage2_best.pt"
    save_stage2_checkpoint(ck, stage2_model, 3, {"loss": 1.0, "loss_prior": 0.5, "loss_final": 0.5},
                           {"fast": None, "full": None}, "best_full_val", {"x": 1})
    obj = torch.load(ck, map_location="cpu", weights_only=False)
    assert set(obj) == SLIM_KEYS
    assert obj["format"] == "stage2_model" and obj["trainer"] == "fs_dino_hq.training.train_stage2"
    assert obj["corr_arch"] == "v2_reduced" and obj["feat_hw"] == 37 and obj["epoch"] == 3
    assert obj["adapter_norm_first"] is True and obj["adapter_dropout"] == 0.0
    assert obj["dino_excluded"] is True and obj["sam_image_encoder_excluded"] is True
    sd = obj["stage2_model"]
    assert not any(k.startswith(("last_extractor.", "layer3_extractor.")) for k in sd)
    assert not any(k.startswith("sam_full.image_encoder") for k in sd)
    assert sum(k.startswith("adapter.") for k in sd) == 44
    assert sd["corr_branch.reduce_conv1.0.weight"].shape == (512, 1369, 1, 1)
    assert not (tmp_path / "stage2_best.tmp").exists()


# ── configuration ───────────────────────────────────────────────────────────

def test_config_file_env_expansion_cli_precedence_and_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("FSDHQ_TEST_DATA", str(tmp_path / "data"))
    cfg = {"synthetic_root": "${FSDHQ_TEST_DATA}/photo", "train_plan": "t.csv",
           "val_plans": ["val5=v5.csv:5", "val1=v1.csv:1"], "select_on": "val5",
           "adapter_ckpt": "a.pt", "dino_ckpt": "d.pth", "sam_ckpt": "s.pth", "save_dir": "out",
           "n_shot": 5, "save_every": 10, "save_epochs": [50, 100], "num_workers": 6}
    p = tmp_path / "c.json"; p.write_text(json.dumps(cfg))
    args = trainer.parse_args(["--config", str(p), "--num-workers", "0"])
    assert args.synthetic_root == os.path.join(str(tmp_path / "data"), "photo")
    assert args.num_workers == 0 and args.n_shot == 5 and args.save_epochs == [50, 100]
    assert args.val_plan_specs == [("val5", "v5.csv", 5), ("val1", "v1.csv", 1)]
    assert args.val_plan == "v5.csv" and args.select_on == "val5"
    assert args.adapter_norm_first is True and args.adapter_dropout == 0.0
    # a single --val-plan becomes val1 (the one-shot configuration)
    one = trainer.parse_args(["--config", str(p), "--val-plans", "--val-plan", "v1.csv", "--select-on", "val1"])
    assert one.val_plan_specs == [("val1", "v1.csv", 1)]
    with pytest.raises(SystemExit):                       # unknown --select-on label
        trainer.parse_args(["--config", str(p), "--select-on", "val3"])
    p.write_text(json.dumps({**cfg, "cldice_weight": 0.1}))
    with pytest.raises(ValueError, match="unknown keys"):  # removed options are refused
        trainer.parse_args(["--config", str(p)])
    with pytest.raises(SystemExit):                       # required values missing
        trainer.parse_args(["--train-plan", "t.csv"])
    assert trainer.parse_plan_spec("x.csv") == ("val1", "x.csv", 1)
    assert trainer.parse_plan_spec("v=a:b/c.csv:5") == ("v", "a:b/c.csv", 5)


# ── trainer end to end ──────────────────────────────────────────────────────

@pytest.fixture()
def tiny_setup(tmp_path, monkeypatch):
    root = tmp_path / "PHOTO"
    plans = make_tiny_photo_dataset(root)
    adapter_ck = _final_adapter_file(tmp_path / "final_adapter_best.pt", _student())
    monkeypatch.setattr(trainer, "load_backbones", lambda d, s: (StubDino(), StubSam()))
    return root, plans, adapter_ck


def test_trainer_one_shot_end_to_end_with_resume(tmp_path, tiny_setup):
    root, plans, adapter_ck = tiny_setup
    save_dir = tmp_path / "run"
    common = ["--synthetic-root", str(root), "--train-plan", str(plans[("train", 1)]),
              "--val-plan", str(plans[("val", 1)]), "--adapter-ckpt", str(adapter_ck),
              "--dino-ckpt", "unused", "--sam-ckpt", "unused", "--save-dir", str(save_dir),
              "--num-workers", "0", "--batch-size", "2", "--smoke", "--smoke-episodes", "4", "--seed", "7"]
    assert trainer.main(common + ["--epochs", "1"]) == 0
    sel = json.loads((save_dir / "checkpoint_selection_report.json").read_text())
    assert sel["primary_epoch"] == 1 and (save_dir / "stage2_best.pt").is_file()
    assert (save_dir / "stage2_best.pt").read_bytes() == (save_dir / "stage2_best_full_val.pt").read_bytes()
    smoke = json.loads((save_dir / "smoke_report.json").read_text())
    assert smoke["ok"] and smoke["trainable_params_match"] and smoke["preprocess"]["ok"]
    assert smoke["kshot"] is None and smoke["gradient_paths"]["ok"]
    assert smoke["preprocess"]["checks"]["bottom_pad_is_black"] is True  # landscape images pad at the bottom
    assert json.loads((save_dir / "frozen_check.json").read_text())["unchanged"]
    cfg = json.loads((save_dir / "config.json").read_text())
    assert cfg["trainer"] == "fs_dino_hq.training.train_stage2" and cfg["corr_arch"] == "v2_reduced"
    assert cfg["adapter_arch"]["norm_first"] is True and cfg["adapter_init"]["provenance"]["epoch_in_stage"] == 14
    assert cfg["val_plans"] == [f"val1={plans[('val', 1)]}:1"] and cfg["select_on"] == "val1"
    assert cfg["trainable_params"] == EXPECTED_TRAINABLE_PARAMS
    assert "training/train_stage2.py" in cfg["code_sha256"]
    assert not any(k.startswith("cldice") for k in cfg) and "aux_loss" not in cfg
    hdr = (save_dir / "history_fast.csv").read_text().splitlines()[0].split(",")
    assert hdr[:4] == ["epoch", "plan_offset", "train_loss", "train_loss_prior"]
    for c in ("loss", "loss_prior", "loss_final", "iou_final", "iou_prior", "episodes", "seconds"):
        assert f"train_{c}" in hdr and f"fast_val_{c}" in hdr
    full_hdr = (save_dir / "history_full.csv").read_text().splitlines()[0]
    assert "full_val_loss" in full_hdr and "full_val_val1_loss" not in full_hdr   # single plan: plain names only
    last1 = torch.load(save_dir / "stage2_last.pt", map_location="cpu", weights_only=False)
    assert set(last1) == SLIM_KEYS | LAST_EXTRA_KEYS and last1["selection_tag"] == "last"
    # a second start without --resume must refuse to overwrite
    with pytest.raises(SystemExit):
        trainer.main(common + ["--epochs", "2"])
    assert trainer.main(common + ["--epochs", "2", "--resume"]) == 0
    sel2 = json.loads((save_dir / "checkpoint_selection_report.json").read_text())
    hist = (save_dir / "history_fast.csv").read_text().strip().splitlines()
    assert len(hist) == 3 and hist[1].startswith("1,") and hist[2].startswith("2,")
    assert sel2["last_epoch"] == 2 and sel2["primary_epoch"] in (1, 2)
    cfg2 = json.loads((save_dir / "config.json").read_text())
    assert cfg2["resumed_from"][0]["epoch"] == 1
    last = torch.load(save_dir / "stage2_last.pt", map_location="cpu", weights_only=False)
    assert last["epoch"] == 2 and last["trackers"]["best_full_epoch"] == sel2["primary_epoch"]
    assert len(last["histories"]["fast"]) == 2 and len(last["histories"]["full"]) == 2
    # the plan offset continues across the resume (4 episodes per epoch over a 6-row plan)
    assert [int(r.split(",")[1]) for r in hist[1:]] == [0, 4]


def test_resume_continues_the_same_trajectory(tmp_path, tiny_setup):
    """Two epochs in one go == one epoch + --resume (same weights at the end)."""
    root, plans, adapter_ck = tiny_setup
    base = ["--synthetic-root", str(root), "--train-plan", str(plans[("train", 1)]),
            "--val-plan", str(plans[("val", 1)]), "--adapter-ckpt", str(adapter_ck),
            "--dino-ckpt", "unused", "--sam-ckpt", "unused", "--num-workers", "0", "--batch-size", "2",
            "--episodes-per-epoch", "2", "--val-episodes", "2", "--full-val-every", "0",
            "--best-selection", "fast_val", "--seed", "5"]
    trainer.main(base + ["--save-dir", str(tmp_path / "a"), "--epochs", "2"])
    trainer.main(base + ["--save-dir", str(tmp_path / "b"), "--epochs", "1"])
    trainer.main(base + ["--save-dir", str(tmp_path / "b"), "--epochs", "2", "--resume"])
    sa = torch.load(tmp_path / "a" / "stage2_last.pt", map_location="cpu", weights_only=False)["stage2_model"]
    sb = torch.load(tmp_path / "b" / "stage2_last.pt", map_location="cpu", weights_only=False)["stage2_model"]
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)   # CPU: bitwise
    # a resume with another schedule would silently train a different run: refused
    other = [("4" if a == "2" and base[i - 1] == "--episodes-per-epoch" else a) for i, a in enumerate(base)]
    with pytest.raises(SystemExit, match="REFUSING to resume"):
        trainer.main(other + ["--save-dir", str(tmp_path / "b"), "--epochs", "3", "--resume"])


def test_trainer_five_shot_with_two_validation_plans(tmp_path, tiny_setup):
    root, plans, adapter_ck = tiny_setup
    save_dir = tmp_path / "run5"
    v5, v1 = str(plans[("val", 5)]), str(plans[("val", 1)])
    assert trainer.main(["--synthetic-root", str(root), "--train-plan", str(plans[("train", 5)]), "--n-shot", "5",
                         "--val-plans", f"val5={v5}:5", f"val1={v1}:1", "--select-on", "val5",
                         "--adapter-ckpt", str(adapter_ck), "--dino-ckpt", "unused", "--sam-ckpt", "unused",
                         "--save-dir", str(save_dir), "--num-workers", "0", "--batch-size", "2", "--smoke",
                         "--smoke-episodes", "2", "--seed", "7", "--epochs", "2", "--save-every", "1"]) == 0
    sel = json.loads((save_dir / "checkpoint_selection_report.json").read_text())
    assert sel["select_on"] == "val5" and sel["n_shot_train"] == 5 and set(sel["val_plans"]) == {"val5", "val1"}
    smoke = json.loads((save_dir / "smoke_report.json").read_text())
    assert smoke["ok"] and smoke["kshot"]["ok"] and smoke["kshot"]["n_shot"] == 5
    hdr = (save_dir / "history_full.csv").read_text().splitlines()[0]
    assert "full_val_val5_loss" in hdr and "full_val_val1_loss" in hdr and "full_val_loss" in hdr
    cfg = json.loads((save_dir / "config.json").read_text())
    assert cfg["episodes"]["n_shot"] == 5 and cfg["episodes"]["select_on"] == "val5"
    assert cfg["episodes"]["kshot_aggregation"].endswith("fs_dino_hq.modeling.kshot")
    assert (save_dir / "stage2_epoch_001.pt").is_file() and (save_dir / "stage2_epoch_002.pt").is_file()
    ck = torch.load(save_dir / "stage2_epoch_002.pt", map_location="cpu", weights_only=False)
    assert ck["val"]["select_on"] == "val5" and set(ck["val"]["full_all"]) == {"val5", "val1"}
    assert ck["val"]["full"] == ck["val"]["full_all"]["val5"]
    # the full-val numbers recorded at training time are reproduced by evaluate_checkpoint
    from torch.utils.data import DataLoader
    from fs_dino_hq.data.episodes import PlanEpisodeDataset, collate_plan_episodes
    from fs_dino_hq.data.letterbox import LetterboxPreprocess
    torch.manual_seed(7)                  # = the run's --seed: same stub DINOv2 (its conv bias uses the global RNG)
    model = Stage2Model(StubDino(), StubSam(), adapter_norm_first=True, adapter_dropout=0.0)
    apply_stage2_freeze(model)
    ds = PlanEpisodeDataset(plans[("val", 5)], root, 5, LetterboxPreprocess(518))
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_plan_episodes)
    res = trainer.evaluate_checkpoint(model, save_dir / "stage2_epoch_002.pt", loader, torch.device("cpu"))
    assert res["episodes"] == len(ds) and res["epoch"] == 2
    assert abs(res["loss"] - ck["val"]["full"]["loss"]) < 1e-4
