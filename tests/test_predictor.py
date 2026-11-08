"""Predictor, checkpoint loading and the predict / evaluate / compare commands
on a tiny synthetic folder (stub DINOv2, randomly initialised weights)."""
import csv
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from stubs import StubDino

import fs_dino_hq.inference.predictor as predictor_mod
from fs_dino_hq.data.labelme import labelme_json_to_mask
from fs_dino_hq.data.letterbox import LetterboxPreprocess, unletterbox_prob
from fs_dino_hq.evaluation import compare as compare_cmd
from fs_dino_hq.evaluation import evaluate as evaluate_cmd
from fs_dino_hq.inference import predict as predict_cmd
from fs_dino_hq.inference.predictor import (DINO_PREFIXES, Predictor, check_hq_checkpoint,
                                            load_model)
from fs_dino_hq.modeling.backbones import build_sam_prompt_decoder
from fs_dino_hq.modeling.distill_adapter import BottleneckAdapter as DistillationAdapter
from fs_dino_hq.modeling.kshot import aggregate_prompts, compute_support_prompts
from fs_dino_hq.modeling.model import FSDinoHQModel
from fs_dino_hq.utils.provenance import file_sha256

DINO_SEED = 0


def _stub_dino():
    # StubDino's patch-embedding bias comes from the global generator: fix it.
    torch.manual_seed(1234)
    return StubDino(seed=DINO_SEED)


def _fake_loader(ckpt, device=None, hub_repo=None):
    return _stub_dino()


@pytest.fixture(autouse=True)
def stub_dinov2(monkeypatch):
    monkeypatch.setattr(predictor_mod, "load_dinov2_vitb14", _fake_loader)


def _trained_like_model():
    """FSDinoHQModel with non-default weights everywhere (so loading is visible)."""
    torch.manual_seed(7)
    model = FSDinoHQModel(_stub_dino(), build_sam_prompt_decoder(), adapter_norm_first=True,
                          adapter_dropout=0.0)
    model.early_adapter = DistillationAdapter(in_channels=768, out_channels=256)
    g = torch.Generator().manual_seed(8)
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if k.startswith(DINO_PREFIXES) or not v.is_floating_point():
                continue
            if k.endswith("running_var"):
                v.copy_(0.5 + torch.rand(v.shape, generator=g))
            else:
                v.add_(0.02 * torch.randn(v.shape, generator=g))
    return model.eval()


def _checkpoint(model, **meta):
    state = {
        "epoch": 3,
        "model_state": {k: v.clone() for k, v in model.state_dict().items()
                        if not k.startswith(DINO_PREFIXES)},
        "decoder_mode": "dual_canonical",
        "residual_scale_mode": "fixed",
        "residual_scale_value": 1.0,
        "early_feature_source": "distilled_adapter",
        "adapter_norm_first": True,
        "adapter_dropout": 0.0,
        "n_shot": 2,
        "dino_excluded": True,
        "selection_tag": "best_full_val",
        "full_val_loss": 0.25,
        "hq_arch_meta": {"corr_arch": "v2_reduced", "corr_grid": 37, "transformer_dim": 256},
    }
    state.update(meta)
    return state


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    model = _trained_like_model()
    path = tmp_path_factory.mktemp("ckpt") / "hq.pt"
    torch.save(_checkpoint(model), path)
    return model, path


@pytest.fixture(scope="module")
def loaded(trained):
    import fs_dino_hq.inference.predictor as pm
    orig = pm.load_dinov2_vitb14
    pm.load_dinov2_vitb14 = _fake_loader          # module-scoped fixture: patch by hand
    try:
        return load_model(trained[1], "dinov2.pth", device=torch.device("cpu"))
    finally:
        pm.load_dinov2_vitb14 = orig


def _img(w, h, seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def _mask(w, h, box):
    m = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 1
    return m


# ── checkpoint loading ──────────────────────────────────────────────────────

def test_load_model_restores_every_tensor(trained, loaded):
    model, path = trained
    ref = model.state_dict()
    got = loaded.state_dict()
    assert set(ref) == set(got)
    for k in ref:
        assert torch.equal(ref[k], got[k]), k
    assert not loaded.training
    assert loaded.adapter.norm_first is True
    assert loaded.checkpoint_meta["decoder_mode"] == "dual_canonical"
    assert loaded.checkpoint_meta["n_shot"] == 2 and loaded.checkpoint_meta["epoch"] == 3


def test_load_model_refuses_unexpected_and_missing_keys(trained, tmp_path):
    model, _ = trained
    state = _checkpoint(model)
    state["model_state"]["hq_mask_decoder.fusion_gates"] = torch.zeros(3)
    p = tmp_path / "unexpected.pt"
    torch.save(state, p)
    with pytest.raises(RuntimeError, match="unexpected"):
        load_model(p, "dinov2.pth", device=torch.device("cpu"))
    p.unlink()
    state = _checkpoint(model)
    del state["model_state"]["adapter.proj_out.weight"]
    p = tmp_path / "missing.pt"
    torch.save(state, p)
    with pytest.raises(RuntimeError, match="missing"):
        load_model(p, "dinov2.pth", device=torch.device("cpu"))
    p.unlink()


def test_load_model_refuses_wrong_decoder_mode(tmp_path):
    p = tmp_path / "shared.pt"
    torch.save({"model_state": {"corr_branch.reduce_conv1.0.weight": torch.zeros(512, 1369, 1, 1),
                                "early_adapter.proj_out.weight": torch.zeros(1)},
                "decoder_mode": "shared_pass", "residual_scale_mode": "fixed",
                "early_feature_source": "distilled_adapter"}, p)
    with pytest.raises(ValueError, match="decoder_mode"):
        load_model(p, "dinov2.pth", device=torch.device("cpu"))


@pytest.mark.parametrize("change,match", [
    ({"decoder_mode": None}, "decoder_mode"),
    ({"decoder_mode": "shared_pass"}, "decoder_mode"),
    ({"residual_scale_mode": "learned"}, "residual scale"),
    ({"early_feature_source": "raw_dino_l3"}, "early adapter"),
])
def test_check_hq_checkpoint_metadata(change, match):
    state = {"model_state": {"corr_branch.reduce_conv1.0.weight": torch.zeros(512, 1369, 1, 1),
                             "early_adapter.proj_out.weight": torch.zeros(1)},
             "decoder_mode": "dual_canonical", "residual_scale_mode": "fixed",
             "early_feature_source": "distilled_adapter"}
    check_hq_checkpoint(state)
    with pytest.raises(ValueError, match=match):
        check_hq_checkpoint({**state, **change})


def test_check_hq_checkpoint_tensors():
    meta = {"decoder_mode": "dual_canonical", "residual_scale_mode": "fixed",
            "early_feature_source": "distilled_adapter"}
    with pytest.raises(ValueError, match="not an HQ checkpoint"):
        check_hq_checkpoint(meta)
    with pytest.raises(ValueError, match="early_adapter"):
        check_hq_checkpoint({**meta, "model_state": {
            "corr_branch.reduce_conv1.0.weight": torch.zeros(512, 1369, 1, 1)}})
    with pytest.raises(ValueError, match="not supported"):
        check_hq_checkpoint({**meta, "model_state": {
            "corr_branch.reduce_support.0.weight": torch.zeros(1),
            "early_adapter.proj_out.weight": torch.zeros(1)}})
    with pytest.raises(ValueError, match="not supported"):
        check_hq_checkpoint({**meta, "model_state": {
            "corr_branch.reduce_conv1.0.weight": torch.zeros(512, 25, 1, 1),
            "early_adapter.proj_out.weight": torch.zeros(1)}})


# ── Predictor ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def supports():
    imgs = [_img(80, 60, 1), _img(50, 70, 2)]
    masks = [_mask(80, 60, (10, 10, 50, 30)), _mask(50, 70, (5, 20, 40, 60))]
    return imgs, masks


@pytest.fixture(scope="module")
def prediction(loaded, supports):
    return Predictor(loaded).predict(_img(90, 40, 3), *supports)


def test_prediction_shapes_dtypes_range(prediction):
    r = prediction
    for prob, mask in ((r.probability, r.mask), (r.base_probability, r.base_mask)):
        assert prob.shape == (40, 90) and prob.dtype == np.float32
        assert mask.shape == (40, 90) and mask.dtype == np.uint8
        assert prob.min() >= 0.0 and prob.max() <= 1.0
        assert np.array_equal(prob * 255, np.round(prob * 255))          # multiples of 1/255
        assert np.array_equal(mask, (prob >= 0.5).astype(np.uint8))
        assert set(np.unique(mask)) <= {0, 1}
    assert set(r.logits) == {"corrected_logits", "canonical_base_logits", "hq_logits", "sam_logits"}
    for v in r.logits.values():
        assert v.shape == (1, 1, 256, 256) and v.device.type == "cpu"
    assert torch.equal(r.logits["corrected_logits"],
                       r.logits["canonical_base_logits"] + r.logits["hq_logits"])


def test_prediction_matches_reference_algorithm(loaded, supports, prediction):
    """Same numbers as the original inference loop (HQ pass, then canonical pass)."""
    m, prep = loaded, LetterboxPreprocess(518)
    q = _img(90, 40, 3)
    with torch.no_grad():
        qd = prep.image_for_dino(q)[None]
        Fq, Fq_l3 = m.last_extractor(qd), m.layer3_extractor(qd)
        sd = torch.stack([prep.image_for_dino(s) for s in supports[0]])
        sm = torch.stack([prep.mask_for_dino(x) for x in supports[1]])
        p = aggregate_prompts(compute_support_prompts(m, sd, sm, Fq))
        a64 = F.interpolate(m.adapter(Fq_l3), (64, 64), mode="bilinear", align_corners=False)
        ea = F.interpolate(m.early_adapter(Fq_l3), (64, 64), mode="bilinear", align_corners=False)
        pe = m.prompt_encoder.get_dense_pe()
        hq = m.hq_mask_decoder.predict_with_hq(
            image_embeddings=a64, image_pe=pe, sparse_prompt_embeddings=p["semantic_prompt"],
            dense_prompt_embeddings=p["dense_prompt"], dino_l3_feat=Fq_l3, adapter_feat=a64,
            distilled_early_feat=ea)
        cb = m.hq_mask_decoder.predict_canonical_stage2_base(
            image_embeddings=a64, image_pe=pe, sparse_prompt_embeddings=p["semantic_prompt"],
            dense_prompt_embeddings=p["dense_prompt"])
        logits = cb["canonical_base_logits"] + m.hq_mask_decoder.residual_scale * hq["hq_logits"]
    prob = unletterbox_prob(torch.sigmoid(logits)[0, 0].numpy(), 90, 40, 518)
    base = unletterbox_prob(torch.sigmoid(cb["canonical_base_logits"])[0, 0].numpy(), 90, 40, 518)
    assert np.array_equal(prediction.probability, prob)
    assert np.array_equal(prediction.base_probability, base)


def test_threshold_and_inputs(loaded, supports, prediction, tmp_path):
    probs = np.unique(prediction.probability)
    assert len(probs) > 1
    thr = float(probs[len(probs) // 2])                    # a threshold hit exactly by some pixels
    r = Predictor(loaded, threshold=thr).predict(_img(90, 40, 3), *supports)
    assert np.array_equal(r.mask, (r.probability >= thr).astype(np.uint8))
    assert 0 < r.mask.sum() < r.mask.size
    assert (r.probability == thr).any() and r.mask[r.probability == thr].all()
    # paths and boolean / 0-255 masks give the same result as PIL images and {0,1} masks
    paths = []
    for i, im in enumerate(supports[0]):
        paths.append(tmp_path / f"s{i}.png")
        im.save(paths[-1])
    _img(90, 40, 3).save(tmp_path / "q.png")
    r2 = Predictor(loaded).predict(tmp_path / "q.png", paths,
                                   [supports[1][0].astype(bool), supports[1][1] * 255])
    assert np.array_equal(r2.probability, prediction.probability)


def test_predictor_input_checks(loaded, supports):
    pred = Predictor(loaded)
    imgs, masks = supports
    with pytest.raises(ValueError, match="does not match"):
        pred.predict(_img(90, 40, 3), imgs, [masks[0], masks[0]])          # wrong mask size
    with pytest.raises(ValueError, match="does not match"):
        pred.predict(_img(90, 40, 3), imgs[:1], [masks[0].T])
    with pytest.raises(ValueError, match="same, non-zero number"):
        pred.predict(_img(90, 40, 3), imgs, masks[:1])
    with pytest.raises(ValueError, match="same, non-zero number"):
        pred.predict(_img(90, 40, 3), [], [])


def test_predictor_puts_model_in_eval_mode(loaded):
    loaded.train()
    try:
        assert not Predictor(loaded).model.training
    finally:
        loaded.eval()


# ── predict / evaluate / compare commands ───────────────────────────────────

def _labelme(path, w, h, boxes, label="spacing"):
    shapes = [{"label": label, "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}
              for x0, y0, x1, y1 in boxes]
    path.write_text(json.dumps({"imageWidth": w, "imageHeight": h, "imagePath": path.stem + ".png",
                                "shapes": shapes}))


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    sup, qry = root / "supports", root / "queries"
    sup.mkdir(), qry.mkdir()
    for i, (stem, w, h) in enumerate((("s_a", 80, 60), ("s_b", 60, 80), ("s_c", 70, 70))):
        _img(w, h, 10 + i).save(sup / f"{stem}.png")
        _labelme(sup / f"{stem}.json", w, h, [(5, 5, 40, 30), (10, 40, 30, 55)])
    (sup / "support_set.json").write_text(json.dumps([["s_b", "s_a"]]))
    _img(96, 64, 20).save(qry / "q1.png")
    _labelme(qry / "q1.json", 96, 64, [(10, 10, 60, 30)])
    _img(64, 80, 21).save(qry / "q2.jpg", quality=95)
    _labelme(qry / "q2.json", 128, 160, [(20, 20, 100, 60), (30, 100, 90, 150)])  # 2x the image
    _img(50, 50, 22).save(qry / "q3.png")                                        # no annotation
    return sup, qry


@pytest.fixture(scope="module")
def predicted(trained, folder, tmp_path_factory):
    _, ckpt = trained
    sup, qry = folder
    out = tmp_path_factory.mktemp("pred")
    import fs_dino_hq.inference.predictor as pm
    orig = pm.load_dinov2_vitb14
    pm.load_dinov2_vitb14 = _fake_loader
    try:
        rc = predict_cmd.main(["--hq-ckpt", str(ckpt), "--dino-ckpt", "dinov2.pth",
                               "--support-dir", str(sup), "--support-set", str(sup / "support_set.json"),
                               "--query-dir", str(qry), "--output-dir", str(out), "--device", "cpu",
                               "--save-base", "--save-visualizations"])
    finally:
        pm.load_dinov2_vitb14 = orig
    assert rc == 0
    return out


def test_predict_cli_outputs(predicted, folder, trained):
    out = predicted
    sup, qry = folder
    sizes = {"q1": (96, 64), "q2": (64, 80), "q3": (50, 50)}
    for stem, size in sizes.items():
        prob = Image.open(out / "pred_prob" / f"{stem}.png")
        mask = Image.open(out / "pred_mask" / f"{stem}.png")
        assert prob.mode == "L" and mask.mode == "L" and prob.size == mask.size == size
        p, m = np.array(prob), np.array(mask)
        assert set(np.unique(m)) <= {0, 255}
        assert np.array_equal(m, (p >= 128).astype(np.uint8) * 255)     # threshold 0.5 on 8 bit
        bp, bm = np.array(Image.open(out / "base_prob" / f"{stem}.png")), \
            np.array(Image.open(out / "base_mask" / f"{stem}.png"))
        assert np.array_equal(bm, (bp >= 128).astype(np.uint8) * 255)
        assert (out / "visualizations" / f"{stem}_overlay.png").is_file()
        assert (out / "visualizations" / f"{stem}_compare.png").is_file()
    assert (out / "visualizations" / "supports.png").is_file()
    with open(out / "predictions.csv", newline="") as f:
        rows = {r["query"]: r for r in csv.DictReader(f)}
    assert set(rows) == {"q1", "q2", "q3"}
    assert rows["q1"]["iou"] and rows["q2"]["iou"] and rows["q3"]["iou"] == ""
    cfg = json.loads((out / "inference_config.json").read_text())
    assert cfg["n_supports"] == 2 and cfg["n_queries"] == 3
    assert [s["image"].rsplit("/", 1)[-1] for s in cfg["supports"]] == ["s_b.png", "s_a.png"]
    assert cfg["hq_checkpoint_sha256"] == file_sha256(trained[1])
    assert cfg["checkpoint_meta"]["decoder_mode"] == "dual_canonical"


def test_predict_cli_equals_predictor(predicted, folder, loaded):
    sup, qry = folder
    imgs = [Image.open(sup / f"{s}.png").convert("RGB") for s in ("s_b", "s_a")]
    masks = [labelme_json_to_mask(sup / f"{s}.json") for s in ("s_b", "s_a")]
    r = Predictor(loaded).predict(qry / "q1.png", imgs, masks)
    assert np.array_equal(np.array(Image.open(predicted / "pred_prob" / "q1.png")),
                          (r.probability * 255).astype(np.uint8))
    assert np.array_equal(np.array(Image.open(predicted / "pred_mask" / "q1.png")), r.mask * 255)


def test_evaluate_and_compare_cli(predicted, folder, tmp_path):
    _, qry = folder
    for sub in ("pred_mask", "base_mask"):
        assert evaluate_cmd.main(["--pred-dir", str(predicted / sub), "--gt-dir", str(qry),
                                  "--out-csv", str(tmp_path / f"{sub}.csv"),
                                  "--out-json", str(tmp_path / f"{sub}.json")]) == 0
    summary = json.loads((tmp_path / "pred_mask.json").read_text())
    assert summary["n_evaluated"] == 2 and summary["n_skipped"] == 1
    assert "q3.png: no annotation" in summary["skipped"]
    with open(tmp_path / "pred_mask.csv", newline="") as f:
        ev = {r["image"]: r for r in csv.DictReader(f)}
    with open(predicted / "predictions.csv", newline="") as f:
        pr = {r["query"]: r for r in csv.DictReader(f)}
    for q in ("q1", "q2"):            # q2: annotation at twice the image size, both resize the mask
        assert float(ev[q]["iou"]) == pytest.approx(float(pr[q]["iou"]), abs=1e-12)
        assert float(ev[q]["dice"]) == pytest.approx(float(pr[q]["dice"]), abs=1e-12)
    out = tmp_path / "cmp.json"
    assert compare_cmd.main(["--base", str(tmp_path / "base_mask.csv"),
                             "--new", str(tmp_path / "pred_mask.csv"),
                             "--n-boot", "200", "--seed", "1", "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert rec["results"]["n_queries"] == 2
    with open(tmp_path / "base_mask.csv", newline="") as f:
        base = {r["image"]: float(r["iou"]) for r in csv.DictReader(f)}
    expected = np.mean([float(ev[q]["iou"]) - base[q] for q in ("q1", "q2")])
    assert rec["results"]["iou"]["delta"] == pytest.approx(expected)
    # same seed, same result
    out2 = tmp_path / "cmp2.json"
    compare_cmd.main(["--base", str(tmp_path / "base_mask.csv"), "--new", str(tmp_path / "pred_mask.csv"),
                      "--n-boot", "200", "--seed", "1", "--out", str(out2)])
    assert json.loads(out2.read_text())["results"] == rec["results"]


def test_compare_refuses_different_queries(tmp_path):
    hdr = "image,iou,dice,boundary_f,thin_recall,fragmentation_ratio\n"
    (tmp_path / "a.csv").write_text(hdr + "x,0.5,0.6,0.4,0.3,2\ny,0.4,0.5,0.3,0.2,3\n")
    (tmp_path / "b.csv").write_text(hdr + "x,0.6,0.7,0.5,0.4,1\nz,0.4,0.5,0.3,0.2,3\n")
    with pytest.raises(SystemExit):
        compare_cmd.main(["--base", str(tmp_path / "a.csv"), "--new", str(tmp_path / "b.csv")])
    assert compare_cmd.main(["--base", str(tmp_path / "a.csv"), "--new", str(tmp_path / "b.csv"),
                             "--allow-different-queries", "--n-boot", "50"]) == 0
    (tmp_path / "c.csv").write_text("image,iou\nx,0.5\n")
    with pytest.raises(SystemExit, match="missing columns"):
        compare_cmd.main(["--base", str(tmp_path / "c.csv"), "--new", str(tmp_path / "a.csv")])


def test_predict_cli_input_checks(folder, tmp_path):
    sup, qry = folder
    base = ["--hq-ckpt", "x.pt", "--dino-ckpt", "d.pth", "--support-dir", str(sup),
            "--output-dir", str(tmp_path / "o")]
    dup = tmp_path / "dup"
    dup.mkdir()
    _img(20, 20, 0).save(dup / "a.png")
    _img(20, 20, 1).save(dup / "a.jpg")
    with pytest.raises(SystemExit, match="share a stem"):
        predict_cmd.main(base + ["--query-dir", str(dup)])
    with pytest.raises(SystemExit, match="also supports"):
        predict_cmd.main(base + ["--query-dir", str(sup)])
    bad_set = tmp_path / "set.json"
    bad_set.write_text(json.dumps(["s_a", "nope"]))
    with pytest.raises(SystemExit, match="nope"):
        predict_cmd.main(base + ["--query-dir", str(qry), "--support-set", str(bad_set)])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no query images"):
        predict_cmd.main(base + ["--query-dir", str(empty)])


def test_read_support_set_formats(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps(["a", "b"]))
    assert predict_cmd.read_support_set(f) == ["a", "b"]
    f.write_text(json.dumps([["a", "b", "c"]]))
    assert predict_cmd.read_support_set(f) == ["a", "b", "c"]
    for bad in ([["a"], ["b"]], ["a", "a"], [], [1, 2]):
        f.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            predict_cmd.read_support_set(f)


def test_evaluate_strip_suffix_and_duplicates(tmp_path):
    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "pred"
    gt_dir.mkdir(), pred_dir.mkdir()
    _labelme(gt_dir / "x.json", 40, 30, [(5, 5, 20, 20)])
    m = np.zeros((30, 40), np.uint8)
    m[5:21, 5:21] = 255
    Image.fromarray(m).save(pred_dir / "x_corrected_bin.png")
    rows, skipped = evaluate_cmd.evaluate_dir(pred_dir, gt_dir, strip_suffix="_corrected_bin")
    assert [r["image"] for r in rows] == ["x"] and rows[0]["iou"] == 1.0 and not skipped
    rows, skipped = evaluate_cmd.evaluate_dir(pred_dir, gt_dir)
    assert not rows and skipped == ["x_corrected_bin.png: no annotation"]
    Image.fromarray(m).save(pred_dir / "x.png")
    with pytest.raises(SystemExit, match="both map to image"):
        evaluate_cmd.main(["--pred-dir", str(pred_dir), "--gt-dir", str(gt_dir),
                           "--strip-suffix", "_corrected_bin"])
    _labelme(gt_dir / "x.json", 40, 30, [(5, 5, 20, 20)], label="intersection")
    rows, skipped = evaluate_cmd.evaluate_dir(pred_dir, gt_dir, label_filter=["spacing"])
    assert not rows and skipped == ["x.png: empty ground truth", "x_corrected_bin.png: no annotation"]
