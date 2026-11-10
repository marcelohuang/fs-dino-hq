"""Stage 2 episodic training: final adapter + MVPG + SAM mask decoder.

Command: ``fs-dino-hq train-stage2 --config configs/stage2/<run>.json``
(``python -m fs_dino_hq.training.train_stage2`` works as well). The recipe
follows the original Stage 2 training script:

  * model and loss: ``fs_dino_hq.modeling.stage2`` (dice+BCE on the prior
    logits and on the decoder logits), AdamW lr 1e-4, weight decay 1e-4, batch
    2, no scheduler, no gradient clipping, no augmentation; DINOv2 input is the
    518 px letterbox, 37x37 MVPG grid, V2 correlation branch; adapter, semantic
    branch, correlation branch and SAM mask decoder are trained, DINOv2 and
    the SAM prompt encoder are frozen, the SAM image encoder is never used;
  * the adapter is initialised from a distilled final adapter (Stage 1,
    ``adapter_state`` with the student's ``.layer.`` infix removed) and built
    with the student's pre-norm / no-dropout convention, recorded as
    ``adapter_norm_first`` in every checkpoint;
  * episodes come from a persistent seeded K-shot plan of a scene-disjoint
    split (``fs-dino-hq make-episodes``), read in plan order;
  * K > 1 supports are aggregated at prompt level exactly as at inference
    (``fs_dino_hq.modeling.kshot``);
  * validation on held-out scenes every epoch (fast: the first
    ``--val-episodes`` rows of the selection plan) and on every validation plan
    in full every ``--full-val-every`` epochs; the training-time primary
    checkpoint is the lowest full-validation loss (``<prefix>_best.pt``). The
    released Stage 2 checkpoint is chosen afterwards among all saved
    checkpoints by ``fs-dino-hq select-stage2``;
  * slim checkpoints (no DINOv2 copies, no SAM image encoder: about 35 MB);
    ``<prefix>_last.pt`` also carries optimizer, RNG states, trackers and
    histories so ``--resume`` continues the same run.

``--config`` takes a JSON file whose keys are argument names (``${VAR}``
references are expanded); command-line flags win. See docs/training.md.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from fs_dino_hq.config import parse_with_config
from fs_dino_hq.data.episodes import PlanEpisodeDataset, collate_plan_episodes
from fs_dino_hq.data.letterbox import LetterboxPreprocess, content_box, content_box_on_grid
from fs_dino_hq.modeling.backbones import load_dinov2_vitb14, load_sam_vit_h
from fs_dino_hq.modeling.stage2 import (DINO_PREFIXES, TRAINABLE_TOP, Stage2Model,
                                        apply_stage2_freeze, batch_iou, stage2_loss,
                                        stage2_param_audit, trainable_parameters)
from fs_dino_hq.utils.provenance import code_hashes_by_name, file_sha256
from fs_dino_hq.utils.seeding import rng_metadata, seed_everything, worker_init

TRAINER = "fs_dino_hq.training.train_stage2"
CORR_ARCH = "v2_reduced"
# parameters only: adapter 1,250,048 + semantic 787,456 + corr 2,501,141 + decoder 4,058,340
# (counting the 4,635 BatchNorm buffers as well gives 8,601,620); valid for --feat-hw 37
EXPECTED_TRAINABLE_PARAMS = 8_596_985

CONFIG_KEYS = ("synthetic_root", "train_plan", "val_plan", "split_manifest", "split_audit",
               "adapter_ckpt", "dino_ckpt", "sam_ckpt", "save_dir", "epochs",
               "episodes_per_epoch", "val_episodes", "full_val_every", "best_selection",
               "batch_size", "lr", "weight_decay", "seed", "num_workers", "print_every",
               "dino_size", "ckpt_prefix", "save_epochs", "adapter_norm_first",
               "adapter_dropout", "label_filter", "smoke_episodes", "feat_hw", "notes",
               "n_shot", "val_plans", "select_on", "save_every")
REQUIRED = ("synthetic_root", "train_plan", "val_plans", "adapter_ckpt", "dino_ckpt", "sam_ckpt", "save_dir")
# Options that define the schedule and the objective; --resume refuses a run in which one of them changed.
RESUME_LOCKED_ARGS = ("episodes_per_epoch", "n_shot", "batch_size", "lr", "weight_decay", "seed",
                      "val_episodes", "full_val_every", "best_selection", "select_on")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fs-dino-hq train-stage2",
                                description="Stage 2 episodic training (adapter + MVPG + SAM mask decoder)")
    p.add_argument("--config", default=None, help="JSON with argument names as keys; CLI wins")
    p.add_argument("--synthetic-root", help="dataset root (component subfolders with images + LabelMe JSON)")
    p.add_argument("--train-plan", help="train episode plan CSV (K supports per row)")
    p.add_argument("--n-shot", type=int, default=1, help="supports per training episode (plan must provide them)")
    p.add_argument("--val-plan", help="single validation plan (K = 1); same as --val-plans val1=<path>:1; "
                                      "on the command line it replaces the val_plans (and select_on) of --config")
    p.add_argument("--val-plans", nargs="*", default=None,
                   help="several validation plans as label=path:k (first = primary unless --select-on); "
                        "all are run at full-validation epochs, the primary also as fast-val")
    p.add_argument("--select-on", default=None, help="label of the validation plan that selects checkpoints")
    p.add_argument("--save-every", type=int, default=0, help="snapshot every N epochs (0 = only --save-epochs)")
    p.add_argument("--split-manifest", default=None, help="train split manifest CSV (hashed for provenance)")
    p.add_argument("--split-audit", default=None, help="split_audit.json (copied into the save dir and hashed)")
    p.add_argument("--adapter-ckpt", help="distilled final adapter (key adapter_state)")
    p.add_argument("--dino-ckpt", help="dinov2_vitb14_pretrain.pth")
    p.add_argument("--sam-ckpt", help="sam_vit_h_4b8939.pth")
    p.add_argument("--save-dir")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--episodes-per-epoch", type=int, default=2000,
                   help="plan rows per epoch in plan order (cycling); 0 = whole plan")
    p.add_argument("--val-episodes", type=int, default=300, help="fast-val rows per epoch (0 = all)")
    p.add_argument("--full-val-every", type=int, default=5,
                   help="whole validation plans every N epochs and at the last epoch (0 = never)")
    p.add_argument("--best-selection", choices=["fast_val", "full_val"], default="full_val")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--dino-size", type=int, default=518)
    p.add_argument("--feat-hw", type=int, default=37)
    p.add_argument("--ckpt-prefix", default="stage2")
    p.add_argument("--save-epochs", type=int, nargs="*", default=[])
    p.add_argument("--adapter-norm-first", action=argparse.BooleanOptionalAction, default=True,
                   help="build the adapter with the distillation student's pre-norm transformer "
                        "layers (default); --no-adapter-norm-first builds the post-norm module of "
                        "the original Stage 2 training script")
    p.add_argument("--adapter-dropout", type=float, default=0.0,
                   help="dropout inside the adapter attention blocks (original script: 0.1 implicit)")
    p.add_argument("--label-filter", nargs="*", default=None,
                   help="LabelMe labels to rasterise (default: all labels)")
    p.add_argument("--notes", default=None, help="free text recorded in config.json")
    p.add_argument("--resume", action="store_true", help="continue from <save-dir>/<prefix>_last.pt")
    p.add_argument("--smoke", action="store_true",
                   help="tiny epochs (--smoke-episodes training and fast-val rows), 1 epoch unless "
                        "--epochs is given on the command line, extra gradient/freeze/preprocessing "
                        "(and K-shot) checks, smoke_report.json; exits non-zero when a check fails. "
                        "Full validation still runs every epoch on every validation plan in full, "
                        "and outputs go to --save-dir (use a separate one)")
    p.add_argument("--smoke-episodes", type=int, default=4)
    return p


def command_line_dests(argv=None) -> set:
    """Option destinations given explicitly on the command line (not from --config or defaults)."""
    p = build_parser()
    for action in p._actions:
        action.default = argparse.SUPPRESS
    given, _ = p.parse_known_args(argv)
    return set(vars(given))


def parse_args(argv=None) -> argparse.Namespace:
    p = build_parser()
    args = parse_with_config(p, argv, allowed_keys=CONFIG_KEYS)
    given = command_line_dests(argv)
    if "val_plan" in given and args.val_plan:
        # an explicit --val-plan replaces the configuration's val_plans; an
        # explicit non-empty --val-plans as well is a conflict
        if "val_plans" in given and args.val_plans:
            p.error("give --val-plan or --val-plans, not both")
        args.val_plans = [f"val1={args.val_plan}:1"]
        if "select_on" not in given:
            args.select_on = None          # a configured label (e.g. val5) does not apply to val1
    elif not args.val_plans and args.val_plan:
        args.val_plans = [f"val1={args.val_plan}:1"]
    missing = [k for k in REQUIRED if not getattr(args, k)]
    if missing:
        p.error("missing required arguments (CLI or --config): " + ", ".join(missing))
    try:
        args.val_plan_specs = [parse_plan_spec(x) for x in args.val_plans]
    except argparse.ArgumentTypeError as e:
        p.error(str(e))
    labels = [x[0] for x in args.val_plan_specs]
    if len(set(labels)) != len(labels):
        p.error(f"duplicate validation plan labels: {labels}")
    args.select_on = args.select_on or labels[0]
    if args.select_on not in labels:
        p.error(f"--select-on {args.select_on!r} not among {labels}")
    args.val_plan = next(path for lab, path, k in args.val_plan_specs if lab == args.select_on)
    if args.smoke:
        args.episodes_per_epoch = args.smoke_episodes
        args.val_episodes = args.smoke_episodes
        args.full_val_every = 1
        if "epochs" not in given:          # a configured epoch count (e.g. 200) is not a smoke run
            args.epochs = min(args.epochs, 1)
    return args


def parse_plan_spec(spec: str):
    """'label=path:k' -> (label, path, k); 'path' alone -> ('val1', path, 1)."""
    if "=" not in spec:
        return "val1", spec, 1
    label, rest = spec.split("=", 1)
    path, _, k = rest.rpartition(":")
    if not path or not k.isdigit():
        raise argparse.ArgumentTypeError(f"bad validation plan entry {spec!r}; expected label=path:k")
    return label, path, int(k)


# ── backbones (module-level so tests can substitute stubs) ──────────────────

def load_backbones(dino_ckpt, sam_ckpt):
    """DINOv2 first, then the full SAM ViT-H: this order fixes how the seeded
    global RNG is consumed before the trainable modules are initialised."""
    print("[INFO] loading DINOv2 ..."); dino = load_dinov2_vitb14(dino_ckpt)
    print("[INFO] loading SAM ViT-H ..."); sam = load_sam_vit_h(sam_ckpt)
    return dino, sam


# ── adapter initialisation ──────────────────────────────────────────────────

def init_adapter_from_distilled(adapter, ckpt_path) -> Dict:
    """Strictly load a distilled final adapter (``adapter_state``; the
    student's ``.layer.`` infix removed). Refuses early-adapter checkpoints."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = next((k for k in ("adapter_state", "adapter_state_dict") if k in ck), None)
    if key is None:
        raise KeyError(f"{ckpt_path}: no adapter_state / adapter_state_dict key "
                       f"(found {list(ck.keys())}); refusing an early-adapter or unknown file")
    kind = (ck.get("config") or {}).get("adapter")
    if kind is not None and kind != "final":
        raise ValueError(f"{ckpt_path} is a {kind!r} adapter checkpoint, not a final adapter")
    sd = {k.replace(".layer.", "."): v for k, v in ck[key].items()}
    adapter.load_state_dict(sd, strict=True)
    return {"path": str(ckpt_path), "sha256": file_sha256(ckpt_path), "state_key": key,
            "remap": ".layer. -> .", "n_tensors": len(sd),
            "n_params": int(sum(v.numel() for v in sd.values() if v.dtype.is_floating_point)),
            "provenance": {k: ck.get(k) for k in ("stage", "res", "epoch_in_stage", "epoch",
                                                   "selected_by", "val_mse", "val_cosine")},
            "source_run": (ck.get("config") or {}).get("save_dir"),
            "loaded_into": {"norm_first": bool(adapter.norm_first),
                            "dropout": float(adapter.dropout_p)}}


# ── checkpoints ─────────────────────────────────────────────────────────────

def frozen_state_sha256(model: Stage2Model) -> str:
    """SHA-256 over the frozen parameters (DINOv2 once, prompt encoder); shows
    that the freeze policy held across the run."""
    h = hashlib.sha256()
    dino = model.last_extractor.model
    for name, mod in (("dino", dino), ("prompt_encoder", model.sam_full.prompt_encoder)):
        for k, v in sorted(mod.state_dict().items()):
            h.update(f"{name}.{k}".encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_capture() -> Dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def rng_restore(st: Dict) -> None:
    random.setstate(st["python"]); np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except RuntimeError as e:       # different device count: ordering only
            print(f"[WARN] cuda RNG state not restored: {e}")


def save_stage2_checkpoint(path: Path, model: Stage2Model, epoch: int, stats: Dict,
                           val: Dict, tag: str, config: Dict, optimizer=None,
                           trackers: Optional[Dict] = None, histories: Optional[Dict] = None) -> None:
    """Slim Stage 2 checkpoint (written atomically). With ``optimizer`` it is a
    resumable ``_last.pt`` that also carries RNG states, trackers and histories."""
    obj = {"format": "stage2_model", "trainer": TRAINER,
           "epoch": epoch, "stage2_model": model.stage2_state_dict(),
           "stats": stats, "val": val, "selection_tag": tag,
           "adapter_norm_first": bool(model.adapter.norm_first),
           "adapter_dropout": float(model.adapter.dropout_p),
           "corr_arch": model.corr_arch, "feat_hw": model.feat_hw,
           "dino_excluded": True, "sam_image_encoder_excluded": True,
           "config": config}
    if optimizer is not None:
        obj["optimizer_state_dict"] = optimizer.state_dict()
        obj["rng"] = rng_capture()
        obj["trackers"] = trackers
        obj["histories"] = histories
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_resume_state(model: Stage2Model, optimizer, path: Path) -> Dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck.get("format") == "stage2_model" and "optimizer_state_dict" in ck, \
        f"{path} is not a resumable stage2 last-checkpoint"
    assert bool(ck.get("adapter_norm_first")) == bool(model.adapter.norm_first), \
        "adapter convention of the checkpoint differs from this run's --adapter-norm-first"
    missing, unexpected = model.load_state_dict(ck["stage2_model"], strict=False)
    bad = [k for k in missing if not k.startswith(DINO_PREFIXES)]
    assert not bad and not unexpected, f"resume mismatch: missing={bad[:5]} unexpected={unexpected[:5]}"
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    rng_restore(ck["rng"])
    return ck


# ── smoke-only checks ───────────────────────────────────────────────────────

# Decoder tensors that receive no gradient under this recipe: single-mask
# output uses hypernetwork 0 only, and no IoU loss is applied (as in the
# original Stage 2 training script). They also stay bit-identical under AdamW:
# at lr 1e-4 and weight decay 1e-4 the decay factor 1 - lr * wd rounds to 1 in
# float32 (see docs/implementation_notes.md).
EXPECTED_NO_GRAD = {"sam_full.mask_decoder": ("output_hypernetworks_mlps.1.", "output_hypernetworks_mlps.2.",
                                              "output_hypernetworks_mlps.3.", "iou_prediction_head.")}


def gradient_path_check(model: Stage2Model) -> Dict:
    """After one backward: every trainable tensor has a finite, nonzero gradient
    except the documented EXPECTED_NO_GRAD decoder heads; each trainable module
    has gradient mass; frozen tensors have no gradient at all."""
    rep: Dict = {"modules": {}, "frozen_with_grad": [], "ok": True,
                 "expected_no_grad": {k: list(v) for k, v in EXPECTED_NO_GRAD.items()}}
    for top in TRAINABLE_TOP:
        mod = model
        for part in top.split("."):
            mod = getattr(mod, part)
        allow = EXPECTED_NO_GRAD.get(top, ())
        n, n_grad, n_finite, gnorm, unexpected = 0, 0, 0, 0.0, []
        for name, p in mod.named_parameters():
            n += 1
            has = p.grad is not None and float(p.grad.detach().float().norm()) > 0
            if p.grad is not None:
                n_grad += 1
                if torch.isfinite(p.grad).all():
                    n_finite += 1
                gnorm += float(p.grad.detach().float().norm() ** 2)
            if not has and not name.startswith(allow):
                unexpected.append(name)
        rec = {"tensors": n, "with_grad": n_grad, "finite": n_finite, "grad_norm": math.sqrt(gnorm),
               "unexpected_without_gradient": unexpected}
        rec["ok"] = (n_grad == n_finite) and rec["grad_norm"] > 0 and not unexpected
        rep["modules"][top] = rec
        rep["ok"] &= rec["ok"]
    for name, p in model.named_parameters():
        if not p.requires_grad and p.grad is not None:
            rep["frozen_with_grad"].append(name)
    rep["ok"] &= not rep["frozen_with_grad"]
    return rep


def preprocess_check(ds: PlanEpisodeDataset, prep: LetterboxPreprocess, feat_hw: int) -> Dict:
    """Letterbox geometry of plan row 0: black pad before ImageNet normalisation,
    content box per fs_dino_hq.data.letterbox, mask pad zero, masks non-empty,
    and the 37-grid support mask footprint inside the content box on that grid."""
    r = ds.plan_row(0)
    item = ds[0]
    img = Image.open(ds.root / r["query_image"]).convert("RGB")
    S = prep.dino_size
    cw, ch = content_box(img.width, img.height, S)
    x = item["query_dino"]                                   # [3,S,S]
    pad_val = ((torch.zeros(3) - torch.tensor(prep.normalize.mean)) / torch.tensor(prep.normalize.std))
    rep = {"query_image": r["query_image"], "orig_wh": [img.width, img.height],
           "content_wh": [cw, ch], "dino_size": S,
           "pad_value_expected": [round(float(v), 4) for v in pad_val]}
    checks = {}
    if cw < S:
        checks["right_pad_is_black"] = bool(torch.allclose(x[:, :, cw:], pad_val.view(3, 1, 1).expand(3, S, S - cw), atol=1e-5))
    if ch < S:
        checks["bottom_pad_is_black"] = bool(torch.allclose(x[:, ch:, :], pad_val.view(3, 1, 1).expand(3, S - ch, S), atol=1e-5))
    qm, sm = item["query_mask"], item["support_mask"][0]
    checks["query_mask_pad_zero"] = bool(float(qm[:, ch:, :].sum() + qm[:, :, cw:].sum()) == 0.0)
    checks["query_mask_nonempty"] = bool(float(qm.sum()) > 0)
    checks["support_mask_nonempty"] = bool(float(sm.sum()) > 0)
    checks["masks_binary"] = bool(set(torch.unique(qm).tolist()) <= {0.0, 1.0})
    sm_feat = torch.nn.functional.interpolate(sm[None].float(), size=(feat_hw, feat_hw), mode="nearest")[0, 0]
    s_img = Image.open(ds.root / r["support_1_image"])
    gw, gh = content_box_on_grid(s_img.width, s_img.height, S, feat_hw)
    checks["support_mask_grid_inside_content_box"] = bool(float(sm_feat[gh:, :].sum() + sm_feat[:, gw:].sum()) == 0.0)
    rep["checks"] = checks
    rep["ok"] = all(checks.values())
    return rep


def kshot_checks(model: Stage2Model, batch: Dict, device) -> Dict:
    """K > 1 smoke checks: every support contributes (leave-one-out changes the
    prompts), support ordering has no effect (eval mode), every support image
    receives gradient, the query-only image embedding ignores the supports,
    and the aggregated prompts equal fs_dino_hq.modeling.kshot's per-episode path."""
    from fs_dino_hq.modeling.kshot import aggregate_prompts, compute_support_prompts
    K = int(batch["support_dino"].shape[1])
    rep: Dict = {"n_shot": K, "checks": {}}
    was_training = model.training
    model.eval()
    b0 = {k: (v[:1] if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        ref = model(b0)
        perm = torch.randperm(K)
        out_p = model({**b0, "support_dino": b0["support_dino"][:, perm], "support_mask": b0["support_mask"][:, perm]})
        # Order invariance is exact at PROMPT level up to float summation order
        # (the mean over K runs in a different order after the permutation);
        # the SAM decoder then amplifies that ~1e-7 input difference through
        # TF32 convolutions / attention kernels on the GPU, so its logits are
        # compared relative to their magnitude and as binary masks (measured on
        # an L40S: max|dlogit| 0.019 at max|logit| ~ 20 with dense-prompt
        # differences of 1e-7; on CPU all three are below 1e-4).
        d_final = (ref["final_logits"] - out_p["final_logits"]).abs().max()
        rep["checks"]["order_invariant_final_logits"] = float(d_final)
        rep["checks"]["order_invariant_final_logits_rel"] = float(d_final / ref["final_logits"].abs().max().clamp_min(1e-6))
        m1, m2 = ref["final_logits"] > 0, out_p["final_logits"] > 0
        union = (m1 | m2).sum()
        rep["checks"]["order_invariant_final_mask_iou"] = float((m1 & m2).sum() / union) if union > 0 else 1.0  # both empty = agree
        rep["checks"]["order_invariant_semantic_prompt"] = float((ref["semantic_prompt"] - out_p["semantic_prompt"]).abs().max())
        rep["checks"]["order_invariant_dense_prompt"] = float((ref["dense_prompt"] - out_p["dense_prompt"]).abs().max())
        rep["checks"]["order_invariant_prior_logits"] = float((ref["prior_logits"] - out_p["prior_logits"]).abs().max())
        loo = []
        for k in range(K):
            keep = [i for i in range(K) if i != k]
            o = model({**b0, "support_dino": b0["support_dino"][:, keep], "support_mask": b0["support_mask"][:, keep]})
            loo.append(float((ref["dense_prompt"] - o["dense_prompt"]).abs().mean()))
        rep["checks"]["leave_one_out_dense_prompt_mean_abs_change"] = loo
        rep["checks"]["image_embedding_ignores_supports"] = float((ref["image_embedding"] - o["image_embedding"]).abs().max())
        # inference-path consistency (kshot: per-episode calls, mean over K)
        sd, sm, q = b0["support_dino"][0].to(device), b0["support_mask"][0].to(device), b0["query_dino"].to(device)
        Fq = model.last_extractor(q)
        agg = aggregate_prompts(compute_support_prompts(model, sd, sm, Fq))
        rep["checks"]["multishot_semantic_max_abs_diff"] = float((agg["semantic_prompt"] - ref["semantic_prompt"]).abs().max())
        rep["checks"]["multishot_dense_max_abs_diff"] = float((agg["dense_prompt"] - ref["dense_prompt"]).abs().max())
        rep["checks"]["multishot_prior_logits_max_abs_diff"] = float((agg["prior_logits"] - ref["prior_logits"]).abs().max())
    # per-support gradient: the loss must depend on every support's DINO
    # features (support pixels only reach the frozen extractor under no_grad)
    model.train() if was_training else model.eval()
    out = model(b0, support_feat_grad=True)
    loss, _, _ = stage2_loss(out)
    g = torch.autograd.grad(loss, out["Fs_k"])[0]            # [K,768,37,37] for B = 1
    rep["checks"]["per_support_feature_grad_norm"] = [float(g[k].norm()) for k in range(K)]
    model.zero_grad(set_to_none=True)
    c = rep["checks"]
    rep["ok"] = (c["order_invariant_semantic_prompt"] < 1e-5 and c["order_invariant_dense_prompt"] < 1e-5
                 and c["order_invariant_prior_logits"] < 1e-5
                 and c["order_invariant_final_logits_rel"] < 5e-3 and c["order_invariant_final_mask_iou"] > 0.999
                 and all(v > 0 for v in c["leave_one_out_dense_prompt_mean_abs_change"])
                 and c["image_embedding_ignores_supports"] == 0.0
                 and max(c["multishot_semantic_max_abs_diff"], c["multishot_dense_max_abs_diff"],
                         c["multishot_prior_logits_max_abs_diff"]) < 1e-3
                 and all(v > 0 for v in c["per_support_feature_grad_norm"]))
    rep["aggregation_level"] = ("prompt: mean over supports of the semantic prompt, the dense prompt and the prior "
                                "logits; one decoder pass; the query image alone forms the SAM image embedding")
    return rep


# ── post-training evaluation (used by fs_dino_hq.training.select_stage2) ────

def evaluate_checkpoint(model: Stage2Model, path: Path, loader, device, load: bool = True) -> Dict:
    """Score one saved checkpoint on a validation loader with the training loss
    and IoU. ``load=False`` reuses the weights already in ``model`` (same file,
    another plan) and only reads the metadata."""
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=not load)
    if load:
        assert bool(ck.get("adapter_norm_first", False)) == bool(model.adapter.norm_first), \
            f"{path}: adapter convention differs from the evaluation model"
        assert ck.get("corr_arch", CORR_ARCH) == getattr(model, "corr_arch", CORR_ARCH), \
            f"{path}: corr_arch {ck.get('corr_arch')} differs from the evaluation model {getattr(model, 'corr_arch', None)}"
        missing, unexpected = model.load_state_dict(ck["stage2_model"], strict=False)
        bad = [k for k in missing if not k.startswith(DINO_PREFIXES)]
        assert not bad and not unexpected, f"{path}: missing={bad[:5]} unexpected={unexpected[:5]}"
    model.eval()
    sums = {"loss": 0.0, "loss_prior": 0.0, "loss_final": 0.0, "iou_final": 0.0, "iou_prior": 0.0}
    count, t0 = 0, time.time()
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            loss, lp, lf = stage2_loss(out)
            bs = out["final_logits"].shape[0]
            sums["loss"] += loss.item() * bs; sums["loss_prior"] += lp.item() * bs; sums["loss_final"] += lf.item() * bs
            sums["iou_final"] += batch_iou(out["final_logits"], out["query_mask_final"]) * bs
            sums["iou_prior"] += batch_iou(out["prior_logits"], out["query_mask_prior"]) * bs
            count += bs
    res = {k: v / max(count, 1) for k, v in sums.items()}
    res.update({"episodes": count, "seconds": round(time.time() - t0, 1), "epoch": ck.get("epoch"),
                "selection_tag": ck.get("selection_tag"), "sha256": file_sha256(path),
                "recorded_full_val": (ck.get("val") or {}).get("full")})
    return res


# ── training ────────────────────────────────────────────────────────────────

# Package files whose hashes are recorded in config.json (code_sha256).
_CODE_FILES = ("training/train_stage2.py", "data/episode_plans.py", "modeling/stage2.py",
               "modeling/model.py", "modeling/correlation.py", "data/letterbox.py",
               "data/episodes.py", "training/losses.py", "training/checkpoints.py")


def _code_files() -> List[Path]:
    pkg = Path(__file__).resolve().parents[1]
    return [pkg / rel for rel in _CODE_FILES if (pkg / rel).is_file()]


def train_stage2(args: argparse.Namespace) -> Dict:
    """Run (or resume) a Stage 2 training run; returns the training-time
    selection summary (also written to checkpoint_selection_report.json),
    with ``smoke_ok`` (the smoke_report.json verdict) added under --smoke."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    pfx = args.ckpt_prefix
    last_path = save_dir / f"{pfx}_last.pt"
    if last_path.is_file() and not args.resume:
        sys.exit(f"ERROR: {last_path} exists; pass --resume to continue or choose a new --save-dir")
    seed_everything(args.seed, deterministic_cudnn=True)

    # RNG order: DINOv2 (hub definition), full SAM ViT-H, then the Stage 2 modules
    dino_model, sam_full = load_backbones(args.dino_ckpt, args.sam_ckpt)
    model = Stage2Model(dino_model, sam_full, feat_hw=args.feat_hw,
                        adapter_norm_first=args.adapter_norm_first,
                        adapter_dropout=args.adapter_dropout, corr_arch=CORR_ARCH).to(device)
    del sam_full        # the model keeps the prompt encoder and the mask decoder
    adapter_init = init_adapter_from_distilled(model.adapter, args.adapter_ckpt)
    print(f"[INFO] adapter initialised from {args.adapter_ckpt} "
          f"(stage {adapter_init['provenance'].get('stage')} res {adapter_init['provenance'].get('res')} "
          f"epoch_in_stage {adapter_init['provenance'].get('epoch_in_stage')}; "
          f"norm_first={model.adapter.norm_first} dropout={model.adapter.dropout_p})")
    apply_stage2_freeze(model)
    audit = stage2_param_audit(model)
    assert audit["trainable_top_modules"] == sorted(TRAINABLE_TOP), audit["trainable_top_modules"]
    if args.feat_hw == 37:
        assert audit["trainable_params"] == EXPECTED_TRAINABLE_PARAMS, \
            f"trainable parameters {audit['trainable_params']:,} != expected {EXPECTED_TRAINABLE_PARAMS:,}"
    print(f"[INFO] trainable params {audit['trainable_params']:,} in {audit['trainable_top_modules']}")
    frozen_sha_start = frozen_state_sha256(model)
    params = trainable_parameters(model)
    assert sum(p.numel() for p in params) == audit["trainable_params"]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # data ----------------------------------------------------------------
    prep = LetterboxPreprocess(dino_size=args.dino_size)
    root = Path(args.synthetic_root)
    K = int(args.n_shot)
    train_ds = PlanEpisodeDataset(Path(args.train_plan), root, K, prep, args.label_filter)
    val_sets = {lab: (PlanEpisodeDataset(Path(path), root, k, prep, args.label_filter, limit=0), k)
                for lab, path, k in args.val_plan_specs}
    full_val_ds, sel_k = val_sets[args.select_on]
    val_ds = PlanEpisodeDataset(Path(args.val_plan), root, sel_k, prep, args.label_filter, limit=args.val_episodes)
    eps = args.episodes_per_epoch if args.episodes_per_epoch > 0 else len(train_ds.rows)
    n_fast = args.val_episodes if args.val_episodes > 0 else len(val_ds.rows)
    gen = torch.Generator(); gen.manual_seed(args.seed)
    print(f"[INFO] plan: train rows={len(train_ds.rows)} (epoch slice {eps}), {K}-shot; fast_val={n_fast} "
          f"[{args.select_on}]; full_val plans: " + ", ".join(f"{lab}={len(ds.rows)}x{k}-shot" for lab, (ds, k) in val_sets.items())
          + f"; selection on {args.select_on}; letterbox {args.dino_size}")
    if args.split_audit and Path(args.split_audit).is_file():
        shutil.copyfile(args.split_audit, save_dir / "split_audit.json")

    # provenance ------------------------------------------------------------
    def sha(p):
        return file_sha256(p) if p and Path(p).is_file() else None
    config = {**{k: getattr(args, k) for k in vars(args)},
              "device": str(device), "trainer": TRAINER,
              "recipe": "original Stage 2 training script: dice+BCE(prior)+dice+BCE(final), AdamW, "
                        "no scheduler, no grad clipping, no augmentation; DINO 518 letterbox",
              "episodes": {"n_shot": K, "plan_order": "persistent seeded plan, cycling",
                           "episodes_per_epoch": eps, "fast_val_episodes": n_fast,
                           "full_val_episodes": len(full_val_ds.rows),
                           "full_val_every": args.full_val_every,
                           "best_selection": args.best_selection,
                           "val_plans": {lab: {"path": path, "n_shot": k, "episodes": len(val_sets[lab][0].rows)}
                                         for lab, path, k in args.val_plan_specs},
                           "select_on": args.select_on,
                           "kshot_aggregation": ("prompt-level mean over supports (semantic, dense, prior "
                                                 "logits), one decoder pass; = fs_dino_hq.modeling.kshot") if K > 1
                                                else "n/a (1-shot)"},
              "preprocess": "letterbox_longest_side_topleft_pad_bilinear_nearest",
              "adapter_arch": {"norm_first": bool(model.adapter.norm_first),
                               "dropout": float(model.adapter.dropout_p),
                               "original_script_convention": {"norm_first": False, "dropout": 0.1}},
              "adapter_init": adapter_init,
              "corr_arch": CORR_ARCH,
              "trainable_params": audit["trainable_params"],
              "trainable_top_modules": audit["trainable_top_modules"],
              "frozen_state_sha256_start": frozen_sha_start,
              "input_sha256": {"adapter_ckpt": adapter_init["sha256"], "dino_ckpt": sha(args.dino_ckpt),
                               "sam_ckpt": sha(args.sam_ckpt), "train_plan": sha(args.train_plan),
                               "val_plan": sha(args.val_plan),
                               "val_plans": {lab: sha(path) for lab, path, k in args.val_plan_specs},
                               "split_manifest": sha(args.split_manifest),
                               "split_audit": sha(args.split_audit)},
              "rng": rng_metadata(args.seed), "code_sha256": code_hashes_by_name(_code_files()),
              "resumed_from": []}

    trackers = {"best_fast": math.inf, "best_full": math.inf,
                "best_fast_epoch": None, "best_full_epoch": None}
    fast_hist: List[Dict] = []; full_hist: List[Dict] = []
    start_epoch = 1
    if args.resume and last_path.is_file():
        ck = load_resume_state(model, optimizer, last_path)
        prev = ck["config"]
        for k in ("adapter_ckpt", "train_plan"):
            assert prev["input_sha256"][k] == config["input_sha256"][k], f"resume: {k} changed"
        changed = [(k, prev[k], config.get(k)) for k in RESUME_LOCKED_ARGS if k in prev and prev[k] != config.get(k)]
        if changed:
            raise SystemExit("REFUSING to resume: these options differ from the interrupted run "
                             "(option, then, now): " + ", ".join(map(str, changed)))
        trackers = ck["trackers"]; fast_hist = ck["histories"]["fast"]; full_hist = ck["histories"]["full"]
        start_epoch = int(ck["epoch"]) + 1
        config["resumed_from"] = list(prev.get("resumed_from", [])) + [
            {"epoch": int(ck["epoch"]), "checkpoint": str(last_path), "time": time.strftime("%Y-%m-%d %H:%M:%S")}]
        print(f"[INFO] resumed from {last_path} at epoch {ck['epoch']} -> continuing at {start_epoch}")
    elif args.resume:
        print(f"[INFO] --resume given but {last_path} absent: starting fresh")
    (save_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (save_dir / "freeze_audit.json").write_text(json.dumps(audit, indent=2))
    (save_dir / "adapter_init_report.json").write_text(json.dumps(adapter_init, indent=2))

    smoke: Dict = {"gradient_paths": None, "preprocess": None, "kshot": None}
    if args.smoke:
        smoke["preprocess"] = preprocess_check(train_ds, prep, args.feat_hw)
        print(f"[SMOKE] preprocess checks: {smoke['preprocess']['checks']}")
        if K > 1:
            apply_stage2_freeze(model)
            smoke["kshot"] = kshot_checks(model, collate_plan_episodes([train_ds[0]]), device)
            print(f"[SMOKE] {K}-shot checks ok={smoke['kshot']['ok']}: {smoke['kshot']['checks']}")

    def run_split(ds: PlanEpisodeDataset, offset: int, n: int, train: bool, epoch: int) -> Dict:
        ds.offset, ds.limit = offset, n
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_plan_episodes,
                            worker_init_fn=functools.partial(worker_init, base_seed=args.seed + epoch),
                            generator=gen, pin_memory=device.type == "cuda")
        if train:
            apply_stage2_freeze(model)
        else:
            model.eval()
        sums = {"loss": 0.0, "loss_prior": 0.0, "loss_final": 0.0, "iou_final": 0.0, "iou_prior": 0.0}
        count, t0 = 0, time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with (torch.enable_grad() if train else torch.no_grad()):
            for step, batch in enumerate(loader, 1):
                out = model(batch)
                loss, lp, lf = stage2_loss(out)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.smoke and smoke["gradient_paths"] is None:
                        smoke["gradient_paths"] = gradient_path_check(model)
                        print(f"[SMOKE] gradient paths ok={smoke['gradient_paths']['ok']}")
                    optimizer.step()
                bs = out["final_logits"].shape[0]
                sums["loss"] += loss.item() * bs; sums["loss_prior"] += lp.item() * bs
                sums["loss_final"] += lf.item() * bs
                sums["iou_final"] += batch_iou(out["final_logits"], out["query_mask_final"]) * bs
                sums["iou_prior"] += batch_iou(out["prior_logits"], out["query_mask_prior"]) * bs
                count += bs
                if train and (step % args.print_every == 0 or step == 1):
                    print(f"[E{epoch:03d} S{step:04d}] loss={loss.item():.5f} prior={lp.item():.5f} "
                          f"final={lf.item():.5f} ({(time.time() - t0) / step:.2f}s/step)")
        res = {k: v / max(count, 1) for k, v in sums.items()}
        res["episodes"] = count; res["seconds"] = round(time.time() - t0, 1)
        res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else 0.0
        return res

    def write_hist():
        for name, hist in (("history_fast.csv", fast_hist), ("history_full.csv", full_hist)):
            if hist:
                with open(save_dir / name, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(hist[0].keys())); w.writeheader(); w.writerows(hist)

    for epoch in range(start_epoch, args.epochs + 1):
        offset = ((epoch - 1) * eps) % len(train_ds.rows)
        tr = run_split(train_ds, offset, eps, True, epoch)
        fast = run_split(val_ds, 0, n_fast, False, epoch)
        full, full_all = None, None
        if args.full_val_every and (epoch % args.full_val_every == 0 or epoch == args.epochs):
            full_all = {lab: run_split(ds, 0, len(ds.rows), False, epoch) for lab, (ds, k) in val_sets.items()}
            full = full_all[args.select_on]
            row = {"epoch": epoch, **{f"full_val_{k}": v for k, v in full.items()}}
            if len(val_sets) > 1:        # per-plan columns (the selection plan also keeps the plain names)
                for lab, r_ in full_all.items():
                    row.update({f"full_val_{lab}_{k}": v for k, v in r_.items()})
            full_hist.append(row)
        fast_hist.append({"epoch": epoch, "plan_offset": offset,
                          **{f"train_{k}": v for k, v in tr.items()},
                          **{f"fast_val_{k}": v for k, v in fast.items()}})
        print(f"[EPOCH {epoch:03d}] train loss={tr['loss']:.5f} (prior {tr['loss_prior']:.4f} final "
              f"{tr['loss_final']:.4f}) iou={tr['iou_final']:.4f} | fast_val loss={fast['loss']:.5f} "
              f"iou={fast['iou_final']:.4f}" + (f" | FULL_val loss={full['loss']:.5f} iou={full['iou_final']:.4f}"
                                                if full else "") + f" | {tr['seconds']:.0f}s train")
        stats = {k: tr[k] for k in ("loss", "loss_prior", "loss_final")}
        val = {"fast": fast, "full": full, "select_on": args.select_on, "full_all": full_all}
        write_hist()
        if fast["loss"] < trackers["best_fast"]:
            trackers.update(best_fast=fast["loss"], best_fast_epoch=epoch)
            save_stage2_checkpoint(save_dir / f"{pfx}_best_fast_val.pt", model, epoch, stats, val, "best_fast_val", config)
            print(f"[INFO] best FAST-val {fast['loss']:.5f} @ epoch {epoch}")
        if full is not None and full["loss"] < trackers["best_full"]:
            trackers.update(best_full=full["loss"], best_full_epoch=epoch)
            save_stage2_checkpoint(save_dir / f"{pfx}_best_full_val.pt", model, epoch, stats, val, "best_full_val", config)
            print(f"[INFO] best FULL-val {full['loss']:.5f} @ epoch {epoch}")
        if epoch in set(args.save_epochs) or (args.save_every and epoch % args.save_every == 0):
            save_stage2_checkpoint(save_dir / f"{pfx}_epoch_{epoch:03d}.pt", model, epoch, stats, val, f"epoch_{epoch}", config)
        # resume state, written once per epoch with the updated trackers
        save_stage2_checkpoint(last_path, model, epoch, stats, val, "last", config, optimizer,
                               trackers, {"fast": fast_hist, "full": full_hist})

    # training-time selection + frozen check --------------------------------
    frozen_sha_end = frozen_state_sha256(model)
    frozen = {"start": frozen_sha_start, "end": frozen_sha_end, "unchanged": frozen_sha_start == frozen_sha_end,
              "covers": "DINOv2 (all tensors) + SAM prompt encoder"}
    (save_dir / "frozen_check.json").write_text(json.dumps(frozen, indent=2))
    assert frozen["unchanged"], "frozen modules changed during training"
    if args.best_selection == "full_val":
        assert full_hist, "--best-selection full_val needs --full-val-every > 0"
        primary = f"{pfx}_best_full_val.pt"; p_epoch, p_loss = trackers["best_full_epoch"], trackers["best_full"]
    else:
        primary = f"{pfx}_best_fast_val.pt"; p_epoch, p_loss = trackers["best_fast_epoch"], trackers["best_fast"]
    if (save_dir / primary).is_file():
        shutil.copyfile(save_dir / primary, save_dir / f"{pfx}_best.pt")
    selection = {"best_selection_rule": args.best_selection, "primary_checkpoint": primary,
                 "primary_alias": f"{pfx}_best.pt", "primary_epoch": p_epoch, "primary_loss": p_loss,
                 "best_fast_val": {"epoch": trackers["best_fast_epoch"], "loss": trackers["best_fast"]},
                 "best_full_val": {"epoch": trackers["best_full_epoch"], "loss": trackers["best_full"],
                                   "episodes": len(full_val_ds.rows), "every": args.full_val_every},
                 "last_epoch": fast_hist[-1]["epoch"] if fast_hist else None,
                 "epochs_requested": args.epochs,
                 "checkpoints_written": sorted(p.name for p in save_dir.glob(f"{pfx}_*.pt")),
                 "select_on": args.select_on, "n_shot_train": K,
                 "val_plans": {lab: {"path": path, "n_shot": k} for lab, path, k in args.val_plan_specs},
                 "note": "validation = held-out scenes of the shared scene-disjoint split; loss = "
                         "dice+BCE(prior)+dice+BCE(final), same formula as training; selection on the "
                         f"'{args.select_on}' plan"}
    (save_dir / "checkpoint_selection_report.json").write_text(json.dumps(selection, indent=2))
    print(f"[DONE] primary={primary} (epoch {p_epoch}, loss {p_loss}); frozen unchanged={frozen['unchanged']}")
    if args.smoke:
        # gradient_paths stays None when no training step ran (e.g. --resume past --epochs): not a pass
        smoke.update({"ok": bool(smoke["gradient_paths"] is not None and smoke["gradient_paths"]["ok"]
                                 and smoke["preprocess"]["ok"] and frozen["unchanged"]
                                 and (smoke["kshot"] is None or smoke["kshot"]["ok"])),
                      "frozen_unchanged": frozen["unchanged"], "trainable_params": audit["trainable_params"],
                      "corr_arch": CORR_ARCH,
                      "expected_trainable_params": EXPECTED_TRAINABLE_PARAMS,
                      "trainable_params_match": audit["trainable_params"] == EXPECTED_TRAINABLE_PARAMS,
                      "train_peak_mem_gb": fast_hist[-1].get("train_peak_mem_gb"),
                      "seconds_per_train_step": round(fast_hist[-1]["train_seconds"] / max(1, math.ceil(eps / args.batch_size)), 3),
                      "seconds_per_val_episode": round(fast_hist[-1]["fast_val_seconds"] / max(1, n_fast), 3),
                      "epochs_run": [h["epoch"] for h in fast_hist], "selection": selection})
        (save_dir / "smoke_report.json").write_text(json.dumps(smoke, indent=2))
        print(f"[SMOKE] ok={smoke['ok']} trainable_match={smoke['trainable_params_match']}")
        return {**selection, "smoke_ok": smoke["ok"]}
    return selection


def main(argv=None) -> int:
    """Exit status 0; with --smoke, 1 when a smoke check failed (details in smoke_report.json)."""
    args = parse_args(argv)
    result = train_stage2(args)
    if args.smoke:
        ok = bool(result.get("smoke_ok"))
        print("[SMOKE] OK" if ok else f"[SMOKE] FAILED (see {Path(args.save_dir) / 'smoke_report.json'})")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
