"""HQ-branch training on a selected Stage 2 checkpoint (``fs-dino-hq train-hq``).

    fs-dino-hq train-hq --config CONFIG.json [--save-dir DIR] [--smoke]

Model: ``FSDinoHQModel`` with the Stage 2 weights (final adapter, MVPG, SAM
mask decoder) loaded strictly from the selected Stage 2 checkpoint, the HQ
modules initialised from SAM-HQ ViT-H where an equivalent exists, and the
frozen distilled early adapter from Stage 1. Only the HQ modules train
(``fs_dino_hq.training.freeze``).

Decoder (``dual_canonical``): a decoder pass without the HQ token gives the
canonical Stage 2 base logits; a second pass with the HQ token gives the HQ
residual; ``corrected = base.detach() + 1.0 * hq``. The loss is dice + BCE on
the corrected logits at 256 x 256 (``fs_dino_hq.training.losses.hq_loss``).
K supports per query are combined by prompt averaging
(``fs_dino_hq.modeling.kshot``), as at inference.

Stage 2 input: ``--stage2-run DIR`` takes ``DIR/<ckpt_prefix>_selected.pt``
(the prefix recorded in the Stage 2 run's config.json, ``stage2`` by default)
after checking the selection record (``verify_selected_stage2``). Another
checkpoint is accepted only with ``--stage2-ckpt FILE --allow-unselected-stage2``
and is recorded as such in ``stage2_input_provenance.json``. Either option on
the command line replaces the other one coming from ``--config``; both on the
command line is an error.

Schedule: the episode plans fix the episodes; epoch e trains on plan rows
[(e-1)*N, e*N) of the training plan (N = --episodes-per-epoch, wrapping
around). Validation each epoch on the first --val-episodes rows of the
validation plan ("fast val", monitoring only) and every --full-val-every
epochs and at the last epoch on the whole validation plan ("full val"). The
primary checkpoint is the one with the lowest full-validation loss.

Outputs in --save-dir:
  config.json                       options, input hashes, matched_provenance, code hashes
  stage2_input_provenance.json      how the Stage 2 checkpoint was chosen
  stage2_selection_report.json      copy of the Stage 2 selection report (--stage2-run)
  stage2_ckpt_inspect.json, stage2_loading_report.json, sam_hq_loading_report.json
  freeze_audit.json                 trainable/frozen parameters per module
  loss_history_fast.csv             epoch, train_loss, fast_val_loss, plan_offset
  loss_history_full.csv             epoch, full_val_loss, n_episodes
  <prefix>_last.pt, <prefix>_best_fast_val.pt, <prefix>_best_full_val.pt,
  <prefix>_epoch_<E>.pt (--save-epochs), <prefix>_best.pt (= best full val)
  checkpoint_selection_report.json

``--smoke`` loads everything, runs one training step on one batch, checks that
only HQ parameters changed and writes smoke_report.json.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from fs_dino_hq.config import parse_with_config
from fs_dino_hq.data.episodes import PlanEpisodeDataset, collate_plan_episodes
from fs_dino_hq.data.letterbox import LetterboxPreprocess
from fs_dino_hq.modeling.backbones import load_dinov2_vitb14, load_sam_vit_h
from fs_dino_hq.modeling.kshot import forward_kshot
from fs_dino_hq.modeling.model import FSDinoHQModel
from fs_dino_hq.training.checkpoints import (EXPLICIT_CHECKPOINT, Stage2SelectionError,
                                             inspect_stage2_checkpoint, load_early_adapter,
                                             load_sam_hq_partial, load_stage2_strict,
                                             verify_selected_stage2)
from fs_dino_hq.training.freeze import (apply_hq_freeze, build_hq_optimizer,
                                        check_frozen_unchanged, snapshot_frozen_params,
                                        verify_freeze)
from fs_dino_hq.training.losses import hq_loss
from fs_dino_hq.utils.provenance import code_hashes_by_name, file_sha256, hq_state_sha256
from fs_dino_hq.utils.seeding import rng_metadata, seed_everything, worker_init

# Fixed by the model: 518 x 518 letterboxed input -> 37 x 37 DINOv2 token grid.
DINO_SIZE = 518
FEAT_HW = 37
# Size of the earlier correlation branch that FSDinoHQModel builds (and then
# replaces) only to keep the order of random-number use; see
# fs_dino_hq.modeling.model.Correlation4DBranchV1.
V1_DS_HW = 18

DECODER_MODE = "dual_canonical"
CORRECTED_FORMULA = "canonical_stage2.detach() + alpha*hq"
RESIDUAL_SCALE_MODE = "fixed"
TRAIN_LOGITS_SOURCE = "corrected"
EARLY_FEATURE_SOURCE = "distilled_adapter"
LOSS_WEIGHTS = {"w_final": 1.0, "w_prior": 0.0, "w_hq": 0.0}   # hq_loss = 1.0 * seg(corrected)
BEST_SELECTION = "full_val"

REQUIRED = ("synthetic_root", "train_plan", "val_plan", "dino_ckpt", "sam_ckpt",
            "sam_hq_ckpt", "early_adapter_ckpt", "save_dir")

# Package files whose hashes are recorded in config.json (code_sha256).
_CODE_FILES = ("training/train_hq.py", "training/checkpoints.py", "training/freeze.py",
               "training/losses.py", "modeling/model.py", "modeling/hq_decoder.py",
               "modeling/correlation.py", "modeling/kshot.py", "modeling/distill_adapter.py",
               "data/letterbox.py", "data/episodes.py", "data/labelme.py")


def _code_files() -> List[Path]:
    pkg = Path(__file__).resolve().parents[1]
    return [pkg / rel for rel in _CODE_FILES if (pkg / rel).is_file()]


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fs-dino-hq train-hq", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="JSON file of option values (keys = option names with underscores, "
                        "${VAR} expanded); command-line values take precedence")

    g = p.add_argument_group("data")
    g.add_argument("--synthetic-root", default=None,
                   help="dataset root (class folders with images and LabelMe JSON); "
                        "plan paths are relative to it")
    g.add_argument("--train-plan", default=None, help="training episode plan CSV (fs-dino-hq make-episodes)")
    g.add_argument("--val-plan", default=None, help="validation episode plan CSV")
    g.add_argument("--split-manifest", default=None,
                   help="training split manifest CSV (hashed into config.json only)")
    g.add_argument("--split-audit", default=None,
                   help="split_audit.json of make-episodes (copied into --save-dir and hashed)")

    g = p.add_argument_group("inputs")
    g.add_argument("--stage2-run", default=None,
                   help="Stage 2 run directory (or its <ckpt_prefix>_selected.pt) with a completed "
                        "'fs-dino-hq select-stage2'; the selection record is verified. On the "
                        "command line it replaces a stage2_ckpt from --config")
    g.add_argument("--stage2-ckpt", default=None,
                   help="any other Stage 2 checkpoint; needs --allow-unselected-stage2. On the "
                        "command line it replaces a stage2_run from --config")
    g.add_argument("--allow-unselected-stage2", action="store_true",
                   help="accept --stage2-ckpt (recorded as not the post-training selection)")
    g.add_argument("--early-adapter-ckpt", default=None,
                   help="distilled early adapter (early_adapter_best.pt of fs-dino-hq distill)")
    g.add_argument("--dino-ckpt", default=None, help="dinov2_vitb14_pretrain.pth")
    g.add_argument("--dinov2-repo", default=None,
                   help="local clone of facebookresearch/dinov2 (default: torch.hub cache / GitHub)")
    g.add_argument("--sam-ckpt", default=None, help="sam_vit_h_4b8939.pth")
    g.add_argument("--sam-hq-ckpt", default=None, help="sam_hq_vit_h.pth (initialises HQ modules)")

    g = p.add_argument_group("recipe")
    g.add_argument("--n-shot", type=int, default=5, help="supports per episode")
    g.add_argument("--epochs", type=int, default=50)
    g.add_argument("--episodes-per-epoch", type=int, default=2000,
                   help="training plan rows per epoch (plan order kept, wrapping around); "
                        "0 = whole plan")
    g.add_argument("--val-episodes", type=int, default=300,
                   help="validation plan rows of the per-epoch fast validation (0 = all)")
    g.add_argument("--full-val-every", type=int, default=5,
                   help="full validation every N epochs and at the last epoch; selects the "
                        "primary checkpoint")
    g.add_argument("--batch-size", type=int, default=2)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("output")
    g.add_argument("--save-dir", default=None)
    g.add_argument("--ckpt-prefix", default="hq", help="checkpoint file name prefix")
    g.add_argument("--save-epochs", type=int, nargs="*", default=[25, 40, 50],
                   help="epochs whose checkpoints are also kept")
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--print-every", type=int, default=20)
    g.add_argument("--smoke", action="store_true",
                   help="load everything, one training step on one batch, write smoke_report.json")
    return p


def check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    missing = [f"--{k.replace('_', '-')}" for k in REQUIRED if not getattr(args, k)]
    if missing:
        parser.error(f"missing required options: {' '.join(missing)}")
    if bool(args.stage2_run) == bool(args.stage2_ckpt):
        parser.error("give exactly one of --stage2-run (selected checkpoint) and --stage2-ckpt")
    if args.stage2_ckpt and not args.allow_unselected_stage2:
        parser.error("--stage2-ckpt is not the post-training selection; pass "
                     "--allow-unselected-stage2 to use it anyway (or use --stage2-run)")
    for name in ("n_shot", "epochs", "batch_size", "full_val_every", "print_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    for name in ("episodes_per_epoch", "val_episodes", "num_workers"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")


# ── Training ─────────────────────────────────────────────────────────────────

def train_hq(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    pfx = args.ckpt_prefix
    if not args.smoke and (save_dir / f"{pfx}_best_full_val.pt").exists():
        raise SystemExit(f"REFUSING: {save_dir} already holds {pfx}_best_full_val.pt "
                         f"(choose another --save-dir)")

    # ── Stage 2 input: the verified post-training selection, or an explicitly
    #    allowed checkpoint (no random numbers are used before seeding) ──────
    if args.stage2_run:
        try:
            stage2_input = verify_selected_stage2(args.stage2_run)
        except Stage2SelectionError as e:
            raise SystemExit(f"REFUSING Stage 2 input: {e}")
        stage2_path = Path(stage2_input["stage2_ckpt"])
    else:
        stage2_path = Path(args.stage2_ckpt).resolve()
        if not stage2_path.is_file():
            raise SystemExit(f"--stage2-ckpt {stage2_path} does not exist")
        stage2_input = {"stage2_ckpt": str(stage2_path), "provenance": EXPLICIT_CHECKPOINT,
                        "file_sha256": file_sha256(stage2_path)}
        print(f"[WARN] Stage 2 input is an explicit checkpoint, not the post-training "
              f"selection: {stage2_path}")
    save_dir.mkdir(parents=True, exist_ok=True)
    stage2_input["selection_report_copied"] = False
    if "selection_report" in stage2_input:
        shutil.copyfile(stage2_input["selection_report"], save_dir / "stage2_selection_report.json")
        stage2_input["selection_report_copied"] = True
    (save_dir / "stage2_input_provenance.json").write_text(json.dumps(stage2_input, indent=2))

    # ── Construction order is part of the recipe: with --seed fixed, the
    #    randomly initialised HQ modules (dino_l3_proj, early_proj) depend on
    #    how much of the global generator DINOv2, the full SAM ViT-H (image
    #    encoder included) and the model modules consumed before them. Keep
    #    seed -> DINOv2 -> SAM ViT-H -> FSDinoHQModel -> loads, so a run
    #    reproduces the initial_hq_state_sha256 of docs/results.md. ─────────
    seed_everything(args.seed, deterministic_cudnn=True)

    # Correlation architecture and adapter convention come from the checkpoint.
    info = inspect_stage2_checkpoint(stage2_path, out_json=save_dir / "stage2_ckpt_inspect.json")
    if info["format"] != "stage2_model":
        raise SystemExit(f"{stage2_path} is not a Stage 2 checkpoint (no 'stage2_model' key)")
    adapter_arch = dict(info["adapter_arch"])
    adapter_arch.pop("recorded_in_checkpoint", None)
    corr_arch, corr_grid = info["corr_arch"]["arch"], info["corr_arch"]["grid"]
    if (corr_arch, corr_grid) != ("v2_reduced", FEAT_HW):
        raise SystemExit(f"{stage2_path}: correlation branch {corr_arch} (grid {corr_grid}) is "
                         f"not supported; this package builds v2_reduced at grid {FEAT_HW}")
    print(f"[INFO] Stage 2: corr {corr_arch} (grid {corr_grid}), adapter "
          f"norm_first={adapter_arch['norm_first']} dropout={adapter_arch['dropout']}")

    # ── Frozen backbones ─────────────────────────────────────────────────
    print("[INFO] loading DINOv2 ...")
    dino_model = load_dinov2_vitb14(args.dino_ckpt, hub_repo=args.dinov2_repo)
    print("[INFO] loading SAM ViT-H ...")
    sam_full = load_sam_vit_h(args.sam_ckpt)
    model = FSDinoHQModel(dino_model=dino_model, sam_full=sam_full,
                          corr_arch=corr_arch, feat_hw=FEAT_HW, ds_hw=V1_DS_HW,
                          dino_ch=768, transformer_dim=256,
                          adapter_norm_first=adapter_arch["norm_first"],
                          adapter_dropout=adapter_arch["dropout"]).to(device)
    del sam_full        # the model keeps the prompt encoder and the decoder transformer

    # ── Weights: Stage 2 (strict), SAM-HQ (HQ modules), early adapter ────
    load_stage2_strict(model, stage2_path, report_path=save_dir / "stage2_loading_report.json",
                       dino_model=dino_model)
    load_sam_hq_partial(model, args.sam_hq_ckpt, save_dir=save_dir)
    ea, ea_meta = load_early_adapter(args.early_adapter_ckpt, device)
    model.early_adapter = ea
    early_meta = {"early_adapter_ckpt": args.early_adapter_ckpt,
                  "early_adapter_epoch": ea_meta["early_adapter_epoch"],
                  "early_adapter_val_cosine": ea_meta["early_adapter_val_cosine"]}

    # ── Freeze policy: only the HQ modules train ─────────────────────────
    # Note: dino_l3_proj (820,000 parameters) is built, saved and counted as
    # trainable, but it receives no gradient: with the distilled early feature
    # the early path is early_proj(early_adapter(Fq_l3)) and dino_l3_proj is not
    # called, so AdamW leaves it at its random initialisation. It is kept so
    # that checkpoints keep their keys and the random initialisation of the
    # modules built after it (and hence initial_hq_state_sha256) is unchanged.
    apply_hq_freeze(model)
    audit = verify_freeze(model)        # raises unless exactly the HQ modules are trainable
    optimizer = build_hq_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    # ── Data ─────────────────────────────────────────────────────────────
    prep = LetterboxPreprocess(dino_size=DINO_SIZE)
    root = Path(args.synthetic_root)
    train_ds = PlanEpisodeDataset(Path(args.train_plan), root, args.n_shot, prep)
    val_ds = PlanEpisodeDataset(Path(args.val_plan), root, args.n_shot, prep,
                                limit=args.val_episodes)
    full_val_ds = PlanEpisodeDataset(Path(args.val_plan), root, args.n_shot, prep, limit=0)
    eps = args.episodes_per_epoch if args.episodes_per_epoch > 0 else len(train_ds)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    print(f"[INFO] plan episodes: train={len(train_ds.rows)} (epoch slice {eps}), "
          f"fast_val={len(val_ds)}, full_val={len(full_val_ds.rows)}  n_shot={args.n_shot}")

    if args.split_audit and Path(args.split_audit).is_file():
        shutil.copyfile(args.split_audit, save_dir / "split_audit.json")

    (save_dir / "freeze_audit.json").write_text(json.dumps(audit, indent=2))

    # ── Config and provenance ────────────────────────────────────────────
    def sha_if_file(pth):
        return file_sha256(Path(pth)) if pth and Path(pth).is_file() else None

    manifest_sha = {Path(pth).name: file_sha256(Path(pth))
                    for pth in (args.train_plan, args.val_plan, args.split_manifest, args.split_audit)
                    if pth and Path(pth).is_file()}
    initial_hq_sha = hq_state_sha256(model.hq_mask_decoder)
    # Inputs and settings that must agree between two runs that are compared.
    matched_provenance = {
        "train_plan_sha256": sha_if_file(args.train_plan),
        "val_plan_sha256": sha_if_file(args.val_plan),
        "split_manifest_sha256": sha_if_file(args.split_manifest),
        "split_audit_sha256": sha_if_file(args.split_audit),
        "stage2_ckpt_sha256": stage2_input["file_sha256"],
        "adapter_norm_first": adapter_arch["norm_first"],
        "dino_ckpt_sha256": file_sha256(Path(args.dino_ckpt)),
        "sam_ckpt_sha256": file_sha256(Path(args.sam_ckpt)),
        "sam_hq_ckpt_sha256": file_sha256(Path(args.sam_hq_ckpt)),
        "early_adapter_ckpt_sha256": sha_if_file(args.early_adapter_ckpt),
        "initial_hq_state_sha256": initial_hq_sha,
        "seed": args.seed, "n_shot": args.n_shot, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "episodes_per_epoch": args.episodes_per_epoch, "val_episodes": args.val_episodes,
        "full_val_every": args.full_val_every, "best_selection": BEST_SELECTION,
        "early_feature_source": EARLY_FEATURE_SOURCE,
        "train_logits_source": TRAIN_LOGITS_SOURCE,
        **LOSS_WEIGHTS,
        "trainable_params": audit["trainable_params"],
        "trainable_top_modules": audit["trainable_top_modules"],
    }
    print(f"[INFO] initial HQ-state sha256 = {initial_hq_sha[:16]}...  "
          f"trainable = {audit['trainable_params']:,}")
    config = {**vars(args),
              "stage2_ckpt": str(stage2_path),
              "dino_size": DINO_SIZE, "feat_hw": FEAT_HW,
              "corr_arch": corr_arch, "corr_grid": corr_grid, "adapter_arch": adapter_arch,
              "decoder_mode": DECODER_MODE, "residual_scale_mode": RESIDUAL_SCALE_MODE,
              "train_logits_source": TRAIN_LOGITS_SOURCE,
              "early_feature_source": EARLY_FEATURE_SOURCE, **LOSS_WEIGHTS,
              "corrected_formula": CORRECTED_FORMULA,
              "preprocess": "letterbox_longest_side_topleft_pad_bilinear_nearest",
              "kshot_aggregation": "prompt_mean(semantic,dense);prior_mask_mean",
              "best_selection": BEST_SELECTION,
              "checkpoint_selection": (f"min {BEST_SELECTION} loss "
                                       f"(full_val every {args.full_val_every} epochs)"),
              "rng": rng_metadata(args.seed), **early_meta,
              "stage2_ckpt_sha256": stage2_input["file_sha256"],
              "stage2_input": stage2_input,
              "manifest_sha256": manifest_sha,
              "matched_provenance": matched_provenance,
              "residual_scale_value": 1.0,
              "residual_scale_initial_value": 1.0,
              "code_sha256": code_hashes_by_name(_code_files())}
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))

    def run_split(ds: PlanEpisodeDataset, offset: int, n_episodes: int,
                  train: bool, epoch: int) -> float:
        """Mean per-episode loss over plan rows [offset, offset + n_episodes)."""
        ds.offset = offset
        ds.limit = n_episodes
        loader = DataLoader(ds, batch_size=args.batch_size,
                            shuffle=False,  # plan order is the schedule
                            num_workers=args.num_workers, collate_fn=collate_plan_episodes,
                            worker_init_fn=functools.partial(worker_init,
                                                             base_seed=args.seed + epoch),
                            generator=gen)
        model.train() if train else model.eval()
        if train:
            apply_hq_freeze(model)  # re-freeze after .train()
        total, count = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for step, batch in enumerate(loader, 1):
                out = forward_kshot(model, batch, device)
                loss = hq_loss(out)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                bs = batch["query_dino"].shape[0]
                total += loss.item() * bs
                count += bs
                if train and (step % args.print_every == 0 or step == 1):
                    print(f"[E{epoch:03d} S{step:04d}] loss={loss.item():.5f} "
                          f"cls={batch['component']}")
        return total / max(count, 1)

    if args.smoke:
        return _smoke(args, model, optimizer, train_ds, device, save_dir, corr_arch)

    # ── Training ─────────────────────────────────────────────────────────
    # Checkpoints exclude the frozen DINOv2 extractors (reloaded from the
    # DINOv2 checkpoint at inference).
    def slim_state():
        return {k: v for k, v in model.state_dict().items()
                if not k.startswith(("last_extractor.", "layer3_extractor."))}

    def save_ckpt(path: Path, epoch, tr, fast, full, tag):
        # model_state: everything inference needs except DINOv2 (Stage 2
        # weights, HQ modules, early adapter). hq_state: the HQ modules alone.
        hq_state = {f"hq_mask_decoder.{k}": v.cpu() for k, v in
                    model.hq_mask_decoder.hq_state_dict().items()}
        torch.save({"epoch": epoch, "model_state": slim_state(),
                    "hq_state": hq_state,
                    "hq_module_names": sorted(model.hq_mask_decoder.HQ_MODULE_NAMES),
                    "decoder_mode": DECODER_MODE,
                    "corrected_formula": CORRECTED_FORMULA,
                    "train_logits_source": TRAIN_LOGITS_SOURCE,
                    "residual_scale_mode": RESIDUAL_SCALE_MODE,
                    "residual_scale_value": float(model.hq_mask_decoder.residual_scale),
                    "residual_scale_initial_value": 1.0,
                    "n_shot": args.n_shot,
                    "early_feature_source": EARLY_FEATURE_SOURCE,
                    "adapter_norm_first": adapter_arch["norm_first"],
                    "adapter_dropout": adapter_arch["dropout"],
                    "hq_arch_meta": {"corr_arch": corr_arch,
                                     "corr_grid": corr_grid,
                                     "adapter_norm_first": adapter_arch["norm_first"],
                                     "early_feature_source": EARLY_FEATURE_SOURCE,
                                     "transformer_dim": 256,
                                     "decoder_mode": DECODER_MODE,
                                     "note": ("dual_canonical decoder: the base is a "
                                              "SEPARATE hf-free pass, so corrected = "
                                              "canonical_stage2.detach() + alpha*hq "
                                              "and hf_token cannot perturb the base")},
                    "dino_excluded": True, "train_loss": tr,
                    "fast_val_loss": fast, "full_val_loss": full,
                    "selection_tag": tag, "config": config}, path)

    full_every = args.full_val_every
    save_epochs = set(args.save_epochs)
    best_fast, best_full = math.inf, math.inf
    best_fast_epoch, best_full_epoch = None, None
    fast_hist: List[Dict] = []
    full_hist: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        offset = ((epoch - 1) * eps) % len(train_ds.rows)
        tr = run_split(train_ds, offset, eps, train=True, epoch=epoch)
        fast = run_split(val_ds, 0, args.val_episodes or len(val_ds.rows),
                         train=False, epoch=epoch)
        fast_hist.append({"epoch": epoch, "train_loss": tr, "fast_val_loss": fast,
                          "plan_offset": offset})

        full = None
        if epoch % full_every == 0 or epoch == args.epochs:
            full = run_split(full_val_ds, 0, len(full_val_ds.rows), train=False, epoch=epoch)
            full_hist.append({"epoch": epoch, "full_val_loss": full,
                              "n_episodes": len(full_val_ds.rows)})
            print(f"[EPOCH {epoch:03d}] train={tr:.5f}  fast_val={fast:.5f}  FULL_val={full:.5f}")
        else:
            print(f"[EPOCH {epoch:03d}] train={tr:.5f}  fast_val={fast:.5f}")

        save_ckpt(save_dir / f"{pfx}_last.pt", epoch, tr, fast, full, "last")
        if fast < best_fast:
            best_fast, best_fast_epoch = fast, epoch
            save_ckpt(save_dir / f"{pfx}_best_fast_val.pt", epoch, tr, fast, full, "best_fast_val")
            print(f"[INFO] best FAST-val  {best_fast:.5f} @ epoch {epoch}")
        if full is not None and full < best_full:
            best_full, best_full_epoch = full, epoch
            save_ckpt(save_dir / f"{pfx}_best_full_val.pt", epoch, tr, fast, full, "best_full_val")
            print(f"[INFO] best FULL-val  {best_full:.5f} @ epoch {epoch}")
        if epoch in save_epochs:
            save_ckpt(save_dir / f"{pfx}_epoch_{epoch}.pt", epoch, tr, fast, full, f"epoch_{epoch}")

        _write_csv(save_dir / "loss_history_fast.csv", fast_hist)
        if full_hist:
            _write_csv(save_dir / "loss_history_full.csv", full_hist)

    # ── Primary checkpoint: lowest full-validation loss ──────────────────
    primary_src = f"{pfx}_best_full_val.pt"
    if (save_dir / primary_src).is_file():
        shutil.copyfile(save_dir / primary_src, save_dir / f"{pfx}_best.pt")
    selection = {
        "best_selection_rule": BEST_SELECTION,
        "primary_checkpoint": primary_src,
        "primary_alias": f"{pfx}_best.pt",
        "primary_epoch": best_full_epoch, "primary_loss": best_full,
        "best_fast_val": {"epoch": best_fast_epoch, "loss": best_fast},
        "best_full_val": {"epoch": best_full_epoch, "loss": best_full,
                          "full_val_episodes": len(full_val_ds.rows),
                          "full_val_every": full_every},
        "checkpoints_written": sorted(p.name for p in save_dir.glob(f"{pfx}_*.pt")),
        "note": (f"full-val is over the ENTIRE val plan; fast-val is the fixed first "
                 f"{len(val_ds)} episodes of it and is monitoring-only when "
                 f"best_selection=full_val."),
    }
    (save_dir / "checkpoint_selection_report.json").write_text(json.dumps(selection, indent=2))
    print(f"[DONE] primary={primary_src} (epoch {best_full_epoch}, loss {best_full:.5f}); "
          f"best_fast={best_fast:.5f}@{best_fast_epoch}")
    return 0


def _write_csv(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _smoke(args, model, optimizer, train_ds, device, save_dir: Path, corr_arch: str) -> int:
    """One training step on the first plan rows; checks that only HQ tensors change."""
    print("[SMOKE] one mini-batch forward/backward ...")
    t0 = time.time()
    batch = collate_plan_episodes([train_ds[i] for i in range(min(args.batch_size, len(train_ds)))])
    model.train()
    apply_hq_freeze(model)
    frozen_snap = snapshot_frozen_params(model)
    hq_before = {k: v.detach().cpu().clone() for k, v in model.hq_mask_decoder.hq_state_dict().items()}
    out = forward_kshot(model, batch, device)
    loss = hq_loss(out)
    optimizer.zero_grad()
    loss.backward()
    grad_modules = sorted({".".join(n.split(".")[:2]) for n, p in model.named_parameters()
                           if p.grad is not None and float(p.grad.abs().sum()) > 0})
    grads_finite = all(bool(torch.isfinite(p.grad).all())
                       for p in model.parameters() if p.grad is not None)
    optimizer.step()
    frozen_ok, frozen_changed = check_frozen_unchanged(model, frozen_snap)
    hq_after = model.hq_mask_decoder.hq_state_dict()
    hq_changed = sorted({k.split(".")[0] for k in hq_before
                         if not torch.equal(hq_before[k], hq_after[k].detach().cpu())})
    ok = (math.isfinite(float(loss.item())) and grads_finite and frozen_ok and bool(hq_changed)
          and all(m.startswith("hq_mask_decoder.") for m in grad_modules))
    report = {
        "ok": ok, "loss": float(loss.item()),
        "shapes": {k: list(out[k].shape) for k in (
            "adapter_embedding", "prior_logits", "canonical_base_logits", "hq_logits",
            "corrected_logits", "final_logits", "query_mask_final")},
        "decoder_mode": DECODER_MODE, "residual_scale_mode": RESIDUAL_SCALE_MODE,
        "n_shot": args.n_shot, "corr_arch": corr_arch, "device": str(device),
        "modules_with_gradient": grad_modules, "grads_finite": grads_finite,
        "hq_modules_changed": hq_changed,
        "frozen_unchanged_after_step": frozen_ok, "frozen_changed": frozen_changed[:5],
        "seconds": round(time.time() - t0, 2),
    }
    (save_dir / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("[SMOKE] OK" if ok else "[SMOKE] FAILED")
    return 0 if ok else 1


def command_line_dests(argv=None) -> set:
    """Option destinations given explicitly on the command line (not from --config or defaults)."""
    p = build_parser()
    for action in p._actions:
        action.default = argparse.SUPPRESS
    given, _ = p.parse_known_args(argv)
    return set(vars(given))


def parse_args(argv=None) -> argparse.Namespace:
    """Options from --config and the command line, checked.

    The two Stage 2 inputs exclude each other, so one given on the command line
    replaces the other one taken from the configuration (e.g. --stage2-ckpt
    with a configuration that sets stage2_run); both on the command line is
    an error (check_args).
    """
    parser = build_parser()
    args = parse_with_config(parser, argv)
    given = command_line_dests(argv)
    if "stage2_ckpt" in given and "stage2_run" not in given:
        args.stage2_run = None
    elif "stage2_run" in given and "stage2_ckpt" not in given:
        args.stage2_ckpt = None
    check_args(parser, args)
    return args


def main(argv=None) -> int:
    return train_hq(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
