"""Checkpoint loading for HQ-branch training.

inspect_stage2_checkpoint   structural summary of a Stage 2 checkpoint (adapter
                            convention, correlation architecture, tensor counts)
load_stage2_strict          strict, audited load of a Stage 2 checkpoint into
                            ``FSDinoHQModel`` (raises on any contract violation)
verify_selected_stage2      the Stage 2 selection gate: only the checkpoint
                            published by ``fs-dino-hq select-stage2`` is accepted
load_sam_hq_partial         SAM-HQ decoder weights for the HQ modules
load_early_adapter          frozen distilled early adapter (Stage 1)

Stage 2 checkpoints hold ``stage2_model``, the state dict of
``fs_dino_hq.modeling.stage2.Stage2Model`` without the DINOv2 copies:
``adapter.*``, ``semantic_branch.*``, ``pcs_branch.*``, ``corr_branch.*``,
``sam_full.prompt_encoder.*``, ``sam_full.mask_decoder.*``.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..modeling.correlation import detect_corr_arch
from ..modeling.distill_adapter import BottleneckAdapter as DistillationAdapter
from ..utils.provenance import file_sha256

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 checkpoints
# ─────────────────────────────────────────────────────────────────────────────

IGNORED_PREFIXES = (
    "last_extractor.",            # DINOv2: loaded from its own checkpoint
    "layer3_extractor.",
    "sam_full.image_encoder.",    # never used
    "sam_full.prompt_encoder.",   # loaded from the SAM checkpoint (checked for drift below)
)
COMPONENT_PREFIXES = {
    "adapter": "adapter.",
    "semantic_branch": "semantic_branch.",
    "pcs_branch": "pcs_branch.",
    "corr_branch": "corr_branch.",
}
DECODER_PREFIX = "sam_full.mask_decoder."


def _load_checkpoint(path: Path, mmap: bool = False) -> Dict:
    """``torch.load`` on CPU, tensors-only unpickling when the file allows it.

    Checkpoints that carry non-tensor Python objects (a Stage 2 ``_last.pt``
    stores NumPy random-generator states) need full unpickling, which can run
    arbitrary code: only load checkpoints you trust.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    except pickle.UnpicklingError:
        print(f"[checkpoints] {path}: not loadable with weights_only=True; "
              f"falling back to full unpickling (trusted files only)")
        return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)


def _tensor_count(sd: Dict) -> int:
    return sum(1 for v in sd.values() if torch.is_tensor(v))


def _strip(sd: Dict, prefix: str) -> Dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def inspect_stage2_checkpoint(ckpt_path, out_json=None) -> Dict:
    """Structural inspection (memory-mapped, cheap); optionally written to ``out_json``.

    ``adapter_arch`` is the final-adapter transformer convention the HQ model
    must be built with. A checkpoint without the ``adapter_norm_first`` key was
    trained post-norm with dropout 0.1 (PyTorch's default); checkpoints from
    ``fs-dino-hq train-stage2`` record ``norm_first=True``, dropout 0.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = _load_checkpoint(ckpt_path, mmap=True)
    info: Dict = {
        "checkpoint_path": str(ckpt_path),
        "top_level_keys": list(ckpt.keys()),
        "epoch": ckpt.get("epoch"),
        "stats": {k: float(v) for k, v in ckpt.get("stats", {}).items()}
                 if isinstance(ckpt.get("stats"), dict) else None,
    }
    info["adapter_arch"] = {"norm_first": bool(ckpt.get("adapter_norm_first", False)),
                            "dropout": float(ckpt.get("adapter_dropout", 0.1)),
                            "recorded_in_checkpoint": "adapter_norm_first" in ckpt}
    info["trainer"] = ckpt.get("trainer", "unknown (no trainer key)")
    info["corr_arch_recorded"] = ckpt.get("corr_arch")
    info["selection_tag"] = ckpt.get("selection_tag")
    sd = ckpt.get("stage2_model")
    if sd is None:
        info["format"] = "unknown"
        info["component_keys"] = {k: _tensor_count(v) for k, v in ckpt.items()
                                  if isinstance(v, dict)}
    else:
        info["format"] = "stage2_model"
        prefixes: Dict[str, Dict] = {}
        for k, v in sd.items():
            p = k.split(".")[0]
            if k.startswith("sam_full."):
                p = "sam_full." + k.split(".")[1]
            d = prefixes.setdefault(p, {"tensors": 0, "params": 0})
            d["tensors"] += 1
            d["params"] += int(v.numel())
        info["prefixes"] = prefixes
        info["total_tensors"] = len(sd)
        arch, grid = detect_corr_arch(sd)
        info["corr_arch"] = {"arch": arch, "grid": grid}
        for sentinel in ("corr_branch.depthwise.weight",
                         "corr_branch.pointwise.weight",
                         "adapter.proj_out.weight",
                         "sam_full.mask_decoder.iou_token.weight"):
            if sentinel in sd:
                info.setdefault("sentinel_shapes", {})[sentinel] = list(sd[sentinel].shape)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(info, indent=2))
    return info


def load_stage2_strict(model: nn.Module, ckpt_path, report_path,
                       dino_ckpt_sha256: Optional[str] = None) -> Dict:
    """Load a Stage 2 checkpoint into ``FSDinoHQModel`` under a strict contract.

    loaded    ``adapter.*``, ``semantic_branch.*``, ``pcs_branch.*``,
              ``corr_branch.*`` (each strict) and ``sam_full.mask_decoder.*``
              -> the Stage 2 (non-HQ) weights of ``model.hq_mask_decoder``
              (every key must land, shapes exact)
    ignored   ``IGNORED_PREFIXES`` (modules loaded from their own checkpoints)
    verified  the ignored SAM prompt encoder must equal ``model.prompt_encoder``;
              the DINOv2 weights file must be the one Stage 2 was trained with
              (``dino_ckpt_sha256`` against the hash recorded in the checkpoint)
    also      the adapter convention (``adapter_norm_first``) and the
              correlation architecture must match the model

    Writes the JSON report to ``report_path`` and then raises RuntimeError if
    any check failed, so a mismatched checkpoint cannot train silently.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = _load_checkpoint(ckpt_path, mmap=True)
    if "stage2_model" not in ckpt:
        raise ValueError(f"{ckpt_path} has no 'stage2_model' key "
                         f"(top-level: {list(ckpt.keys())})")
    full_sd: Dict[str, torch.Tensor] = ckpt["stage2_model"]

    report: Dict = {
        "checkpoint_path": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "stats": {k: float(v) for k, v in ckpt.get("stats", {}).items()}
                 if isinstance(ckpt.get("stats"), dict) else None,
        "loaded": {}, "ignored": {}, "remapped": {},
        "missing": {}, "mismatched": [], "unexpected_prefixes": [],
        "consistency_checks": {},
    }
    errors: List[str] = []

    # The adapter's transformer convention must match the checkpoint's (the
    # parameter names are the same either way, so only this flag tells).
    ck_norm_first = bool(ckpt.get("adapter_norm_first", False))
    model_norm_first = bool(getattr(model.adapter, "norm_first", False))
    report["adapter_arch"] = {"checkpoint_norm_first": ck_norm_first,
                              "model_norm_first": model_norm_first,
                              "recorded_in_checkpoint": "adapter_norm_first" in ckpt}
    if ck_norm_first != model_norm_first:
        errors.append(f"adapter convention mismatch: checkpoint norm_first={ck_norm_first} "
                      f"but model.adapter.norm_first={model_norm_first} (build the model "
                      f"with adapter_norm_first from inspect_stage2_checkpoint)")

    # The correlation architecture recorded in the checkpoint (if any) and the
    # one detected from its tensors must both match the model's branch.
    det_arch, det_grid = detect_corr_arch(full_sd)
    model_arch = getattr(model, "corr_arch", None)
    if model_arch is None:
        model_arch = detect_corr_arch(model.corr_branch.state_dict(), prefix="")[0]
    rec_arch = ckpt.get("corr_arch")
    report["corr_arch"] = {"detected": det_arch, "grid": det_grid, "recorded": rec_arch,
                           "model": model_arch}
    if det_arch != model_arch:
        errors.append(f"correlation architecture mismatch: checkpoint {det_arch}, model {model_arch}")
    if rec_arch is not None and rec_arch != det_arch:
        errors.append(f"checkpoint metadata corr_arch={rec_arch} but tensors are {det_arch}")

    # Every checkpoint key must be claimed by a component, the decoder remap
    # or the ignore list.
    claimed = set()
    for k in full_sd:
        if k.startswith(tuple(COMPONENT_PREFIXES.values())) or k.startswith(DECODER_PREFIX):
            claimed.add(k)
        elif k.startswith(IGNORED_PREFIXES):
            p = next(p for p in IGNORED_PREFIXES if k.startswith(p))
            report["ignored"][p] = report["ignored"].get(p, 0) + 1
            claimed.add(k)
    unclaimed = sorted(set(full_sd) - claimed)
    if unclaimed:
        report["unexpected_prefixes"] = unclaimed[:50]
        errors.append(f"{len(unclaimed)} checkpoint keys outside the audited "
                      f"contract, e.g. {unclaimed[:5]}")

    # Strict per-component loads.
    for name, prefix in COMPONENT_PREFIXES.items():
        sub = _strip(full_sd, prefix)
        module = getattr(model, name)
        own = module.state_dict()
        mismatched = [f"{k}: ckpt{tuple(sub[k].shape)} vs model{tuple(own[k].shape)}"
                      for k in sub if k in own and own[k].shape != sub[k].shape]
        missing = sorted(set(own) - set(sub))
        unexpected = sorted(set(sub) - set(own))
        if mismatched or missing or unexpected:
            report["mismatched"] += [f"{name}: {m}" for m in mismatched]
            report["missing"][name] = missing
            if unexpected:
                report["unexpected_prefixes"] += [prefix + k for k in unexpected[:50]]
            errors.append(f"component '{name}': missing={len(missing)} "
                          f"unexpected={len(unexpected)} shape={len(mismatched)}")
        else:
            module.load_state_dict(sub, strict=True)
            report["loaded"][name] = len(sub)

    # Decoder: sam_full.mask_decoder.* -> the Stage 2 keys of hq_mask_decoder.
    dec_sd = _strip(full_sd, DECODER_PREFIX)
    hq_names = tuple(model.hq_mask_decoder.HQ_MODULE_NAMES)
    own_dec = model.hq_mask_decoder.state_dict()
    base_keys = {k for k in own_dec if not k.startswith(hq_names)}
    missing_dec = sorted(base_keys - set(dec_sd))
    unexpected_dec = sorted(set(dec_sd) - base_keys)
    mismatched_dec = [k for k in dec_sd if k in own_dec
                      and own_dec[k].shape != dec_sd[k].shape]
    if missing_dec or unexpected_dec or mismatched_dec:
        errors.append(f"decoder base: missing={len(missing_dec)} "
                      f"unexpected={len(unexpected_dec)} shape={len(mismatched_dec)}")
        report["missing"]["hq_mask_decoder_base"] = missing_dec
        report["mismatched"] += [f"decoder: {k}" for k in mismatched_dec]
        if unexpected_dec:
            report["unexpected_prefixes"] += [DECODER_PREFIX + k for k in unexpected_dec[:50]]
    else:
        model.hq_mask_decoder.load_state_dict(dec_sd, strict=False)
        report["loaded"]["hq_mask_decoder_base"] = len(dec_sd)
        report["remapped"][DECODER_PREFIX + "*"] = "hq_mask_decoder.* (base keys)"
        report["missing"]["hq_mask_decoder_base"] = \
            f"HQ-only modules keep their SAM-HQ/random init ({len(own_dec) - len(base_keys)} keys)"

    # Frozen components that are not loaded from this checkpoint must equal
    # the ones the model already holds.
    pe_sd = _strip(full_sd, "sam_full.prompt_encoder.")
    own_pe = model.prompt_encoder.state_dict()
    pe_diff = [k for k in pe_sd if k in own_pe
               and not torch.allclose(pe_sd[k].float(), own_pe[k].cpu().float(), atol=1e-6)]
    report["consistency_checks"]["prompt_encoder_equal"] = (len(pe_diff) == 0)
    if pe_diff:
        errors.append(f"prompt_encoder drifted in stage-2 ckpt: {pe_diff[:3]}")

    # Stage 2 checkpoints do not store DINOv2; they record the hash of the
    # weights file they were trained with.
    recorded = (ckpt.get("config") or {}).get("input_sha256", {}).get("dino_ckpt")
    if dino_ckpt_sha256 is not None and recorded is not None:
        report["consistency_checks"]["dino_ckpt_sha256_equal"] = (recorded == dino_ckpt_sha256)
        if recorded != dino_ckpt_sha256:
            errors.append(f"DINOv2 weights differ from those of the Stage 2 run "
                          f"(sha256 {dino_ckpt_sha256[:16]}... vs recorded {recorded[:16]}...)")
    else:
        report["consistency_checks"]["dino_ckpt_sha256_equal"] = None      # not recorded or not given

    report["status"] = "OK" if not errors else "FAILED"
    report["errors"] = errors
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2))
    print(f"[stage2] strict load {report['status']}: "
          + ", ".join(f"{n} {c}" for n, c in report["loaded"].items())
          + f"  (report: {report_path})")
    if errors:
        raise RuntimeError("stage-2 strict load FAILED:\n  " + "\n  ".join(errors))
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 selection gate
# ─────────────────────────────────────────────────────────────────────────────

SELECTED_SUFFIX = "_selected.pt"
POST_TRAINING_SELECTION = "post-training selection (stage2_selection_report.json)"
EXPLICIT_CHECKPOINT = "explicit checkpoint (not the post-training selection)"


class Stage2SelectionError(RuntimeError):
    """The Stage 2 input is not a verified post-training selection."""


def run_ckpt_prefix(run: Path, default: str = "stage2") -> str:
    """Checkpoint prefix of a Stage 2 run (``ckpt_prefix`` in its config.json), as select-stage2 uses it."""
    try:
        cfg = json.loads((Path(run) / "config.json").read_text())
    except (OSError, ValueError):
        return default
    return (cfg.get("ckpt_prefix") if isinstance(cfg, dict) else None) or default


def _selection_files(path: Path, prefix: Optional[str]) -> Tuple[Path, Path, Path, Path]:
    """(run_dir, report, sidecar, artifact) for a run directory or its artifact.

    For a directory, ``prefix=None`` means the run's own ``ckpt_prefix``; for a
    ``<prefix>_selected.pt`` file the prefix comes from the file name.
    """
    if path.is_dir():
        run = path
        if prefix is None:
            prefix = run_ckpt_prefix(run)
    elif path.name.endswith(SELECTED_SUFFIX):
        run, prefix = path.parent, path.name[:-len(SELECTED_SUFFIX)]
    elif not path.exists():
        raise Stage2SelectionError(f"{path} does not exist")
    else:
        raise Stage2SelectionError(
            f"{path} is neither a Stage 2 run directory nor a '<prefix>{SELECTED_SUFFIX}' "
            f"artifact of 'fs-dino-hq select-stage2'")
    return (run, run / f"{prefix}_selection_report.json", run / f"{prefix}_selected.json",
            run / f"{prefix}{SELECTED_SUFFIX}")


def verify_selected_stage2(path, prefix: Optional[str] = None) -> Dict:
    """Verify that a Stage 2 checkpoint is the published post-training selection.

    ``path`` is a Stage 2 run directory or its ``<prefix>_selected.pt``. For a
    directory the prefix is ``prefix`` when given, else the run's
    ``ckpt_prefix`` from its config.json (``stage2`` when absent), the prefix
    under which ``fs-dino-hq select-stage2`` writes. Requires

      * ``<prefix>_selection_report.json`` with ``status == "OK"`` (a failed
        selection rewrites the report but leaves an earlier artifact in place),
      * the sidecar ``<prefix>_selected.json`` with ``state_sha256`` equal to the
        report's ``winner.state_sha256``,
      * the SHA-256 of the ``<prefix>_selected.pt`` bytes equal to the sidecar's
        ``file_sha256`` (the file has not changed since selection).

    The model-state hash itself is checked by the selection step when it
    publishes the artifact. Returns a provenance record; raises
    ``Stage2SelectionError`` otherwise.
    """
    run, rep_p, side_p, art = _selection_files(Path(path), prefix)
    missing = [p.name for p in (rep_p, side_p, art) if not p.is_file()]
    if missing:
        raise Stage2SelectionError(
            f"no completed Stage 2 selection in {run}: missing {missing} "
            f"(run 'fs-dino-hq select-stage2' first)")
    try:
        rep = json.loads(rep_p.read_text())
        side = json.loads(side_p.read_text())
    except json.JSONDecodeError as e:
        raise Stage2SelectionError(f"unreadable selection record in {run}: {e}") from e
    if rep.get("status") != "OK":
        raise Stage2SelectionError(
            f"{rep_p.name} status is {rep.get('status')!r}"
            + (f" ({rep['error']})" if rep.get("error") else ""))
    winner = rep.get("winner") or {}
    if not side.get("state_sha256") or side.get("state_sha256") != winner.get("state_sha256"):
        raise Stage2SelectionError(
            f"{side_p.name} state_sha256 {side.get('state_sha256')!r} does not match the "
            f"report's winner {winner.get('state_sha256')!r}")
    digest = file_sha256(art)
    if digest != side.get("file_sha256"):
        raise Stage2SelectionError(
            f"{art.name} changed since selection: sha256 {digest} != recorded "
            f"{side.get('file_sha256')}")
    print(f"[stage2] selected checkpoint verified: epoch {winner.get('epoch')} "
          f"{winner.get('paths')} loss {winner.get('loss')} on "
          f"{(rep.get('selection_plan') or {}).get('label')}")
    return {
        "stage2_ckpt": str(art.resolve()),
        "provenance": POST_TRAINING_SELECTION,
        "run_dir": str(run.resolve()),
        "selection_report": str(rep_p.resolve()),
        "file_sha256": digest,
        "state_sha256": side["state_sha256"],
        "selected_epoch": winner.get("epoch"),
        "selected_from": winner.get("paths"),
        "selection_loss": winner.get("loss"),
        "selection_plan": (rep.get("selection_plan") or {}).get("label"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAM-HQ partial load
# ─────────────────────────────────────────────────────────────────────────────

# SAM-HQ mask-decoder module -> HQMaskDecoder module of the same architecture.
SAM_HQ_MODULES = {"hf_token": "hf_token", "hf_mlp": "hf_mlp",
                  "embedding_encoder": "adapter_proj", "embedding_maskfeature": "decoder_mask_proj"}


def load_sam_hq_partial(model: nn.Module, ckpt_path, save_dir=None) -> Dict:
    """Initialise four HQ modules of ``model.hq_mask_decoder`` from SAM-HQ ViT-H weights.

        mask_decoder.hf_token.*               -> hf_token.*
        mask_decoder.hf_mlp.*                 -> hf_mlp.*
        mask_decoder.embedding_encoder.*      -> adapter_proj.*
        mask_decoder.embedding_maskfeature.*  -> decoder_mask_proj.*

    Every tensor of these four modules must be in the file with the model's
    shape; otherwise a ``ValueError`` is raised, because a wrong file would
    leave the HQ branch at its random initialisation. ``early_proj`` and
    ``dino_l3_proj`` have no SAM-HQ counterpart (``compress_vit_feat`` expects
    1280-channel SAM features) and keep their random initialisation; all other
    tensors of the file are ignored. Writes
    ``<save_dir>/sam_hq_loading_report.json`` when ``save_dir`` is given and
    returns the same report.
    """
    ckpt_path = Path(ckpt_path)
    raw: Dict[str, torch.Tensor] = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    decoder = model.hq_mask_decoder
    own = decoder.state_dict()
    to_load: Dict[str, torch.Tensor] = {}
    problems: List[str] = []
    for src, dst in SAM_HQ_MODULES.items():
        for key in (k for k in own if k.startswith(dst + ".")):
            src_key = f"mask_decoder.{src}.{key[len(dst) + 1:]}"
            if src_key not in raw:
                problems.append(f"{src_key} is missing")
            elif raw[src_key].shape != own[key].shape:
                problems.append(f"{src_key} has shape {tuple(raw[src_key].shape)}, "
                                f"expected {tuple(own[key].shape)}")
            else:
                to_load[key] = raw[src_key]
    if problems:
        raise ValueError(f"{ckpt_path} is not a usable SAM-HQ ViT-H checkpoint: " + "; ".join(problems[:4])
                         + (f" (and {len(problems) - 4} more)" if len(problems) > 4 else ""))
    decoder.load_state_dict(to_load, strict=False)
    random_init = sorted(k for k in decoder.hq_state_dict() if k not in to_load)
    report = {"checkpoint_path": str(ckpt_path), "loaded_count": len(to_load),
              "loaded_keys": sorted(to_load), "random_init_keys": random_init}
    print(f"[sam_hq] {ckpt_path.name}: {len(to_load)} tensors loaded into "
          f"{sorted(SAM_HQ_MODULES.values())}; random initialisation kept for "
          f"{sorted({k.split('.')[0] for k in random_init})}")
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "sam_hq_loading_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Distilled early adapter
# ─────────────────────────────────────────────────────────────────────────────

def new_early_adapter() -> nn.Module:
    """Architecture of the distilled early adapter (768 -> 256 channels)."""
    return DistillationAdapter(in_channels=768, out_channels=256)


def load_early_adapter(ckpt_path, device) -> Tuple[nn.Module, Dict]:
    """Frozen, eval-mode early adapter from a Stage 1 checkpoint
    (``early_adapter_best.pt``, key ``early_adapter_state``, strict load)."""
    ckpt = _load_checkpoint(Path(ckpt_path))
    if "early_adapter_state" not in ckpt:
        raise KeyError(f"{ckpt_path} has no 'early_adapter_state' key (found {list(ckpt.keys())}); "
                       f"expected an early-adapter checkpoint of 'fs-dino-hq distill'")
    kind = (ckpt.get("config") or {}).get("adapter")
    if kind is not None and kind != "early":
        raise ValueError(f"{ckpt_path} is a {kind!r} adapter checkpoint, not an early adapter")
    ea = new_early_adapter()
    ea.load_state_dict(ckpt["early_adapter_state"], strict=True)
    ea.eval()
    for p in ea.parameters():
        p.requires_grad_(False)
    ea.to(device)
    meta = {"early_adapter_epoch": ckpt.get("epoch"),
            "early_adapter_val_mse": ckpt.get("val_mse"),
            "early_adapter_val_cosine": ckpt.get("val_cosine")}
    print(f"[early_adapter] {Path(ckpt_path).name}: epoch={meta['early_adapter_epoch']} "
          f"val_mse={meta['early_adapter_val_mse']} val_cosine={meta['early_adapter_val_cosine']}")
    return ea, meta
