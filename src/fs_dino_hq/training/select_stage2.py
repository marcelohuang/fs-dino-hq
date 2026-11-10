"""Post-training checkpoint selection for a Stage 2 run.

Command: ``fs-dino-hq select-stage2 --run-dir <stage2 run>``.

Why: during training, full validation runs every ``--full-val-every`` epochs,
so a checkpoint saved by another rule (fast-validation best, periodic snapshot,
final epoch) may never have been scored on the full plan, and such a
checkpoint can be better than the training-time best.

What this does (``select_stage2_checkpoint``):
  1. enumerate every saved candidate ``<prefix>_*.pt`` of the run (snapshots,
     training-time full-val best, fast-val best, final/last), validate it
     (format ``stage2_model``, adapter convention and correlation architecture
     recorded) and deduplicate by a SHA-256 of the model state (identical
     weights under several names are one candidate);
  2. score every distinct state on the SAME complete, fixed selection plan with
     ``fs_dino_hq.training.train_stage2.evaluate_checkpoint`` (identical
     preprocessing, loss, shot count and settings), plus the run's other
     validation plans as diagnostics;
  3. reuse a cached score only when its state hash AND settings signature
     (plan sha256, shot count, dino size, preprocessing, loss definition,
     evaluation implementation hash, batch size, label filter, feat_hw,
     adapter convention, correlation architecture) match; fast-validation and
     training-time numbers are never used;
  4. pick the minimum full-validation TOTAL loss (prior + decoder); ties broken
     by lower decoder loss, then higher final IoU, then lower epoch;
  5. write ``stage2_selection_report.json`` and copy the winner atomically to
     ``stage2_selected.pt`` (+ ``stage2_selected.json`` sidecar); the
     training-time ``stage2_best.pt`` is never modified.

Scope: the best among the SAVED candidates; epochs that were never saved cannot
be selected. Resumable and idempotent: scores are appended to
``stage2_selection_cache.json`` after each evaluation; a rerun recomputes
nothing compatible and rewrites the artifact only when the winner changed. Any
unreadable candidate or an incomplete plan fails the step (non-zero exit,
report status FAILED, no new artifact).

The dataset root and the backbone checkpoints default to the run's
config.json; ``--synthetic-root``, ``--dino-ckpt`` and ``--sam-ckpt`` override
them (for example after moving the data), and the values used are recorded in
the report. Selection and diagnostic plans can be overridden with
``--select-plan`` / ``--diagnostic-plans``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from fs_dino_hq.config import parse_with_config
from fs_dino_hq.utils.provenance import file_sha256

EVAL_IMPL_VERSION = "stage2_eval_v3"     # bump when evaluate_checkpoint / stage2_loss / letterbox semantics change
LOSS_DEFINITION = "dice(eps=1e-6)+BCE on prior logits (37x37 nearest mask) + dice+BCE on decoder logits (256x256 nearest mask)"
PREPROCESS = "letterbox_longest_side_topleft_pad_bilinear_nearest"

# Package files that define the evaluation (trainer/evaluator, Stage 2 model and
# its components, loss, preprocessing, episode loading); their bytes enter the
# cache signature.
_EVAL_CODE = ("training/train_stage2.py", "modeling/stage2.py", "training/losses.py",
              "data/letterbox.py", "data/episodes.py", "data/labelme.py",
              "modeling/model.py", "modeling/correlation.py")


def model_state_sha256(sd: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def code_sha256() -> str:
    """Hash of the evaluation implementation (see _EVAL_CODE)."""
    pkg = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for rel in _EVAL_CODE:
        h.update((pkg / rel).read_bytes())
    return h.hexdigest()


def _stat_id(path) -> List:
    """[resolved path, size, modification time]: identifies a file without reading it."""
    p = Path(path).resolve()
    if not p.is_file():
        return [str(p), None, None]
    st = p.stat()
    return [str(p), st.st_size, st.st_mtime_ns]


def plan_data_fingerprint(plan_path, root) -> str:
    """Hash over (relative path, size, modification time) of every image and
    annotation the plan refers to, so that scores are not reused for other data
    under the same plan file."""
    with open(plan_path, newline="") as f:
        rows = list(csv.DictReader(f))
    rel = sorted({v for r in rows for k, v in r.items() if k.endswith(("_image", "_json")) and v})
    h = hashlib.sha256()
    for name in rel:
        st = (Path(root) / name).stat()
        h.update(f"{name}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def settings_signature(plan_path, n_shot: int, cfg: Dict, batch_size: int, code_hash: str,
                       inputs: Dict) -> Dict:
    """Everything a cached score depends on besides the model weights."""
    return {"eval_impl": EVAL_IMPL_VERSION, "code_sha256": code_hash,
            "val_plan_sha256": file_sha256(plan_path), "n_shot": int(n_shot),
            "data": plan_data_fingerprint(plan_path, inputs["synthetic_root"]),
            "dino_ckpt": _stat_id(inputs["dino_ckpt"]), "sam_ckpt": _stat_id(inputs["sam_ckpt"]),
            "dino_size": int(cfg["dino_size"]), "feat_hw": int(cfg.get("feat_hw", 37)),
            "preprocess": PREPROCESS, "loss": LOSS_DEFINITION, "batch_size": int(batch_size),
            "label_filter": cfg.get("label_filter"),
            "adapter_arch": {"norm_first": bool(cfg["adapter_arch"]["norm_first"]),
                             "dropout": float(cfg["adapter_arch"]["dropout"])},
            "corr_arch": cfg.get("corr_arch", "v2_reduced")}


def signature_key(sig: Dict) -> str:
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:24]


def enumerate_candidates(run: Path, prefix: str) -> List[Path]:
    """Every saved checkpoint of the run except the selection artifact itself."""
    return sorted(p for p in run.glob(f"{prefix}_*.pt") if p.name != f"{prefix}_selected.pt")


def validate_candidate(path: Path) -> Tuple[Dict, Dict[str, torch.Tensor]]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != "stage2_model" or "stage2_model" not in ck:
        raise ValueError(f"{path.name}: not a stage2_model checkpoint")
    if "adapter_norm_first" not in ck:
        raise ValueError(f"{path.name}: adapter convention not recorded (needed for strict HQ loading)")
    if "epoch" not in ck:
        raise ValueError(f"{path.name}: no epoch")
    if "corr_arch" not in ck:
        raise ValueError(f"{path.name}: correlation architecture not recorded")
    return ck, ck["stage2_model"]


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def rank_key(r: Dict):
    """min total loss; ties: lower decoder loss, higher IoU, lower epoch (deterministic)."""
    return (round(r["loss"], 10), round(r["loss_final"], 10), -round(r["iou_final"], 10), int(r["epoch"]))


def select_stage2_checkpoint(run_dir, select_plan: Optional[Tuple[str, str, int]] = None,
                             diagnostic_plans: Optional[List[Tuple[str, str, int]]] = None,
                             batch_size: int = 4, num_workers: int = 4, scorer=None,
                             load_backbones=None, prefix: Optional[str] = None,
                             synthetic_root: Optional[str] = None, dino_ckpt: Optional[str] = None,
                             sam_ckpt: Optional[str] = None) -> Dict:
    """Returns the selection report (also written to <run>/<prefix>_selection_report.json).

    scorer(model, path, loader, device, load) -> metrics dict is injectable for tests;
    the default is fs_dino_hq.training.train_stage2.evaluate_checkpoint.
    synthetic_root / dino_ckpt / sam_ckpt default to the run's config.json values.
    """
    from torch.utils.data import DataLoader
    from fs_dino_hq.data.episodes import PlanEpisodeDataset, collate_plan_episodes
    from fs_dino_hq.data.letterbox import LetterboxPreprocess
    from fs_dino_hq.modeling.stage2 import Stage2Model, apply_stage2_freeze
    from fs_dino_hq.training import train_stage2 as trainer

    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text())
    pfx = prefix or cfg.get("ckpt_prefix", "stage2")
    scorer = scorer or trainer.evaluate_checkpoint
    load_backbones = load_backbones or trainer.load_backbones
    report_path, cache_path = run / f"{pfx}_selection_report.json", run / f"{pfx}_selection_cache.json"
    selected_path, sidecar_path = run / f"{pfx}_selected.pt", run / f"{pfx}_selected.json"
    inputs = {"synthetic_root": synthetic_root or cfg["synthetic_root"],
              "dino_ckpt": dino_ckpt or cfg["dino_ckpt"], "sam_ckpt": sam_ckpt or cfg["sam_ckpt"]}
    inputs["overridden"] = sorted(k for k, v in (("synthetic_root", synthetic_root), ("dino_ckpt", dino_ckpt),
                                                 ("sam_ckpt", sam_ckpt)) if v)

    # plans: selection plan = the run's --select-on plan unless overridden; others = diagnostics
    specs = [trainer.parse_plan_spec(x) for x in cfg["val_plans"]] if cfg.get("val_plans") else [("val1", cfg["val_plan"], 1)]
    sel_label = cfg.get("select_on") or specs[0][0]
    if select_plan is None:
        select_plan = next(x for x in specs if x[0] == sel_label)
    if diagnostic_plans is None:
        diagnostic_plans = [x for x in specs if x[0] != select_plan[0]]
    plans = [select_plan] + list(diagnostic_plans)
    code_hash = code_sha256()
    sigs = {lab: settings_signature(path, k, cfg, batch_size, code_hash, inputs) for lab, path, k in plans}
    keys = {lab: signature_key(sigs[lab]) for lab in sigs}

    # candidates + dedup
    files = enumerate_candidates(run, pfx)
    if not files:
        raise FileNotFoundError(f"no {pfx}_*.pt candidates in {run}")
    states: Dict[str, Dict] = {}
    for f in files:
        ck, sd = validate_candidate(f)
        h = model_state_sha256(sd)
        rec = states.setdefault(h, {"state_sha256": h, "paths": [], "epoch": int(ck["epoch"]),
                                    "selection_tags": [], "adapter_norm_first": bool(ck["adapter_norm_first"])})
        if rec["epoch"] != int(ck["epoch"]):
            raise ValueError(f"{f.name}: identical weights recorded with different epochs "
                             f"({rec['epoch']} vs {ck['epoch']})")
        rec["paths"].append(f.name); rec["selection_tags"].append(ck.get("selection_tag"))
    print(f"[select] {len(files)} checkpoint files -> {len(states)} distinct model states")

    cache: Dict = json.loads(cache_path.read_text()) if cache_path.is_file() else {}

    # which evaluations are missing?
    todo = [(h, lab) for h in states for lab in keys if keys[lab] not in cache.get(h, {})]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if todo:
        dino_model, sam_full = load_backbones(inputs["dino_ckpt"], inputs["sam_ckpt"])
        model = Stage2Model(dino_model, sam_full, feat_hw=int(cfg.get("feat_hw", 37)),
                            adapter_norm_first=bool(cfg["adapter_arch"]["norm_first"]),
                            adapter_dropout=float(cfg["adapter_arch"]["dropout"]),
                            corr_arch=cfg.get("corr_arch", "v2_reduced")).to(device)
        del sam_full        # the model keeps the prompt encoder and the mask decoder
        apply_stage2_freeze(model)
        prep = LetterboxPreprocess(dino_size=int(cfg["dino_size"]))
        loaders = {}
        for lab, path, k in plans:
            ds = PlanEpisodeDataset(Path(path), Path(inputs["synthetic_root"]), k, prep, cfg.get("label_filter"), limit=0)
            loaders[lab] = (DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_plan_episodes, pin_memory=device.type == "cuda"), len(ds))
        for h in states:
            loaded = False
            for lab, path, k in plans:
                if keys[lab] in cache.get(h, {}):
                    continue
                loader, n_plan = loaders[lab]
                res = scorer(model, run / states[h]["paths"][0], loader, device, load=not loaded)
                loaded = True
                if int(res["episodes"]) != n_plan:
                    raise RuntimeError(f"{states[h]['paths'][0]} on {lab}: evaluated {res['episodes']} of {n_plan} episodes")
                entry = {k_: res[k_] for k_ in ("loss", "loss_prior", "loss_final", "iou_final", "iou_prior", "episodes", "seconds")}
                entry.update({"epoch": states[h]["epoch"], "signature": sigs[lab], "plan": lab,
                              "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "source_file": states[h]["paths"][0]})
                cache.setdefault(h, {})[keys[lab]] = entry
                _atomic_write_json(cache_path, cache)           # resumable after every evaluation
                print(f"[select] {states[h]['paths'][0]:28s} epoch {states[h]['epoch']:>3} [{lab}] total {res['loss']:.4f} "
                      f"prior {res['loss_prior']:.4f} decoder {res['loss_final']:.4f} iou {res['iou_final']:.4f} ({res['seconds']}s)")

    # assemble rows
    sel = select_plan[0]
    rows = []
    for h, st in states.items():
        e = cache[h][keys[sel]]
        row = {**{k_: st[k_] for k_ in ("state_sha256", "paths", "epoch", "selection_tags", "adapter_norm_first")},
               "plan": sel, "episodes": e["episodes"], "loss": e["loss"], "loss_prior": e["loss_prior"],
               "loss_final": e["loss_final"], "iou_final": e["iou_final"], "iou_prior": e["iou_prior"],
               "reused": (h, sel) not in todo, "computed_at": e["computed_at"],
               "diagnostics": {lab: {k_: cache[h][keys[lab]][k_] for k_ in ("loss", "loss_prior", "loss_final", "iou_final", "episodes")}
                               for lab, _, _ in diagnostic_plans}}
        rows.append(row)
    rows.sort(key=rank_key)
    winner = rows[0]
    winner_file = run / sorted(winner["paths"])[0]
    by_decoder = min(rows, key=lambda r: (r["loss_final"], r["loss"], int(r["epoch"])))
    by_iou = max(rows, key=lambda r: (r["iou_final"], -r["loss"], -int(r["epoch"])))

    # artifact: atomic copy, verified, only when changed
    changed = True
    if selected_path.is_file() and sidecar_path.is_file():
        side = json.loads(sidecar_path.read_text())
        changed = side.get("state_sha256") != winner["state_sha256"]
    if changed:
        tmp = selected_path.with_suffix(".pt.tmp")
        shutil.copyfile(winner_file, tmp)
        ck_sel, sd_sel = validate_candidate(tmp)
        if model_state_sha256(sd_sel) != winner["state_sha256"]:
            tmp.unlink(); raise RuntimeError("selected artifact hash mismatch after copy")
        tmp.replace(selected_path)
        _atomic_write_json(sidecar_path, {"source_file": winner_file.name, "state_sha256": winner["state_sha256"],
                                          "file_sha256": file_sha256(selected_path), "epoch": winner["epoch"],
                                          "selection_plan": sel, "loss": winner["loss"], "rule": "min full-val total loss",
                                          "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"), "run_dir": str(run)})
    report = {"status": "OK", "run_dir": str(run), "prefix": pfx,
              "rule": ("minimum full-validation TOTAL loss (prior + decoder) on the selection plan; ties: lower decoder "
                       "loss, then higher final IoU, then lower epoch"),
              "scope": ("best among the SAVED candidate checkpoints of this run (snapshots, training-time full-val best, "
                        "fast-val best, final); epochs that were never saved cannot be selected"),
              "selection_plan": {"label": sel, "path": select_plan[1], "n_shot": select_plan[2], "signature": sigs[sel]},
              "diagnostic_plans": {lab: {"path": path, "n_shot": k, "signature": sigs[lab]} for lab, path, k in diagnostic_plans},
              "inputs": inputs,
              "n_checkpoint_files": len(files), "n_distinct_states": len(states),
              "n_fresh_evaluations": len(todo), "n_reused_evaluations": len(states) * len(plans) - len(todo),
              "candidates": rows,
              "winner": {"paths": winner["paths"], "state_sha256": winner["state_sha256"], "epoch": winner["epoch"],
                         "loss": winner["loss"], "loss_prior": winner["loss_prior"], "loss_final": winner["loss_final"],
                         "iou_final": winner["iou_final"], "artifact": selected_path.name, "artifact_updated": changed},
              "preferred_by_decoder_loss": {"paths": by_decoder["paths"], "epoch": by_decoder["epoch"], "loss_final": by_decoder["loss_final"]},
              "preferred_by_final_iou": {"paths": by_iou["paths"], "epoch": by_iou["epoch"], "iou_final": by_iou["iou_final"]},
              "training_time_best": {"full_val": [r["paths"] for r in rows if "best_full_val" in r["selection_tags"]],
                                     "fast_val": [r["paths"] for r in rows if "best_fast_val" in r["selection_tags"]]},
              "cache_file": cache_path.name, "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    _atomic_write_json(report_path, report)
    print(f"[select] winner epoch {winner['epoch']} ({winner['paths']}) total {winner['loss']:.4f} -> {selected_path.name} "
          f"({'updated' if changed else 'unchanged'}); fresh {len(todo)}, reused {report['n_reused_evaluations']}")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fs-dino-hq select-stage2",
                                description="Post-training Stage 2 checkpoint selection over all saved candidates")
    p.add_argument("--config", default=None, help="JSON with argument names as keys; CLI wins")
    p.add_argument("--run-dir", default=None, help="Stage 2 run directory (reads its config.json)")
    p.add_argument("--select-plan", default=None, help="label=path:k (default: the run's --select-on plan)")
    p.add_argument("--diagnostic-plans", nargs="*", default=None, help="label=path:k ... (default: the run's other plans)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--synthetic-root", default=None, help="dataset root (default: the run's config.json value)")
    p.add_argument("--dino-ckpt", default=None, help="DINOv2 checkpoint (default: the run's config.json value)")
    p.add_argument("--sam-ckpt", default=None, help="SAM ViT-H checkpoint (default: the run's config.json value)")
    return p


def _report_path_for_failure(run: Path) -> Path:
    """Where a FAILED report goes: the run's prefix when config.json is readable."""
    try:
        pfx = json.loads((run / "config.json").read_text()).get("ckpt_prefix", "stage2")
    except Exception:
        pfx = "stage2"
    return run / f"{pfx}_selection_report.json"


def main(argv=None) -> int:
    p = build_parser()
    args = parse_with_config(p, argv)
    if not args.run_dir:
        p.error("missing required argument --run-dir (CLI or --config)")
    from fs_dino_hq.training.train_stage2 import parse_plan_spec
    try:
        sel = parse_plan_spec(args.select_plan) if args.select_plan else None
        diag = [parse_plan_spec(x) for x in args.diagnostic_plans] if args.diagnostic_plans is not None else None
    except argparse.ArgumentTypeError as e:
        p.error(str(e))
    try:
        rep = select_stage2_checkpoint(args.run_dir, sel, diag, args.batch_size, args.num_workers,
                                       synthetic_root=args.synthetic_root, dino_ckpt=args.dino_ckpt,
                                       sam_ckpt=args.sam_ckpt)
    except Exception as e:      # explicit failure, no new artifact
        run = Path(args.run_dir)
        if run.is_dir():
            _atomic_write_json(_report_path_for_failure(run),
                               {"status": "FAILED", "error": repr(e), "run_dir": str(run),
                                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        print(f"[select] FAILED: {e!r}", file=sys.stderr)
        return 1
    return 0 if rep["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
