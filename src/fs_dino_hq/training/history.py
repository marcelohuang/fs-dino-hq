"""Loss curves from a run's training history.

    fs-dino-hq plot-history RUN_DIR [--out DIR] [--name NAME] [--format png pdf]

The run type is detected from the files in RUN_DIR:

  hq      ``loss_history_{fast,full}.csv`` (train-hq). Selected epoch: ``primary_epoch``
          of ``checkpoint_selection_report.json`` (the copy of the Stage 2 selection
          report kept in an HQ run directory is not used).
  stage2  ``history_fast.csv`` (every epoch) and ``history_full.csv`` (full-validation
          epochs, with ``full_val_<plan>_*`` columns when several plans are validated)
          (train-stage2). Selected epoch: ``winner.epoch`` of
          ``stage2_selection_report.json`` (select-stage2), else the trainer's
          ``primary_epoch`` in ``checkpoint_selection_report.json``.
  stage1  ``history.csv`` (distill): ``stage_start`` and ``epoch`` rows. Training and
          validation losses are measured at the stage resolution and are drawn per
          stage; only the fixed-resolution diagnostic (``diag_*``) is comparable
          across stages. Selected epoch: first minimum of ``diag_mse`` over epoch rows.

Columns are read by name; empty or non-numeric cells and missing optional
columns are skipped. A per-plan series identical to the plain ``full_val_*``
columns (the selection plan) is folded into them. Writes ``<out>/<name>.png``
(and ``.pdf``) and ``<out>/<name>_series.csv`` (columns panel, series, epoch,
value). Needs matplotlib (``pip install fs-dino-hq[plot]``); draws on the Agg canvas.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# fixed colour per role (train, fast val, full val, further plans); ink for markers
COLORS = {"train": "#2a78d6", "fast": "#eb6834", "full": "#1baf7a", "plan": ["#eda100", "#e87ba4", "#6250d6"],
          "ink": "#0b0b0b", "muted": "#52514e"}
HQ_FILES, STAGE2_FILES = ("loss_history_fast.csv", "loss_history_full.csv"), ("history_fast.csv", "history_full.csv")
# HQ first: an HQ run directory also holds Stage 2 reports
RUN_FILES = (("hq", HQ_FILES), ("stage2", STAGE2_FILES), ("stage1", ("history.csv",)))


@dataclass
class Series:
    name: str                 # identifier in the series CSV
    label: Optional[str]      # legend entry (None: not listed, e.g. later stage segments)
    x: List[float]
    y: List[float]
    color: str
    style: str = "-"          # line style, or "*" for single-point markers


@dataclass
class Panel:
    key: str
    title: str
    series: List[Series] = field(default_factory=list)
    logy: bool = False


@dataclass
class History:
    kind: str                                   # "stage1", "stage2" or "hq"
    panels: List[Panel]
    selected_epoch: Optional[int] = None
    selected_source: str = "none"
    stages: List[tuple] = field(default_factory=list)       # stage1: (start epoch, end epoch, label) per stage

    @property
    def selection(self) -> str:
        if self.selected_epoch is None:
            return "no selected epoch found"
        return f"selected epoch {self.selected_epoch} ({self.selected_source})"


def _num(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _read(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def _series(rows, xcol, ycol, label, color, name=None, style="-") -> Series:
    pts = [(x, y) for x, y in ((_num(r.get(xcol)), _num(r.get(ycol))) for r in rows) if x is not None and y is not None]
    return Series(name or ycol, label, [p[0] for p in pts], [p[1] for p in pts], color, style)


def detect_run_type(run_dir) -> str:
    run = Path(run_dir)
    for kind, files in RUN_FILES:
        if any((run / f).is_file() for f in files):
            return kind
    raise FileNotFoundError(f"{run}: no training history found (expected one of "
                            f"{', '.join(f for _, files in RUN_FILES for f in files)})")


def _primary_epoch(run: Path):
    e = _json(run / "checkpoint_selection_report.json").get("primary_epoch")
    return (int(e), "checkpoint_selection_report.json primary_epoch") if e is not None else (None, "none")


def _load_hq(run: Path) -> History:
    fast, full = _read(run / HQ_FILES[0]), _read(run / HQ_FILES[1])
    panel = Panel("loss", "HQ loss (dice + BCE)", [
        _series(fast, "epoch", "train_loss", "train", COLORS["train"]),
        _series(fast, "epoch", "fast_val_loss", "fast val", COLORS["fast"]),
        _series(full, "epoch", "full_val_loss", "full val", COLORS["full"])])
    return History("hq", [panel], *_primary_epoch(run))


def _load_stage2(run: Path) -> History:
    fast, full = _read(run / STAGE2_FILES[0]), _read(run / STAGE2_FILES[1])
    header = list(full[0]) if full else []
    plans = [m.group(1) for m in (re.fullmatch(r"full_val_(.+)_loss", c) for c in header) if m]
    pfx = _json(run / "config.json").get("ckpt_prefix") or "stage2"
    report = _json(run / f"{pfx}_selection_report.json")
    winner = report.get("winner") if report.get("status", "OK") == "OK" else None
    if isinstance(winner, dict) and winner.get("epoch") is not None:
        epoch, source = int(winner["epoch"]), f"{pfx}_selection_report.json winner.epoch"
    else:
        winner, (epoch, source) = None, _primary_epoch(run)
    panels = []
    for m, title in (("loss", "total loss (prior + decoder)"), ("loss_prior", "prior term (dice + BCE)"),
                     ("loss_final", "decoder term (dice + BCE)"), ("iou_final", "decoder IoU")):
        plain = _series(full, "epoch", f"full_val_{m}", "full val", COLORS["full"])
        p = Panel(m, title, [_series(fast, "epoch", f"train_{m}", "train", COLORS["train"]),
                             _series(fast, "epoch", f"fast_val_{m}", "fast val", COLORS["fast"]), plain])
        for i, lab in enumerate(plans):
            s = _series(full, "epoch", f"full_val_{lab}_{m}", f"full val ({lab})",
                        COLORS["plan"][i % len(COLORS["plan"])], style="--")
            if (s.x, s.y) == (plain.x, plain.y):
                plain.label = f"full val ({lab})"
            else:
                p.series.append(s)
        if winner and _num(winner.get(m)) is not None:   # full-validation value of the selected checkpoint
            p.series.append(Series(f"selected_full_val_{m}", "selected (full val)",
                                   [float(epoch)], [_num(winner[m])], COLORS["ink"], "*"))
        panels.append(p)
    return History("stage2", panels, epoch, source)


def _load_stage1(run: Path) -> History:
    rows = _read(run / "history.csv")
    if rows and "diag_mse" not in rows[0]:
        raise ValueError(f"{run / 'history.csv'}: not a Stage 1 distillation history (no diag_mse column)")
    ep = [r for r in rows if r.get("event", "epoch") == "epoch"]
    diag_res = _json(run / "config.json").get("diag_res")
    at = f" at {diag_res} px" if diag_res else ""
    loss = Panel("stage_loss", "loss at the stage resolution", logy=True)
    segments = []        # a stage starts where the previous one ended (stage 0: one epoch before its first)
    for i, st in enumerate(dict.fromkeys(r.get("stage") for r in ep)):
        part = [r for r in ep if r.get("stage") == st]
        res = part[0].get("res") or "?"
        for col, lab, c in (("train_loss", "train", COLORS["train"]), ("val_loss", "val", COLORS["full"])):
            loss.series.append(_series(part, "global_epoch", col, lab if i == 0 else None, c, f"{col}@{res}"))
        xs = [x for x in (_num(r.get("global_epoch")) for r in part) if x is not None]
        if xs:
            segments.append((segments[-1][1] if segments else min(xs) - 1, max(xs), f"{res} px"))
    panels = [loss] + [Panel(f"diag_{m}", f"diagnostic {lab}{at}", [
        _series(ep, "global_epoch", f"diag_{m}", f"diag {lab}", COLORS["train"]),
        _series(ep, "global_epoch", f"diag_{m}_content", f"diag {lab} (content)", COLORS["fast"], style="--")],
        logy=m == "mse") for m, lab in (("mse", "MSE"), ("cosine", "cosine"))]
    mse = panels[1].series[0]
    epoch = int(mse.x[min(range(len(mse.y)), key=mse.y.__getitem__)]) if mse.y else None   # first minimum
    return History("stage1", panels, epoch, "first minimum of diag_mse" if mse.y else "none", segments)


def load_history(run_dir) -> History:
    run = Path(run_dir)
    kind = detect_run_type(run)
    hist = {"hq": _load_hq, "stage2": _load_stage2, "stage1": _load_stage1}[kind](run)
    for p in hist.panels:
        p.series = [s for s in p.series if s.x]
    hist.panels = [p for p in hist.panels if p.series]
    if not hist.panels:
        raise ValueError(f"{run}: the {kind} history files contain no numeric rows")
    return hist


def write_series_csv(hist: History, path) -> Path:
    path = Path(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "series", "epoch", "value"])
        w.writerows([p.key, s.name, int(x) if x.is_integer() else x, repr(y)]
                    for p in hist.panels for s in p.series for x, y in zip(s.x, s.y))
    return path


def _figure_class():
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as e:
        raise ImportError("plot-history needs matplotlib: pip install fs-dino-hq[plot]") from e
    return Figure, FigureCanvasAgg


def plot_history(hist: History, out_dir, name: str, formats: Sequence[str] = ("png",)) -> List[Path]:
    """Draw one panel per metric (shared epoch axis); returns the written figure paths."""
    Figure, FigureCanvasAgg = _figure_class()     # Agg canvas, no pyplot state
    n = len(hist.panels)
    ncols = 2 if n > 3 else 1
    nrows = math.ceil(n / ncols)
    fig = Figure(figsize=(6.4 * ncols, 2.6 * nrows + 0.8), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(nrows, ncols, sharex=True, squeeze=False).ravel()
    for ax, p in zip(axes, hist.panels):
        for s in p.series:      # style "*": single-point marker, drawn on top
            ax.plot(s.x, s.y, s.style, color=s.color, label=s.label, linewidth=1.6, markersize=11,
                    zorder=3 if s.style == "*" else 2)
        for x0, _, _ in hist.stages[1:]:
            ax.axvline(x0, color=COLORS["muted"], linestyle=":", linewidth=1)
        if p is hist.panels[0]:
            for x0, x1, text in hist.stages:
                ax.text((x0 + x1) / 2, 0.97, text, transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=8, color=COLORS["muted"])
        if hist.selected_epoch is not None:
            ax.axvline(hist.selected_epoch, color=COLORS["ink"], linestyle="--", linewidth=1,
                       label=f"selected epoch {hist.selected_epoch}")
        if p.logy:
            ax.set_yscale("log")
        ax.set_title(p.title, fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=7, framealpha=0.85, edgecolor="none")
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[max(0, n - ncols):n]:
        ax.set_xlabel("epoch (all stages)" if hist.stages else "epoch")
        ax.xaxis.set_tick_params(labelbottom=True)
    fig.suptitle(f"{name} ({hist.kind})\n{hist.selection}", fontsize=10)
    paths = [Path(out_dir) / f"{name}.{fmt}" for fmt in formats]
    for path in paths:
        fig.savefig(path, dpi=150)
    return paths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fs-dino-hq plot-history", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="training output directory (Stage 1, Stage 2 or HQ)")
    p.add_argument("--out", default=None, help="output directory (default: RUN_DIR)")
    p.add_argument("--name", default=None, help="output file stem (default: the run directory name)")
    p.add_argument("--format", nargs="+", default=["png"], choices=["png", "pdf"], help="figure formats")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run = Path(args.run_dir)
    try:
        _figure_class()
        hist = load_history(run)
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"plot-history: {e}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else run
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or run.resolve().name
    written = plot_history(hist, out, name, args.format) + [write_series_csv(hist, out / f"{name}_series.csv")]
    print(f"{run}: {hist.kind}; {hist.selection}")
    print("\n".join(f"  wrote {path}" for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
