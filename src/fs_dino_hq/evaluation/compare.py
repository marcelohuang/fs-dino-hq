"""Paired comparison of two evaluations of the same queries (bootstrap over queries).

    fs-dino-hq compare --base runA/metrics.csv --new runB/metrics.csv --out comparison.json

Rows of the two per-image CSV files written by ``fs-dino-hq evaluate`` are
matched by image id; for each metric the per-query difference d = new - base
is resampled ``--n-boot`` times with replacement (``numpy.random.default_rng(seed)``,
one fresh generator per comparison), and the 2.5 / 97.5 percentiles of the
resampled means give a 95 % interval (the generator is drawn sequentially
over ``--metrics`` in the given order). Queries with a NaN for a metric are
dropped for that metric only; a metric without any finite pair is reported as
null and draws nothing. A win is a higher value, except for metrics in
``--lower-better`` (default: fragmentation_ratio). The interval covers
query-sampling variation for the two given sets of predictions; it does not
cover training-run variation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

METRICS = ("iou", "dice", "boundary_f", "thin_recall", "fragmentation_ratio")


def load_rows(path: Path, metrics: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """{image id: {metric: value}} from a per-image CSV of ``fs-dino-hq evaluate``."""
    if not Path(path).is_file():
        raise ValueError(f"{path}: no such file (write it with fs-dino-hq evaluate --out-csv)")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("image", *metrics) if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        out: Dict[str, Dict[str, float]] = {}
        for r in reader:
            if r["image"] in out:
                raise ValueError(f"{path}: image {r['image']!r} listed twice")
            out[r["image"]] = {m: float(r[m]) for m in metrics}
    return out


def paired_bootstrap(base: Dict[str, Dict[str, float]], new: Dict[str, Dict[str, float]],
                     seed: int = 42, n_boot: int = 10000, metrics: Sequence[str] = METRICS,
                     lower_better: Iterable[str] = ("fragmentation_ratio",)) -> Dict:
    rng = np.random.default_rng(seed)
    lower_better = set(lower_better)
    ids = sorted(set(base) & set(new))
    rec: Dict = {"n_queries": len(ids)}
    for m in metrics:
        qs = [q for q in ids if np.isfinite(base[q][m]) and np.isfinite(new[q][m])]
        n = len(qs)
        lower = m in lower_better
        direction = "lower_is_better" if lower else "higher_is_better"
        if n == 0:
            rec[m] = {"base": None, "new": None, "delta": None, "ci95": None,
                      "ci_excludes_zero": False, "win_tie_loss": [0, 0, 0], "n_queries": 0,
                      "direction": direction}
            continue
        x = np.array([base[q][m] for q in qs])
        y = np.array([new[q][m] for q in qs])
        d = y - x
        idx = rng.integers(0, n, size=(n_boot, n))
        lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
        win = int((d < 0).sum()) if lower else int((d > 0).sum())
        tie = int((d == 0).sum())
        rec[m] = {"base": float(x.mean()), "new": float(y.mean()), "delta": float(d.mean()),
                  "ci95": [float(lo), float(hi)], "ci_excludes_zero": bool(lo > 0 or hi < 0),
                  "win_tie_loss": [win, tie, n - win - tie], "n_queries": n,
                  "direction": direction}
    return rec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fs-dino-hq compare", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="per-image metrics CSV of the reference")
    p.add_argument("--new", required=True, help="per-image metrics CSV compared against it")
    p.add_argument("--metrics", nargs="+", default=list(METRICS))
    p.add_argument("--lower-better", nargs="*", default=["fragmentation_ratio"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--allow-different-queries", action="store_true",
                   help="compare on the intersection when the two files list different images")
    p.add_argument("--out", default=None, help="JSON output")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_boot < 1:
        parser.error("--n-boot must be at least 1")
    try:
        base = load_rows(Path(args.base), args.metrics)
        new = load_rows(Path(args.new), args.metrics)
    except ValueError as e:
        raise SystemExit(str(e))
    if set(base) != set(new) and not args.allow_different_queries:
        raise SystemExit(f"the two files list different images ({len(set(base) ^ set(new))} differ)")
    rec = paired_bootstrap(base, new, args.seed, args.n_boot, args.metrics, args.lower_better)
    print(f"n = {rec['n_queries']} queries, delta = new - base, 95 % paired bootstrap interval")
    for m in args.metrics:
        r = rec[m]
        if r["n_queries"] == 0:
            print(f"  {m:20s} no query with finite values in both files")
            continue
        print(f"  {m:20s} base {r['base']:.4f}  new {r['new']:.4f}  delta {r['delta']:+.4f} "
              f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  W/T/L {r['win_tie_loss']}")
    if args.out:
        out = {"base": args.base, "new": args.new, "seed": args.seed, "n_boot": args.n_boot,
               "delta": "new - base", "results": rec}
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
