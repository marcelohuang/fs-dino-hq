r"""Sequence-grouped train/val split of an unlabelled image pool for adapter distillation.

    fs-dino-hq make-distill-split --manifest pool/manifest.txt --image-root pool/images \\
        --exclude-dir queries/ --exclude-dir supports/ --exclude-list eval_images.txt \\
        --exclude-sequences-of-excluded --out-dir splits/distill

Frames of the same capture sequence (site visit or video) are near-duplicates,
so an image-level random split leaks across train and validation, and frames
that neighbour an evaluation image leak evaluation content into distillation.
This command

  * groups images by their source sequence: the file stem with the matches of
    ``--group-regex`` removed (default ``[_-]\d+$``, the trailing frame
    counter ``_<digits>`` or ``-<digits>``). Camera-style names such as
    ``IMG_1234`` / ``DSC_0001`` then fall into one group per prefix; pass
    another ``--group-regex`` for such pools (``(?!)`` matches nothing and
    makes every image its own group, an image-level split),
  * removes every image whose stem appears in an exclusion source: a LabelMe
    folder (``--exclude-dir``, stems of its ``*.json`` files) or a text list of
    image names or paths (``--exclude-list``); pass every evaluation-related
    set (query images, the support pool, labelled validation lists),
  * with ``--exclude-sequences-of-excluded``, also removes every frame whose
    sequence contains an excluded stem, so that evaluation images keep no
    near-duplicate neighbours in the pool (the stricter isolation; recommended),
  * assigns whole groups to validation (seeded shuffle) until validation holds
    about ``--val-frac`` of the kept images; groups larger than half of that
    target stay in train so validation spans several sequences,
  * writes ``train.txt`` / ``val.txt`` (entries as in the manifest, relative to
    ``--image-root``), ``excluded.txt`` and ``split_report.json`` (counts,
    groups and the exclusion audit). Paths in the report are recorded exactly
    as given on the command line.

The command fails without writing a split when train or validation would be
empty, and warns when one group holds more than half of the kept images.

Distillation uses no masks, so images of labelled training lists need not be
excluded; see docs/data.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

DEFAULT_GROUP_REGEX = r"[_-]\d+$"
FRAME_RE = re.compile(DEFAULT_GROUP_REGEX)
GROUP_RULE = "stem with the trailing [_-]<digits> frame counter removed"
DOMINANT_GROUP_FRACTION = 0.5        # warn when one group holds more of the kept images


def group_of(stem: str, group_re: "re.Pattern[str]" = FRAME_RE) -> str:
    """Source sequence of a frame: the stem with the matches of ``group_re`` removed
    (default: the trailing frame counter)."""
    return group_re.sub("", stem)


def group_rule(group_re: "re.Pattern[str]") -> str:
    """Human-readable grouping rule recorded in split_report.json."""
    if group_re.pattern == DEFAULT_GROUP_REGEX:
        return GROUP_RULE
    return f"stem with the matches of {group_re.pattern!r} removed"


def stems_from_dir(d: Path) -> Set[str]:
    """Stems of the LabelMe annotations (``*.json``) in a folder."""
    return {p.stem for p in Path(d).glob("*.json")}


def stems_from_list(p: Path) -> Set[str]:
    """Stems of the image names or paths listed one per line in a text file."""
    with open(p) as f:
        return {Path(line.strip()).stem for line in f if line.strip()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fs-dino-hq make-distill-split", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True,
                   help="text file with one image path per line, relative to --image-root")
    p.add_argument("--image-root", default=None,
                   help="folder the manifest entries are relative to (default: current directory); "
                        "every entry must exist")
    p.add_argument("--exclude-dir", action="append", default=[],
                   help="LabelMe folder whose stems are removed (repeatable)")
    p.add_argument("--exclude-list", action="append", default=[],
                   help="text file of image names/paths whose stems are removed (repeatable)")
    p.add_argument("--exclude-sequences-of-excluded", action="store_true",
                   help="also remove every frame whose sequence (filename group) contains an excluded "
                        "image, so evaluation images have no near-duplicate neighbours in the pool")
    p.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX,
                   help="regular expression removed from each file stem (re.sub with an empty "
                        "replacement); what remains is the sequence group (default: %(default)s, the "
                        "trailing frame counter). '(?!)' matches nothing: every image is its own group")
    p.add_argument("--val-frac", type=float, default=0.05, help="target fraction of kept images in val")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True)
    return p


def make_split(entries: List[str], excluded_sets: Dict[str, Set[str]], val_frac: float, seed: int,
               exclude_sequences_of_excluded: bool, group_re: "re.Pattern[str]" = FRAME_RE) -> Dict:
    """Grouped split of ``entries``; returns train/val/excluded lists and the report fields."""
    excl = set().union(*excluded_sets.values()) if excluded_sets else set()

    def group(stem: str) -> str:
        return group_of(stem, group_re)

    kept, excluded = [], []
    for e in entries:
        (excluded if Path(e).stem in excl else kept).append(e)
    n_excluded_by_stem = len(excluded)
    excluded_sequences = sorted({group_of(s) for s in excl})
    n_excluded_by_sequence = 0
    if exclude_sequences_of_excluded:
        seqs = set(excluded_sequences)
        still, dropped = [], []
        for e in kept:
            (dropped if group(Path(e).stem) in seqs else still).append(e)
        kept, excluded = still, excluded + dropped
        n_excluded_by_sequence = len(dropped)

    groups = collections.defaultdict(list)
    for e in kept:
        groups[group(Path(e).stem)].append(e)
    names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(names)
    target = val_frac * len(kept)
    val_groups, n_val = [], 0
    for g in names:
        if n_val >= target:
            break
        # keep very large sequences in train so validation spans several sites
        if len(groups[g]) > 0.5 * target:
            continue
        val_groups.append(g)
        n_val += len(groups[g])
    val_set = set(val_groups)
    train = sorted(e for g in names if g not in val_set for e in groups[g])
    val = sorted(e for g in val_groups for e in groups[g])
    if {group(Path(e).stem) for e in train} & {group(Path(e).stem) for e in val}:
        raise RuntimeError("a sequence group was assigned to both train and val")
    if {Path(e).stem for e in train + val} & excl:
        raise RuntimeError("an excluded image remained in the split")

    manifest_stems = {Path(e).stem for e in entries}
    report = {
        "n_manifest": len(entries), "n_excluded": len(excluded), "n_kept": len(kept),
        "n_excluded_by_stem": n_excluded_by_stem, "n_excluded_by_sequence": n_excluded_by_sequence,
        "exclude_sequences_of_excluded": exclude_sequences_of_excluded,
        "excluded_sequences": excluded_sequences if exclude_sequences_of_excluded else None,
        "n_train": len(train), "n_val": len(val),
        "n_groups_total": len(groups), "n_groups_val": len(val_groups),
        "val_groups": sorted(val_groups),
        "excluded_per_source": {k: len(v & manifest_stems) for k, v in excluded_sets.items()},
        "group_regex": group_re.pattern,
        "group_rule": group_rule(group_re),
        "largest_groups": collections.Counter({g: len(v) for g, v in groups.items()}).most_common(10),
    }
    return {"train": train, "val": val, "excluded": excluded, "report": report}


def split_problems(report: Dict) -> Dict[str, List[str]]:
    """Errors (an empty train or val side) and warnings (one dominant group) of a split report."""
    errors, warnings = [], []
    largest = report["largest_groups"]
    top = ", ".join(f"{g!r}: {n}" for g, n in largest[:5]) or "none"
    hint = (f"groups come from the {report['group_rule']} (--group-regex {report['group_regex']!r}); "
            f"largest groups: {top}")
    if report["n_kept"] == 0:
        errors.append("no images left after the exclusions")
    elif report["n_train"] == 0 or report["n_val"] == 0:
        side = "train" if report["n_train"] == 0 else "val"
        errors.append(f"the {side} split would be empty ({report['n_groups_total']} groups for "
                      f"{report['n_kept']} kept images; a group larger than half of the validation "
                      f"target stays in train); {hint}")
    if largest and report["n_kept"] and largest[0][1] > DOMINANT_GROUP_FRACTION * report["n_kept"]:
        warnings.append(f"group {largest[0][0]!r} holds {largest[0][1]} of the {report['n_kept']} kept "
                        f"images; {hint}")
    return {"errors": errors, "warnings": warnings}


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.val_frac < 1.0:
        parser.error("--val-frac must be in (0, 1)")
    try:
        group_re = re.compile(args.group_regex)
    except re.error as e:
        parser.error(f"--group-regex {args.group_regex!r}: {e}")

    with open(args.manifest) as f:
        entries = [line.strip() for line in f if line.strip()]
    if len(entries) != len({Path(e).stem for e in entries}):
        parser.error(f"duplicate file stems in {args.manifest}")
    root: Optional[Path] = Path(args.image_root) if args.image_root else None
    missing = [e for e in entries if not ((root / e) if root is not None else Path(e)).is_file()]
    if missing:
        where = args.image_root if args.image_root else "the current directory"
        parser.error(f"{len(missing)} manifest entries missing under {where}: {missing[:3]}")

    excluded_sets: Dict[str, Set[str]] = {}
    for d in args.exclude_dir:
        if not Path(d).is_dir():
            parser.error(f"--exclude-dir {d} is not a directory")
        excluded_sets[d] = stems_from_dir(Path(d))
    for f in args.exclude_list:
        excluded_sets[f] = stems_from_list(Path(f))

    split = make_split(entries, excluded_sets, args.val_frac, args.seed,
                       args.exclude_sequences_of_excluded, group_re)
    problems = split_problems(split["report"])
    for w in problems["warnings"]:
        print(f"WARNING: make-distill-split: {w}", file=sys.stderr)
    if problems["errors"]:
        for e in problems["errors"]:
            print(f"ERROR: make-distill-split: {e}", file=sys.stderr)
        print(f"ERROR: nothing written to {args.out_dir}; change --group-regex, --val-frac or the "
              f"exclusions", file=sys.stderr)
        return 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(split["train"]) + "\n")
    (out / "val.txt").write_text("\n".join(split["val"]) + "\n")
    (out / "excluded.txt").write_text("\n".join(sorted(split["excluded"])) + "\n")
    # Paths exactly as given on the command line (nothing is resolved to an absolute path).
    report = {"manifest": args.manifest, "image_root": args.image_root, "seed": args.seed,
              "val_frac_target": args.val_frac, **split["report"], "warnings": problems["warnings"]}
    (out / "split_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("val_groups", "largest_groups", "excluded_sequences")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
