"""Dataset check, group-disjoint train/val split and K-shot episode plans.

    fs-dino-hq make-episodes --synthetic-root DATA --out-dir manifests \\
        --n-shot 5 --cycles 8 --suffix _5shot_8cyc

CPU only. ``--synthetic-root`` holds one folder per class (``--classes``) with
LabelMe JSON files and their images, for example
``<root>/beams/real-<scene>-<camera>-<id>.png`` plus the ``.json`` of the same
stem (docs/data.md).

1. Inventory. Every ``*.json`` under a class folder (recursive, sorted) is
   paired with an image: ``imagePath`` (absolute, relative to the JSON folder,
   or relative to the class folder), otherwise the JSON stem with one of
   IMG_EXTENSIONS. Malformed JSON, missing images, images already paired with
   another JSON, missing ``imageHeight``/``imageWidth``, annotations without a
   polygon of at least three points (after ``--label-filter``) or with a total
   polygon area below one pixel, and, with ``--verify-images``, unreadable
   images are excluded and listed in ``invalid_files.json``.
2. Grouping. ``--group-mode scene`` (default) takes the scene id from the image
   stem with ``--scene-regex`` (named group ``scene``, case-insensitive). If
   any valid stem does not match, ``scene_parse_failures.json`` is written and
   the command stops without writing a split. ``--group-mode image`` makes
   every image its own group: an image-level split, for datasets without scene
   ids, that is NOT scene-disjoint.
3. Split, per class. Groups are ordered by (number of images, tie rank, id),
   with tie ranks drawn from ``random.Random(f"{seed}:{class}")``. The prefix of
   this order whose image count is closest to ``--val-frac`` of the class is
   held out for validation, keeping at least one group and ``--n-shot`` + 1
   images for training.
4. Plans. Training: in each of ``--cycles`` cycles every training image is the
   query exactly once. Per (cycle, class), ``random.Random(f"{seed}:{cycle}:{class}")``
   shuffles the queries and then draws K = ``--n-shot`` distinct supports of the
   same class from the other groups (from any other image when fewer than K
   remain). The classes are interleaved with
   ``random.Random(f"{seed}:{cycle}:interleave")`` and
   ``episode_idx = cycle * episodes_per_cycle + position``. Validation: every
   validation image once (one cycle, seed + 1) with supports drawn from the
   TRAINING images of its class. The query order depends neither on K nor on
   later cycles, so for the same seed and split the 1-shot and 5-shot plans
   list the same queries, and a plan with C cycles is the first C cycles of a
   plan with more.

Outputs in ``--out-dir``; ``--suffix`` is inserted before the extension of the
names marked *, so several plan sets can share one directory (the file
contents do not depend on it):

  synthetic_train_manifest.csv, synthetic_val_manifest.csv
      component,image_path,json_path,group,n_shapes,width,height,split
  train_episode_plan*.csv, val_episode_plan*.csv
      cycle,component,query_image,query_json,query_group,
      support_1_image..support_K_image,support_1_json..support_K_json,
      support_1_group..support_K_group,episode_idx
  dataset_inventory.json   counts per class (and against --expected)
  invalid_files.json       excluded files and reasons
  split_audit*.json        split algorithm, held-out groups, scene parsing
  episode_coverage*.json   query/support usage of the training plan
  audit_config*.json       the options of this run

Paths in all outputs are relative to ``--synthetic-root``. If the manifests
already in ``--out-dir`` describe a different split, nothing is written.
The plans used for the reported runs (docs/data.md):

  --n-shot 1 --cycles 8 --suffix _1shot
  --n-shot 5 --cycles 2 --suffix _5shot --no-verify-images
  --n-shot 5 --cycles 8 --suffix _5shot_8cyc --no-verify-images
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional

# Ordered: the stem fallback tries these in order, each in lower then upper
# case, and the first existing file wins (fs_dino_hq.data.labelme keeps an
# unordered set for its own lookups).
IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
DEFAULT_CLASSES = ["beams", "columns", "slabs", "walls"]
# real-{scene}-{camera}-{uid}; the scene is the coarse group held out together.
DEFAULT_SCENE_REGEX = r"^real[-_](?P<scene>\d+)[-_](?P<cam>\d+)[-_](?P<uid>\d+)$"
GROUP_MODES = ("scene", "image")
MANIFEST_FILES = ("synthetic_train_manifest.csv", "synthetic_val_manifest.csv")
SCENE_PARSE_FAILURES = "scene_parse_failures.json"


def output_names(suffix: str = "") -> Dict[str, str]:
    """File names written to ``--out-dir``; ``suffix`` only renames the per-plan files."""
    return {
        "train_manifest": MANIFEST_FILES[0],
        "val_manifest": MANIFEST_FILES[1],
        "train_plan": f"train_episode_plan{suffix}.csv",
        "val_plan": f"val_episode_plan{suffix}.csv",
        "inventory": "dataset_inventory.json",
        "invalid": "invalid_files.json",
        "coverage": f"episode_coverage{suffix}.json",
        "audit_config": f"audit_config{suffix}.json",
        "split_audit": f"split_audit{suffix}.json",
    }


def resolve_image(json_path: Path, root: Path, image_path_in_json):
    """Resolve the image for a LabelMe JSON. Returns (path|None, how)."""
    cands = []
    if image_path_in_json:
        ip = Path(image_path_in_json)
        if ip.is_absolute():
            cands.append((ip, "absolute imagePath"))
        cands.append((json_path.parent / ip, "imagePath relative to json dir"))
        cands.append((root / ip, "imagePath relative to component root"))
    for ext in IMG_EXTENSIONS:
        cands.append((json_path.with_suffix(ext), f"stem match {ext}"))
        cands.append((json_path.with_suffix(ext.upper()), f"stem match {ext.upper()}"))
    for cand, how in cands:
        if cand.exists() and cand.is_file():
            return cand, how
    return None, "no candidate exists"


def polygon_area(points) -> float:
    """Shoelace area of a polygon given as [[x, y], ...]."""
    n = len(points)
    a = 0.0
    for i in range(n):
        x1, y1 = points[i][:2]
        x2, y2 = points[(i + 1) % n][:2]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def compile_scene_regex(pattern: str) -> "re.Pattern[str]":
    """Compile a scene pattern (case-insensitive); it must define the group ``scene``."""
    rx = re.compile(pattern, re.IGNORECASE)
    if "scene" not in rx.groupindex:
        raise ValueError(f"scene regex {pattern!r} has no named group (?P<scene>...)")
    return rx


def parse_scene(cls: str, stem: str, scene_re: "re.Pattern[str]"):
    """Return (group_id, parsed_ok). A failed parse gets a transient
    ``<cls>/UNPARSED/<stem>`` label used only for the failure report; scene
    mode never writes a split when any stem fails."""
    m = scene_re.match(stem)
    if m:
        return f"{cls}/{m.group('scene')}", True
    return f"{cls}/UNPARSED/{stem.lower()}", False


def _closest_prefix_subset(scenes_sorted, sizes, target, min_train_left, min_train_scenes):
    """Choose the prefix of the ordered groups for validation whose image sum
    is closest to ``target``, keeping >= 1 validation group and leaving enough
    images and groups for training. Returns the chosen group ids (None if no
    prefix is allowed)."""
    n = len(scenes_sorted)
    total = sum(sizes)
    prefix = 0
    best_choice, best_dev = None, None
    for p in range(1, n):  # never take ALL groups (must leave >= 1 for train)
        prefix += sizes[p - 1]
        train_left = total - prefix
        train_scenes_left = n - p
        if train_left < min_train_left or train_scenes_left < min_train_scenes:
            break
        dev = abs(prefix - target)
        if best_dev is None or dev < best_dev:
            best_dev, best_choice = dev, scenes_sorted[:p]
    return best_choice


def scene_disjoint_split(valid_by_class, seed, val_frac, n_shot, group_mode: str = "scene"):
    """Group-disjoint train/val split (whole groups held out); exits if a
    class cannot be split with valid episodes.

    Group order: (group_size, seeded_tie_rank, group_id). The tie rank comes
    from a local RNG seeded by (seed, class), so equal-sized groups are ordered
    reproducibly and a different seed can pick a different equal-sized group.
    No global RNG state is used. In scene mode the groups are scenes."""
    train_rows, val_rows = [], []
    audit = {"algorithm": "ascending (scene_size, seeded_tie_rank, scene_id); "
                          "prefix closest to val_frac; whole scenes held out; "
                          "≥1 val scene, ≥1 train scene, train images ≥ n_shot+1 "
                          "per component. tie_rank = Random((seed, component)).",
             "seed": seed, "val_frac": val_frac, "n_shot": n_shot,
             "group_mode": group_mode, "per_component": {}, "overlap_ok": True}
    if group_mode == "image":
        audit["note"] = ("image mode: every image is its own group; the split is "
                         "image-level, NOT scene-disjoint.")
    for cls, items in valid_by_class.items():
        by_scene = defaultdict(list)
        for it in items:
            by_scene[it["group"]].append(it)
        n_scenes = len(by_scene)
        if n_scenes < 2:
            raise SystemExit(
                f"[FATAL] {cls}: only {n_scenes} group(s) ({group_mode} mode); a "
                f"group-disjoint split is impossible. Check the file names, "
                f"--classes and --scene-regex.")
        # seeded tie-break: one deterministic rank per group from (seed, cls)
        trng = random.Random(f"{seed}:{cls}")
        tie_rank = {g: trng.random() for g in sorted(by_scene)}
        scenes_sorted = sorted(
            by_scene, key=lambda g: (len(by_scene[g]), tie_rank[g], g))
        sizes = [len(by_scene[g]) for g in scenes_sorted]
        target = max(1, int(round(len(items) * val_frac)))
        chosen = _closest_prefix_subset(
            scenes_sorted, sizes, target,
            min_train_left=n_shot + 1, min_train_scenes=1)
        if not chosen:
            raise SystemExit(
                f"[FATAL] {cls}: cannot hold out any group while leaving valid "
                f"episodes ({len(items)} imgs, {n_scenes} groups, n_shot={n_shot}).")
        val_scenes = set(chosen)
        train_scenes = [g for g in scenes_sorted if g not in val_scenes]
        n_val = sum(len(by_scene[g]) for g in val_scenes)
        for it in items:
            in_val = it["group"] in val_scenes
            (val_rows if in_val else train_rows).append(
                {**it, "split": "val" if in_val else "train"})
        audit["per_component"][cls] = {
            "n_images": len(items), "n_scenes": n_scenes,
            "val_scenes": sorted(val_scenes), "train_scenes": sorted(train_scenes),
            "n_val_scenes": len(val_scenes), "n_train_scenes": len(train_scenes),
            "n_val_images": n_val, "n_train_images": len(items) - n_val,
            "val_frac_actual": round(n_val / len(items), 4),
            "val_frac_target": val_frac,
            "scene_sizes": dict(zip(scenes_sorted, sizes)),
            "tie_rank_seed": f"{seed}:{cls}",
        }
    # global overlap assertion (no group in both splits)
    tr_scenes = {r["group"] for r in train_rows}
    va_scenes = {r["group"] for r in val_rows}
    overlap = tr_scenes & va_scenes
    audit["overlap_ok"] = not overlap
    audit["overlapping_scenes"] = sorted(overlap)
    if overlap:
        raise SystemExit(f"[FATAL] group overlap between train/val: {sorted(overlap)[:5]}")
    return train_rows, val_rows, audit


def index_class(root: Path, cls: str, label_filter, verify_images: bool, cap: int):
    """Pair and check the LabelMe files of one class folder.

    Returns (valid, invalid, label_counts, n_unpaired_images). Valid records
    carry ``group=None``; the grouping step fills it in. Paths in both lists
    are relative to ``root``."""
    cls_dir = root / cls
    invalid, valid = [], []
    labels = Counter()
    if not cls_dir.is_dir():
        return valid, invalid, labels, 0
    json_files = sorted(cls_dir.rglob("*.json"))
    if cap > 0:
        json_files = json_files[:cap]
    img_owner = {}
    for jf in json_files:
        json_rel = jf.relative_to(root).as_posix()
        rec = {"json": json_rel}
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            invalid.append({**rec, "reason": f"malformed json: {e!r}"})
            continue
        if not isinstance(data, dict):
            invalid.append({**rec, "reason": "malformed json: not an object"})
            continue
        img, how = resolve_image(jf, cls_dir, data.get("imagePath"))
        if img is None:
            invalid.append({**rec, "reason": "image not found",
                            "imagePath": data.get("imagePath")})
            continue
        key = str(img.resolve())
        if key in img_owner:
            invalid.append({**rec, "reason": f"duplicate: image already paired "
                                             f"with {img_owner[key]}"})
            continue
        img_owner[key] = json_rel
        h, w = data.get("imageHeight"), data.get("imageWidth")
        if not h or not w:
            invalid.append({**rec, "reason": "missing imageHeight/imageWidth"})
            continue
        n_valid_shapes, area = 0, 0.0
        for sh in data.get("shapes", []):
            labels[sh.get("label", "")] += 1
            if label_filter and sh.get("label", "") not in label_filter:
                continue
            pts = sh.get("points", [])
            if len(pts) >= 3:
                n_valid_shapes += 1
                area += polygon_area(pts)
        if n_valid_shapes == 0 or area < 1.0:
            invalid.append({**rec, "reason": "empty mask after label filter",
                            "n_shapes": len(data.get("shapes", []))})
            continue
        if verify_images:
            try:
                from PIL import Image
                with Image.open(img) as im:
                    im.verify()
            except Exception as e:
                reason = f"corrupt image: {e!r}"
                if img.is_relative_to(root):     # report the path relative to the root
                    reason = reason.replace(str(img), img.relative_to(root).as_posix())
                invalid.append({**rec, "reason": reason})
                continue
        valid.append({
            "component": cls,
            "image_path": img.relative_to(root).as_posix(),
            "json_path": json_rel,
            "group": None,
            "n_shapes": n_valid_shapes,
            "width": int(w), "height": int(h),
        })
    n_unpaired_images = sum(
        1 for p in cls_dir.rglob("*")
        if p.suffix.lower() in IMG_EXTENSIONS and str(p.resolve()) not in img_owner)
    return valid, invalid, labels, n_unpaired_images


def assign_scene_groups(valid_by_class, scene_re):
    """Set ``group`` = ``<class>/<scene>`` on every valid record.

    Returns (per-class parse statistics, failures per class, total failures)."""
    parse_stats, all_failures, total_fail = {}, {}, 0
    for cls, items in valid_by_class.items():
        ok, fails = 0, []
        for it in items:
            stem = Path(it["image_path"]).stem
            g, parsed = parse_scene(cls, stem, scene_re)
            it["group"] = g
            if parsed:
                ok += 1
            else:
                fails.append(it["image_path"])
        n_fail = len(fails)
        total_fail += n_fail
        all_failures[cls] = fails
        parsed_scenes = sorted({it["group"] for it in items
                                if not it["group"].startswith(f"{cls}/UNPARSED/")})
        by_scene = Counter(it["group"] for it in items
                           if not it["group"].startswith(f"{cls}/UNPARSED/"))
        parse_stats[cls] = {
            "n_images": len(items), "parsed_ok": ok, "parse_fail": n_fail,
            "n_scenes": len(parsed_scenes), "scene_examples": parsed_scenes[:5],
            "failure_examples": fails[:5],
            "largest_scene": max(by_scene.values()) if by_scene else 0,
            "smallest_scene": min(by_scene.values()) if by_scene else 0,
            "scene_size_distribution": dict(Counter(by_scene.values())),
        }
    return parse_stats, all_failures, total_fail


def build_plan(pairs_by_class, rng_seed: int, n_shot: int, cycles: int,
               support_pool_by_class=None):
    """Full-coverage plan: each image queries once per cycle; class-mixed order.

    Supports come from ``support_pool_by_class`` (default: the query pool) and
    from groups other than the query's when the pool has several groups."""
    support_pool_by_class = support_pool_by_class or pairs_by_class
    plan, support_use = [], Counter()
    for cyc in range(cycles):
        episodes = []
        for cls, items in pairs_by_class.items():
            pool = support_pool_by_class[cls]
            by_group = defaultdict(list)
            for it in pool:
                by_group[it["group"]].append(it)
            multi_group = len(by_group) > 1
            rng = random.Random(f"{rng_seed}:{cyc}:{cls}")
            queries = items[:]
            rng.shuffle(queries)
            for q in queries:
                if multi_group:
                    cand = [it for it in pool if it["group"] != q["group"]]
                else:
                    cand = [it for it in pool
                            if it["image_path"] != q["image_path"]]
                if len(cand) < n_shot:
                    cand = [it for it in pool
                            if it["image_path"] != q["image_path"]]
                sup = rng.sample(cand, k=n_shot)
                for s in sup:
                    support_use[s["image_path"]] += 1
                episodes.append({
                    "cycle": cyc, "component": cls,
                    "query_image": q["image_path"], "query_json": q["json_path"],
                    "query_group": q["group"],
                    **{f"support_{i+1}_image": s["image_path"] for i, s in enumerate(sup)},
                    **{f"support_{i+1}_json": s["json_path"] for i, s in enumerate(sup)},
                    **{f"support_{i+1}_group": s["group"] for i, s in enumerate(sup)},
                })
        random.Random(f"{rng_seed}:{cyc}:interleave").shuffle(episodes)
        for i, ep in enumerate(episodes):
            ep["episode_idx"] = cyc * len(episodes) + i
        plan.extend(episodes)
    return plan, support_use


def csv_text(rows, fieldnames=None) -> str:
    """CSV with a header from the first row's keys (``\\r\\n`` line ends, the
    csv module default); an empty string when there are no rows."""
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames or list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def write_csv(path: Path, rows, fieldnames=None) -> None:
    """Stream rows to ``path`` in the format of :func:`csv_text`."""
    if not rows:
        write_text(path, "")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_text(path: Path, text: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(text)


def parse_expected(spec: Optional[str]) -> Dict[str, int]:
    """``"beams=8128,columns=7387"`` -> {"beams": 8128, "columns": 7387}."""
    if not spec:
        return {}
    return {k: int(v) for k, v in (kv.split("=") for kv in spec.split(","))}


def audit_config(args) -> Dict:
    """Options recorded in audit_config.json (no paths)."""
    return {
        "classes": args.classes,
        "label_filter": args.label_filter,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "n_shot": args.n_shot,
        "group_mode": args.group_mode,
        "scene_regex": args.scene_regex if args.group_mode == "scene" else None,
        "cycles": args.cycles,
        "expected": args.expected,
        "verify_images": args.verify_images,
        "max_per_class": args.max_per_class,
    }


def make_episodes(args) -> int:
    """Run the inventory, split and plan generation for parsed ``args``."""
    root, out = Path(args.synthetic_root), Path(args.out_dir)
    if not root.is_dir():
        raise SystemExit(f"[FATAL] dataset root not found: {root}")
    out.mkdir(parents=True, exist_ok=True)
    names = output_names(args.suffix)
    expected = parse_expected(args.expected)

    inventory, all_invalid, label_dist = {}, {}, {}
    valid_by_class = {}
    for cls in args.classes:
        valid, invalid, labels, unpaired = index_class(
            root, cls, args.label_filter, args.verify_images, args.max_per_class)
        valid_by_class[cls] = valid
        all_invalid[cls] = invalid
        label_dist[cls] = dict(labels.most_common())
        inventory[cls] = {
            "expected_pairs": expected.get(cls),
            "valid_pairs": len(valid),
            "invalid_pairs": len(invalid),
            "images_without_json": unpaired,
            "delta_vs_expected": (len(valid) + len(invalid)) - (expected.get(cls) or 0),
            "n_groups": None,          # set by the grouping step below
        }
        print(f"[{cls}] valid={len(valid)} invalid={len(invalid)} "
              f"unpaired_images={unpaired} labels={list(labels)[:5]}")

    split_audit = {"group_mode": args.group_mode, "seed": args.seed,
                   "val_frac": args.val_frac, "n_shot": args.n_shot}

    if args.group_mode == "scene":
        # Strict: any valid stem that fails the scene parser aborts before a
        # manifest or plan is written; there is no silent per-image fallback.
        scene_re = compile_scene_regex(args.scene_regex)
        parse_stats, all_failures, total_fail = assign_scene_groups(valid_by_class, scene_re)
        for cls, st in parse_stats.items():
            inventory[cls]["n_groups_original"] = st["n_scenes"]
            inventory[cls]["n_groups"] = st["n_scenes"]
            inventory[cls]["grouping_mode"] = "scene"
            inventory[cls]["scene_parse_ok"] = st["parsed_ok"]
            inventory[cls]["scene_parse_fail"] = st["parse_fail"]
            print(f"[{cls}] scenes={st['n_scenes']} parsed_ok={st['parsed_ok']}/{st['n_images']}"
                  + (f"  PARSE_FAIL={st['parse_fail']}" if st["parse_fail"] else ""))
        if total_fail:
            # keep the failure report before aborting
            write_text(out / SCENE_PARSE_FAILURES, json.dumps(
                {"group_mode": "scene", "total_fail": total_fail,
                 "regex": scene_re.pattern, "per_component": parse_stats,
                 "failures": all_failures}, indent=2))
            raise SystemExit(
                f"[FATAL] scene parsing failed for {total_fail} valid file(s); "
                f"see {out / SCENE_PARSE_FAILURES}. Refusing to write a "
                f"scene-disjoint split with unparsed file names.")
    else:
        for cls, items in valid_by_class.items():
            for it in items:
                it["group"] = f"{cls}/{Path(it['image_path']).stem}"
            inventory[cls]["n_groups"] = len({it["group"] for it in items})
            inventory[cls]["grouping_mode"] = "image"
        parse_stats = None

    train_rows, val_rows, sd = scene_disjoint_split(
        valid_by_class, args.seed, args.val_frac, args.n_shot, args.group_mode)
    split_audit.update(sd)
    if parse_stats is not None:
        split_audit["scene_parse"] = parse_stats
    for cls in args.classes:
        inventory[cls]["n_train"] = sum(1 for r in train_rows if r["component"] == cls)
        inventory[cls]["n_val"] = sum(1 for r in val_rows if r["component"] == cls)
        inventory[cls]["grouping_signal"] = args.group_mode == "scene"

    train_by_class = {c: [r for r in train_rows if r["component"] == c]
                      for c in args.classes}
    val_by_class = {c: [r for r in val_rows if r["component"] == c]
                    for c in args.classes}
    train_plan, support_use = build_plan(train_by_class, args.seed,
                                         args.n_shot, args.cycles)
    val_plan, _ = build_plan(val_by_class, args.seed + 1, args.n_shot, cycles=1,
                             support_pool_by_class=train_by_class)

    n_train = len(train_rows)
    coverage = {
        "cycles": args.cycles,
        "episodes_total": len(train_plan),
        "query_coverage": "every train image appears as query exactly once per cycle (by construction)",
        "support_usage": {
            "images_used_as_support": len(support_use),
            "images_never_support": n_train - len(support_use),
            "min": min(support_use.values()) if support_use else 0,
            "max": max(support_use.values()) if support_use else 0,
            "mean": (sum(support_use.values()) / max(1, len(support_use))),
        },
        "val_episodes": len(val_plan),
    }
    inventory["_total"] = {
        "expected": sum(v for v in expected.values()),
        "valid": n_train + len(val_rows),
        "train": n_train, "val": len(val_rows),
        "invalid": sum(len(v) for v in all_invalid.values()),
    }

    # Plans in this directory built from other manifests would no longer match
    # them, so a changed split is refused before anything is written.
    manifests = {names["train_manifest"]: csv_text(train_rows),
                 names["val_manifest"]: csv_text(val_rows)}
    changed = [n for n, text in manifests.items()
               if (out / n).exists() and (out / n).read_bytes() != text.encode("utf-8")]
    if changed:
        raise SystemExit(
            f"[FATAL] {', '.join(changed)} in {out} describe a different split than "
            f"this run; plans already there were built from it. Use a new --out-dir "
            f"or remove the old files.")

    for name, text in manifests.items():
        write_text(out / name, text)
    write_csv(out / names["train_plan"], train_plan)
    write_csv(out / names["val_plan"], val_plan)
    write_text(out / names["inventory"], json.dumps(
        {"inventory": inventory, "label_distribution": label_dist}, indent=2))
    write_text(out / names["invalid"], json.dumps(all_invalid, indent=2))
    write_text(out / names["coverage"], json.dumps(coverage, indent=2))
    write_text(out / names["audit_config"], json.dumps(audit_config(args), indent=2))
    write_text(out / names["split_audit"], json.dumps(split_audit, indent=2))

    tot = inventory["_total"]
    print(f"[total] valid={tot['valid']} (train {tot['train']}, val {tot['val']}) "
          f"invalid={tot['invalid']}; train episodes={len(train_plan)}, "
          f"val episodes={len(val_plan)}")
    if expected and tot["valid"] + tot["invalid"] != tot["expected"]:
        print(f"[WARN] pairs found ({tot['valid'] + tot['invalid']}) != expected "
              f"({tot['expected']}); see {names['inventory']} / {names['invalid']}.")
    print(f"[OK] manifests and plans written to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fs-dino-hq make-episodes", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic-root", required=True,
                   help="dataset root with one folder of LabelMe pairs per class")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES),
                   help="class folders, in this order (default: %(default)s)")
    p.add_argument("--label-filter", nargs="+", default=None,
                   help="count only shapes with these labels when checking for an empty "
                        "mask (default: all labels)")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="target fraction of each class held out for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-shot", type=int, default=5, help="supports per episode (K)")
    p.add_argument("--cycles", type=int, default=2,
                   help="training-plan cycles; each training image is the query once per cycle")
    p.add_argument("--group-mode", choices=GROUP_MODES, default="scene",
                   help="scene: groups parsed with --scene-regex (scene-disjoint split); "
                        "image: every image its own group (image-level split)")
    p.add_argument("--scene-regex", default=DEFAULT_SCENE_REGEX,
                   help="pattern matched against the image stem, case-insensitive; the named "
                        "group 'scene' is the scene id (default: %(default)s)")
    p.add_argument("--expected", default=None,
                   help='expected pairs per class, e.g. "beams=100,walls=80"; '
                        "reported in dataset_inventory.json")
    p.add_argument("--verify-images", action=argparse.BooleanOptionalAction, default=True,
                   help="check that every image file can be read (slower)")
    p.add_argument("--max-per-class", type=int, default=-1,
                   help="debugging: use only the first N JSON files of each class (-1 = all)")
    p.add_argument("--suffix", default="",
                   help="inserted before the extension of the plan, coverage, split-audit and "
                        "config files, e.g. _5shot -> train_episode_plan_5shot.csv")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_shot < 1:
        parser.error("--n-shot must be >= 1")
    if args.cycles < 1:
        parser.error("--cycles must be >= 1")
    if not 0.0 < args.val_frac < 1.0:
        parser.error("--val-frac must be in (0, 1)")
    if len(set(args.classes)) != len(args.classes):
        parser.error("--classes lists a class twice")
    if "/" in args.suffix or "\\" in args.suffix:
        parser.error("--suffix must not contain a path separator")
    try:
        compile_scene_regex(args.scene_regex)
    except (re.error, ValueError) as e:
        parser.error(f"--scene-regex: {e}")
    try:
        parse_expected(args.expected)
    except ValueError:
        parser.error('--expected must look like "beams=100,walls=80"')
    return make_episodes(args)


if __name__ == "__main__":
    raise SystemExit(main())
