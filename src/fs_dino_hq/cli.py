"""Command-line entry point: ``fs-dino-hq <command> [options]``.

Each command is implemented by a module with a ``main(argv)`` function and
can also be run as ``python -m <module>``.
"""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "make-distill-split": ("fs_dino_hq.data.distill_split",
                           "split unlabelled images for adapter distillation"),
    "distill": ("fs_dino_hq.training.distill",
                "Stage 1: coarse-to-fine distillation of the final or early adapter"),
    "make-episodes": ("fs_dino_hq.data.episode_plans",
                      "scene-disjoint split and K-shot episode plans for a LabelMe dataset"),
    "train-stage2": ("fs_dino_hq.training.train_stage2", "Stage 2: episodic training"),
    "select-stage2": ("fs_dino_hq.training.select_stage2",
                      "Stage 2: full-validation selection among saved checkpoints"),
    "train-hq": ("fs_dino_hq.training.train_hq", "HQ branch training on a selected Stage 2 checkpoint"),
    "predict": ("fs_dino_hq.inference.predict", "predict masks for query images from annotated supports"),
    "evaluate": ("fs_dino_hq.evaluation.evaluate", "metrics of predicted masks against LabelMe annotations"),
    "compare": ("fs_dino_hq.evaluation.compare", "paired bootstrap comparison of two evaluations"),
    "plot-history": ("fs_dino_hq.training.history", "loss curves from a run's training history"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = ["usage: fs-dino-hq <command> [options]", "", "commands:"]
    lines += [f"  {name:<{width}}  {desc}" for name, (_, desc) in COMMANDS.items()]
    lines += ["", "Run 'fs-dino-hq <command> --help' for the options of a command."]
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    if argv[0] == "--version":
        from . import __version__
        print(__version__)
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"unknown command {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[name][0])
    ret = module.main(rest)
    return int(ret or 0)


if __name__ == "__main__":
    raise SystemExit(main())
