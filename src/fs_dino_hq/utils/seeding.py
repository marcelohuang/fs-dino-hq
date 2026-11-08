"""Random-seed handling shared by the trainers."""
from __future__ import annotations

import os
import random
import sys
from typing import Dict

import numpy as np
import torch


def seed_everything(seed: int, deterministic_cudnn: bool = False) -> None:
    """Seed python, numpy and torch (all CUDA devices).

    deterministic_cudnn=True additionally disables the cuDNN autotuner and
    requests deterministic cuDNN kernels (Stage 2 and HQ trainers). We deliberately
    do not call torch.use_deterministic_algorithms(True): some SAM/DINO CUDA
    kernels lack deterministic implementations and it would raise at runtime.
    Seeds control ordering, selection and initialisation; exact floating-point
    reproduction is not guaranteed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def worker_init(worker_id: int, base_seed: int) -> None:
    """Top-level (picklable) DataLoader worker seeder."""
    s = base_seed * 100003 + worker_id
    random.seed(s)
    np.random.seed(s % (2 ** 31))
    torch.manual_seed(s)


def rng_metadata(seed: int) -> Dict:
    """Environment and seed record written into a run's config.json."""
    gpu = (torch.cuda.get_device_name(0)
           if torch.cuda.is_available() else "cpu")
    try:
        det_algos = torch.are_deterministic_algorithms_enabled()
    except Exception:
        det_algos = None
    return {"seed": seed,
            "episode_plan_seed": seed,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "gpu_model": gpu,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "torch_deterministic_algorithms": det_algos,
            "worker_seed_policy": "base_seed*100003 + worker_id "
                                  "(base_seed = --seed + epoch); "
                                  "functools.partial(worker_init)",
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "reproducibility_scope": (
                "Episode ordering, split selection and worker seeding are "
                "controlled by recorded seeds. Exact "
                "floating-point or bitwise reproducibility is not guaranteed "
                "across runs or environments.")}
