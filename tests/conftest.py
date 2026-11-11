"""Shared pytest configuration: make tests/stubs.py importable as ``stubs``; device fixtures."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def cpu_only(monkeypatch):
    """Make the trainers run on the CPU even when a GPU is visible.

    The trainers select CUDA whenever it is available. Tests that require two
    runs to agree bit for bit use this fixture, because that equality is a
    property of CPU arithmetic; the same scenarios are checked on a GPU with a
    tolerance by the ``test_gpu_*`` tests, which need a CUDA device.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def check_gpu_error(name: str, error: float, tolerance: float) -> None:
    """Assert ``error <= tolerance`` and print both (shown with ``pytest -rP`` or ``-s``)."""
    print(f"[gpu-check] {name}: error {error:.3e} (tolerance {tolerance:.0e})")
    assert error <= tolerance, f"{name}: {error:.3e} > {tolerance:.0e}"
