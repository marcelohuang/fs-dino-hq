"""Hashes recorded in run configs and manifests so a result can be traced to
its exact inputs and code."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable


def file_sha256(path, chunk_size: int = 1024 * 1024) -> str:
    """Streaming SHA-256; never loads a multi-GB checkpoint fully into RAM."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hq_state_sha256(decoder) -> str:
    """Deterministic SHA-256 over the HQ-specific decoder state.

    Sorted keys; hashes key name, dtype, shape and contiguous CPU bytes, so it
    does not depend on dict iteration order. Recorded as
    ``initial_hq_state_sha256`` to show that two runs start from the same
    initial HQ state.
    """
    h = hashlib.sha256()
    sd = decoder.hq_state_dict()
    for k in sorted(sd):
        v = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(v.dtype).encode("utf-8"))
        h.update(str(tuple(v.shape)).encode("utf-8"))
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def code_sha256(paths: Iterable[Path]) -> str:
    """One SHA-256 over the concatenated bytes of the given source files (existing ones only)."""
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def code_hashes_by_name(paths: Iterable[Path]) -> Dict[str, str]:
    """{relative-looking name: sha256} for a list of source files."""
    out: Dict[str, str] = {}
    for p in paths:
        p = Path(p)
        out[f"{p.parent.name}/{p.name}"] = file_sha256(p)
    return out
