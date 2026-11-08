"""Freeze policy of HQ-branch training.

Only the HQ modules of the decoder (``HQMaskDecoder.HQ_MODULE_NAMES``) train.
Everything else is frozen (``requires_grad=False``) and kept in eval mode:
DINOv2, the final adapter, the MVPG (semantic, prototype-cosine and
correlation branches), the SAM prompt encoder, the Stage 2 decoder weights
(transformer, IoU/mask tokens, upscaling, hypernetworks, IoU head) and the
distilled early adapter.

``model.train()`` puts every submodule back into train mode, which would let
the BatchNorm layers of the frozen adapters and MVPG update their running
statistics; the trainer therefore re-applies ``apply_hq_freeze`` after every
``model.train()``.

apply_hq_freeze(model)        set requires_grad and train/eval modes
build_hq_optimizer(model)     AdamW over the trainable (HQ) parameters
freeze_audit(model)           trainable/frozen counts per module (freeze_audit.json)
verify_freeze(model)          raise if the policy does not hold; returns the audit
snapshot_frozen_params(model) / check_frozen_unchanged(model, snapshot)
                              byte-level check that a step left frozen tensors alone
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ..modeling.hq_decoder import HQMaskDecoder

HQ_PREFIX = "hq_mask_decoder."


def is_hq_parameter(name: str) -> bool:
    """True for a parameter name of the full model that belongs to an HQ module."""
    return name.startswith(HQ_PREFIX) and HQMaskDecoder.is_hq_key(name[len(HQ_PREFIX):])


# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────

def apply_hq_freeze(model: nn.Module) -> None:
    """HQ modules trainable (train mode); every other module frozen (eval mode).

    Called after the weights are loaded and again after each ``model.train()``.
    """
    # DINOv2 extractors (one shared backbone)
    _freeze(model.last_extractor)
    _freeze(model.layer3_extractor)
    # final adapter and MVPG (Stage 2)
    _freeze(model.adapter)
    for attr in ("semantic_branch", "pcs_branch", "corr_branch"):
        _freeze(getattr(model, attr))
    # Stage 2 decoder weights frozen, HQ modules trainable
    _set_decoder_modules(model.hq_mask_decoder, hq=False, train=False)
    _set_decoder_modules(model.hq_mask_decoder, hq=True, train=True)
    # SAM prompt encoder (dense positional encoding only)
    _freeze(model.prompt_encoder)
    # Distilled early adapter. Its BatchNorm layers must stay in eval mode, so
    # this is needed after every model.train().
    if getattr(model, "early_adapter", None) is not None:
        _freeze(model.early_adapter)


def build_hq_optimizer(model: nn.Module, lr: float = 1e-4,
                       weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """AdamW with one parameter group: the parameters with ``requires_grad``,
    in ``model.parameters()`` order (after ``apply_hq_freeze``: the HQ modules
    in their construction order hf_token, hf_mlp, dino_l3_proj, adapter_proj,
    early_proj, decoder_mask_proj)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters; call apply_hq_freeze(model) first")
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    print(f"[freeze] optimizer built: {sum(p.numel() for p in trainable):,} trainable params")
    return opt


# ─────────────────────────────────────────────────────────────────────────────
# Audit and checks
# ─────────────────────────────────────────────────────────────────────────────

def freeze_audit(model: nn.Module) -> Dict:
    """Trainable/frozen parameter counts by module (first two name components).

    Parameters shared between modules (the DINOv2 backbone of both extractors)
    are counted once, under the first name.
    """
    audit: Dict = {"trainable_params": 0, "frozen_params": 0, "by_module": {}}
    for name, p in model.named_parameters():
        top = ".".join(name.split(".")[:2])
        rec = audit["by_module"].setdefault(top, {"trainable": 0, "frozen": 0})
        if p.requires_grad:
            audit["trainable_params"] += p.numel()
            rec["trainable"] += p.numel()
        else:
            audit["frozen_params"] += p.numel()
            rec["frozen"] += p.numel()
    audit["trainable_top_modules"] = sorted(
        m for m, v in audit["by_module"].items() if v["trainable"] > 0)
    return audit


def verify_freeze(model: nn.Module) -> Dict:
    """Check the HQ policy and return ``freeze_audit(model)``.

    Raises RuntimeError if a non-HQ parameter is trainable, an HQ parameter is
    frozen, or a frozen module (anything outside the HQ modules, apart from the
    model and decoder containers) is in train mode.
    """
    wrong_trainable = [n for n, p in model.named_parameters()
                       if p.requires_grad and not is_hq_parameter(n)]
    wrong_frozen = [n for n, p in model.named_parameters()
                    if not p.requires_grad and is_hq_parameter(n)]
    in_train_mode = [n for n, m in model.named_modules()
                     if n not in ("", "hq_mask_decoder") and m.training
                     and not is_hq_parameter(n)]
    problems: List[str] = []
    if wrong_trainable:
        problems.append(f"non-HQ parameters trainable: {wrong_trainable[:5]}")
    if wrong_frozen:
        problems.append(f"HQ parameters frozen: {wrong_frozen[:5]}")
    if in_train_mode:
        problems.append(f"frozen modules in train mode: {in_train_mode[:5]}")
    if problems:
        raise RuntimeError("HQ freeze policy violated: " + "; ".join(problems))
    audit = freeze_audit(model)
    print(f"[freeze] trainable={audit['trainable_params']:,} frozen={audit['frozen_params']:,} "
          f"modules={audit['trainable_top_modules']}")
    return audit


def snapshot_frozen_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """CPU copies of all frozen parameters.

    Covers parameters only. BatchNorm running statistics are buffers; they are
    protected by keeping the frozen modules in eval mode.
    """
    return {name: p.detach().cpu().clone()
            for name, p in model.named_parameters() if not p.requires_grad}


def check_frozen_unchanged(model: nn.Module,
                           snapshot: Dict[str, torch.Tensor]) -> Tuple[bool, List[str]]:
    """(all unchanged, names of changed tensors), compared bit for bit."""
    params = dict(model.named_parameters())
    changed = [name for name, orig in snapshot.items()
               if not torch.equal(params[name].detach().cpu(), orig)]
    return not changed, changed


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _freeze(module: nn.Module) -> None:
    """requires_grad=False and eval mode (eval keeps BatchNorm statistics fixed)."""
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def _set_decoder_modules(decoder: nn.Module, hq: bool, train: bool) -> None:
    """Set mode and requires_grad of the HQ (``hq=True``) or the Stage 2
    (``hq=False``) submodules of the decoder. The decoder container itself is
    left as is, because it holds both kinds."""
    names = decoder.HQ_MODULE_NAMES
    for mod_name, mod in decoder.named_modules():
        if mod_name and (mod_name.split(".")[0] in names) == hq:
            mod.train(train)
    for param_name, param in decoder.named_parameters():
        if (param_name.split(".")[0] in names) == hq:
            param.requires_grad = train
