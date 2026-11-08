"""Frozen backbones: DINOv2 ViT-B/14 and the SAM ViT-H components.

DINOv2 is instantiated from its ``torch.hub`` definition
(``facebookresearch/dinov2``, pinned to commit ``DINOV2_HUB_COMMIT``) and its
weights are loaded from a local checkpoint file; SAM comes from the
``segment_anything`` package. All backbones are returned frozen and in eval
mode.

The DINOv2 code is pinned because building the model draws from the global
random generator before the other modules are built, so the hub code takes
part in the random initialisation of the trained modules. At the pinned
commit the files that build ``dinov2_vitb14`` (``hubconf.py``,
``dinov2/hub``, ``dinov2/models``, ``dinov2/layers``) are identical to the
code used for the reported runs. The repository's ``hubconf.py`` also imports
its Cell-DINO and XRay-DINO hub modules, which carry their own licences; only
``dinov2_vitb14`` is used here.

``torch.hub`` needs network access the first time it fetches the DINOv2 code
(it is cached under ``$TORCH_HOME/hub/facebookresearch_dinov2_<commit>``). On
machines without network access, point ``FS_DINO_HQ_DINOV2_REPO`` (or the
``hub_repo`` argument) at a local clone of
https://github.com/facebookresearch/dinov2 checked out at ``DINOV2_HUB_COMMIT``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

DINOV2_HUB_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_HUB_REPO = f"facebookresearch/dinov2:{DINOV2_HUB_COMMIT}"
DINOV2_MODEL = "dinov2_vitb14"
DINOV2_REPO_ENV = "FS_DINO_HQ_DINOV2_REPO"


def freeze_(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def dinov2_hub_source(hub_repo: Optional[str] = None) -> Tuple[str, str]:
    """Return ``(repo_or_dir, source)`` for ``torch.hub.load``.

    The first non-empty value of ``hub_repo``, ``$FS_DINO_HQ_DINOV2_REPO`` and
    ``DINOV2_HUB_REPO`` is used. An existing directory (``~`` is expanded) is
    a local clone (``source="local"``); anything else is a GitHub reference
    ``owner/repo[:ref]`` (``source="github"``). A value that looks like a path
    (absolute, or starting with ``.`` or ``~``) but is not a directory raises
    ``FileNotFoundError`` instead of being sent to GitHub.
    """
    repo = hub_repo or os.environ.get(DINOV2_REPO_ENV) or DINOV2_HUB_REPO
    path = os.path.expanduser(repo)
    if os.path.isdir(path):
        return path, "local"
    if os.path.isabs(path) or repo.startswith((".", "~")):
        raise FileNotFoundError(f"DINOv2 hub directory not found: {repo}")
    return repo, "github"


_hub_source = dinov2_hub_source  # earlier private name


def build_dinov2_vitb14(hub_repo: Optional[str] = None) -> nn.Module:
    """DINOv2 ViT-B/14 architecture with random initial weights, frozen and in eval mode.

    The hub code is resolved by ``dinov2_hub_source``. Building the model
    draws from the global torch random generator. For a GitHub reference,
    ``skip_validation=True`` is passed because ``torch.hub`` otherwise accepts
    only branch or tag names (or the commits at their heads), not a pinned
    older commit.
    """
    repo, source = dinov2_hub_source(hub_repo)
    extra = {"skip_validation": True} if source == "github" else {}
    model = torch.hub.load(repo, DINOV2_MODEL, pretrained=False, source=source, **extra)
    return freeze_(model)


def load_dinov2_vitb14(ckpt_path, device: Optional[torch.device] = None,
                       hub_repo: Optional[str] = None) -> nn.Module:
    """DINOv2 ViT-B/14 with weights from ``ckpt_path`` (dinov2_vitb14_pretrain.pth)."""
    model = build_dinov2_vitb14(hub_repo)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(device) if device is not None else model


def load_sam_vit_h(ckpt_path, device: Optional[torch.device] = None) -> nn.Module:
    """Full SAM ViT-H (image encoder, prompt encoder, mask decoder) from sam_vit_h_4b8939.pth.

    Used by the trainers: the image encoder is the distillation teacher, and
    the prompt encoder and mask decoder initialise Stage 2. The trainers build
    the full model even when the image encoder is not used, because the
    construction order fixes how the seeded random generator is consumed.
    """
    sam_full = sam_model_registry["vit_h"](checkpoint=None)
    sam_full.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    freeze_(sam_full)
    return sam_full.to(device) if device is not None else sam_full


class SamPromptDecoder(nn.Module):
    """SAM prompt encoder + mask decoder without the image encoder."""

    def __init__(self, prompt_encoder: nn.Module, mask_decoder: nn.Module) -> None:
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder


def build_sam_prompt_decoder() -> SamPromptDecoder:
    """Uninitialised SAM ViT-H prompt encoder and mask decoder.

    Same hyper-parameters as ``segment_anything.build_sam._build_sam``. Used at
    inference, where every weight of these modules comes from the HQ
    checkpoint, so the SAM ViT-H checkpoint is not needed.
    """
    prompt_embed_dim, image_size, vit_patch_size = 256, 1024, 16
    image_embedding_size = image_size // vit_patch_size
    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth=2, embedding_dim=prompt_embed_dim,
                                      mlp_dim=2048, num_heads=8),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    return freeze_(SamPromptDecoder(prompt_encoder, mask_decoder))
