"""Support-query inference with a trained FS-DINO-HQ checkpoint.

Example::

    from fs_dino_hq.inference import Predictor
    from fs_dino_hq.data.labelme import labelme_json_to_mask

    predictor = Predictor.from_checkpoint("fs_dino_hq_5shot.pt",
                                          "dinov2_vitb14_pretrain.pth")
    result = predictor.predict(query_image,
                               support_images=[img1, ..., img5],
                               support_masks=[mask1, ..., mask5])
    result.probability   # float32 [H, W] in [0, 1], query resolution
    result.mask          # uint8   [H, W] in {0, 1}

Conventions (identical to training and to the evaluation in docs/results.md):

* images are PIL images or paths (converted to RGB); masks are 2-D arrays at
  the resolution of their support image, non-zero = foreground;
* every image is letterboxed to 518 x 518 (longest side resized with
  bilinear interpolation, top-left padding); masks use nearest neighbour;
* the K supports are combined by averaging their prompts (one decoder pass);
  the order of the supports changes floating-point summation order only;
* the output is the corrected prediction: canonical Stage 2 base logits plus
  the HQ residual, sigmoid, cropped to the letterbox content box and resized
  bilinearly to the query size through an 8-bit image (so probabilities are
  multiples of 1/255); ``mask = probability >= threshold``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..data.labelme import open_image_rgb
from ..data.letterbox import LetterboxPreprocess, unletterbox_prob
from ..modeling.backbones import build_sam_prompt_decoder, load_dinov2_vitb14
from ..modeling.correlation import detect_corr_arch
from ..modeling.distill_adapter import BottleneckAdapter as DistillationAdapter
from ..modeling.kshot import aggregate_prompts, compute_support_prompts, early_feature
from ..modeling.model import FSDinoHQModel

ImageLike = Union[str, Path, Image.Image]
DINO_PREFIXES = ("last_extractor.", "layer3_extractor.")


def _as_rgb(image: ImageLike) -> Image.Image:
    """RGB image in LabelMe's frame (EXIF orientation applied)."""
    return open_image_rgb(image)


def check_hq_checkpoint(state: Dict) -> None:
    """Refuse checkpoints whose output semantics this package does not implement."""
    if "model_state" not in state:
        raise ValueError(f"not an HQ checkpoint (top-level keys: {sorted(state)[:10]})")
    mode = state.get("decoder_mode")
    if mode != "dual_canonical":
        raise ValueError(f"checkpoint decoder_mode={mode!r}; this package implements "
                         "'dual_canonical' only")
    if state.get("residual_scale_mode", "fixed") != "fixed":
        raise ValueError("checkpoint uses a learned residual scale, which this package does not implement")
    if state.get("early_feature_source") != "distilled_adapter":
        raise ValueError("checkpoint does not use the distilled early adapter")
    ms = state["model_state"]
    if not any(k.startswith("early_adapter.") for k in ms):
        raise ValueError("checkpoint has no early_adapter.* tensors")
    arch, grid = detect_corr_arch(ms)
    if (arch, grid) != ("v2_reduced", 37):
        raise ValueError(f"checkpoint correlation branch {arch} (grid {grid}) is not supported")


def load_model(hq_checkpoint, dinov2_checkpoint, device: Optional[torch.device] = None,
               dinov2_repo: Optional[str] = None) -> FSDinoHQModel:
    """Build ``FSDinoHQModel`` from an HQ checkpoint and the DINOv2 ViT-B/14 weights.

    Every tensor except DINOv2 comes from the HQ checkpoint (Stage 2 adapter,
    MVPG, SAM prompt encoder and mask decoder, HQ modules and the early
    adapter), so the SAM ViT-H checkpoint is not needed. Loading is strict:
    only the DINOv2 extractor keys may be missing, and nothing may be unexpected.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(hq_checkpoint, map_location="cpu", weights_only=True)
    check_hq_checkpoint(state)
    dino = load_dinov2_vitb14(dinov2_checkpoint, hub_repo=dinov2_repo)
    model = FSDinoHQModel(dino_model=dino, sam_full=build_sam_prompt_decoder(),
                          corr_arch="v2_reduced", feat_hw=37,
                          adapter_norm_first=bool(state.get("adapter_norm_first", False)),
                          adapter_dropout=float(state.get("adapter_dropout", 0.1)))
    model.early_adapter = DistillationAdapter(in_channels=768, out_channels=256)
    missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in {hq_checkpoint}: {unexpected[:5]}")
    hard_missing = [k for k in missing if not k.startswith(DINO_PREFIXES)]
    if hard_missing:
        raise RuntimeError(f"keys missing from {hq_checkpoint}: {hard_missing[:5]}")
    model.eval().to(device)
    model.checkpoint_meta = {k: state.get(k) for k in (
        "epoch", "n_shot", "decoder_mode", "residual_scale_mode", "adapter_norm_first",
        "early_feature_source", "selection_tag", "full_val_loss")}
    return model


@dataclass
class Prediction:
    """Outputs at the query resolution (``probability`` and ``mask`` are the model output)."""
    probability: np.ndarray        # float32 [H, W], corrected prediction
    mask: np.ndarray               # uint8 [H, W] in {0, 1}
    base_probability: np.ndarray   # float32 [H, W], canonical Stage 2 base (no HQ branch)
    base_mask: np.ndarray          # uint8 [H, W]
    logits: Dict[str, torch.Tensor]  # [1, 1, 256, 256] letterboxed logits per output


class Predictor:
    """K-shot support-query prediction (see the module docstring for conventions)."""

    def __init__(self, model: FSDinoHQModel, threshold: float = 0.5, dino_size: int = 518) -> None:
        self.model = model.eval()      # BatchNorm running statistics, no dropout
        self.threshold = float(threshold)
        self.dino_size = int(dino_size)
        self.prep = LetterboxPreprocess(dino_size=self.dino_size)
        self.device = next(model.hq_mask_decoder.parameters()).device

    @classmethod
    def from_checkpoint(cls, hq_checkpoint, dinov2_checkpoint, device=None,
                        threshold: float = 0.5, dinov2_repo: Optional[str] = None) -> "Predictor":
        return cls(load_model(hq_checkpoint, dinov2_checkpoint, device, dinov2_repo),
                   threshold=threshold)

    @torch.no_grad()
    def predict(self, query_image: ImageLike, support_images: Sequence[ImageLike],
                support_masks: Sequence[np.ndarray]) -> Prediction:
        if len(support_images) != len(support_masks) or not support_images:
            raise ValueError("need the same, non-zero number of support images and masks")
        model, prep, dev = self.model, self.prep, self.device
        q = _as_rgb(query_image)
        query_dino = prep.image_for_dino(q).unsqueeze(0).to(dev)
        Fq = model.last_extractor(query_dino)
        Fq_l3 = model.layer3_extractor(query_dino)

        sup_imgs = [_as_rgb(s) for s in support_images]
        for img, m in zip(sup_imgs, support_masks):
            if np.asarray(m).shape != (img.height, img.width):
                raise ValueError(f"support mask shape {np.asarray(m).shape} does not match "
                                 f"its image ({img.height}, {img.width})")
        sup_d = torch.stack([prep.image_for_dino(s) for s in sup_imgs], 0).to(dev)
        sup_m = torch.stack([prep.mask_for_dino((np.asarray(m) > 0).astype(np.uint8))
                             for m in support_masks], 0).to(dev)
        prompts = aggregate_prompts(compute_support_prompts(model, sup_d, sup_m, Fq))

        adapter_64 = F.interpolate(model.adapter(Fq_l3), (64, 64), mode="bilinear",
                                   align_corners=False)
        early = early_feature(model, Fq_l3)
        out = model.hq_mask_decoder.predict_dual_canonical(
            image_embeddings=adapter_64, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=prompts["semantic_prompt"],
            dense_prompt_embeddings=prompts["dense_prompt"],
            dino_l3_feat=Fq_l3, adapter_feat=adapter_64, distilled_early_feat=early)

        def to_image(logits):
            grid = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
            return unletterbox_prob(grid, q.width, q.height, self.dino_size)

        prob = to_image(out["corrected_logits"])
        base = to_image(out["canonical_base_logits"])
        return Prediction(
            probability=prob, mask=(prob >= self.threshold).astype(np.uint8),
            base_probability=base, base_mask=(base >= self.threshold).astype(np.uint8),
            logits={k: out[k].cpu() for k in ("corrected_logits", "canonical_base_logits",
                                             "hq_logits", "sam_logits")})
