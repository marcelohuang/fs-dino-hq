# Implementation notes

This document describes how the code relates to the methods it builds on, and
the conventions that checkpoints depend on. "Paper" refers to an explicit
statement in the cited publication; anything else is a choice made in this
implementation. Statements about FS-DINO refer to the first arXiv version of
the paper [4] (see the note under References).

## Building blocks

| Component | Origin | Use here |
|---|---|---|
| DINOv2 ViT-B/14 [1] | `torch.hub` repository `facebookresearch/dinov2` (pinned commit, see `modeling/backbones.py`), entry point `dinov2_vitb14`, with the official weights | frozen feature extractor (last-layer tokens and block-3 tokens) |
| SAM ViT-H [2] | `segment_anything` package and official ViT-H weights | image encoder: distillation teacher only; prompt encoder (dense positional encoding) and mask decoder |
| HQ-SAM [3] | design of `MaskDecoderHQ` and the official `sam_hq_vit_h.pth` weights | HQ output token, its MLP and the feature-fusion layers of the HQ branch |
| FS-DINO [4] | method description (paper); code ported from an earlier, unpublished implementation (see below) | bottleneck adapter, coarse-to-fine distillation from SAM, meta visual prompt generator (MVPG) with 4D correlation, episodic training |

No code from the official FS-DINO repository is used. The FS-DINO components
(the adapter and MVPG classes in `modeling/model.py`, `Correlation4DBranchV2`
in `modeling/correlation.py`, `modeling/stage2.py`) and the Stage 2 recipe
(`training/train_stage2.py`) were ported from an earlier, unpublished
implementation of FS-DINO Stage 2; the differences from the paper listed below
are those of that implementation, kept because the reported checkpoints depend
on them. `HQMaskDecoder`
(`src/fs_dino_hq/modeling/hq_decoder.py`) adapts code from SAM's
`MaskDecoder` and HQ-SAM's `MaskDecoderHQ`; see `THIRD_PARTY_NOTICES.md`.

## Pipeline and checkpoint dependencies

```
Stage 1   distill --adapter final  -> final_adapter_best.pt  (student of SAM's final image embedding)
          distill --adapter early  -> early_adapter_best.pt  (student of SAM block-7 features, PCA to 256 channels)
Stage 2   train-stage2  (init: final adapter, SAM ViT-H mask decoder)
                        -> stage2_*.pt candidates
          select-stage2 -> stage2_selected.pt (+ stage2_selected.json, stage2_selection_report.json)
HQ        train-hq      (input: stage2_selected.pt, early_adapter_best.pt, SAM-HQ ViT-H weights)
                        -> hq_best.pt
Inference predict       (input: hq_best.pt and the DINOv2 weights only)
```

An HQ checkpoint stores every tensor of the model except DINOv2: the Stage 2
weights (copied unchanged), the HQ modules and the frozen early adapter, so
inference needs no other file from training. Stage 2 uses only the final
adapter (step 1a); HQ training uses only the early adapter (step 1b) and the
selected Stage 2 checkpoint. The two distillation runs are independent and the
code does not check how they relate; `train-hq` records the SHA-256 of both
inputs. The reported results use a final and an early adapter distilled with
the configs in `configs/` on the same unlabelled split and the same DINOv2
weights; other combinations have not been tested.

## Stage 1: adapter distillation

* Student: the bottleneck adapter (1x1 conv 768->256, 3x3 conv, 1x1 conv
  256->128, two pre-norm transformer layers at 128 channels, 1x1 conv
  128->256) on the raw output of DINOv2 block index 2 (the third block,
  without the final LayerNorm). FS-DINO also distils from the third-layer
  features.
* Final adapter: the teacher is SAM ViT-H's final image embedding
  (256 x 64 x 64), as in FS-DINO.
* Early adapter (an extension of this project): the teacher is the output of
  SAM ViT-H block 7, the first global-attention block, which HQ-SAM uses as
  its early-layer feature; it is reduced from 1280 to 256 channels by a PCA
  fitted on 400 training images, so both adapters share one architecture.
* Coarse-to-fine schedule: the student sees the image at 126, 252 and
  518 pixels (8, 8 and 14 epochs); the adapter weights carry over between
  stages; the teacher always sees 1024 pixels and the student output is
  bilinearly upsampled to the 64 x 64 teacher grid. Loss: plain MSE on raw
  features, as in FS-DINO.
* Differences from FS-DINO's distillation setup: unlabelled construction-site
  photographs instead of 1 % of SA-1B, 8/8/14 instead of 15 epochs per stage,
  batch 4, AdamW (lr 5e-4, weight decay 1e-4) with a one-epoch warm-up and a
  cosine schedule restarted at every stage, letterbox views for student and
  teacher.
* Checkpoint selection: `*_best.pt` has the lowest MSE on the held-out split
  at 518 pixels across all stages.

## Stage 2: episodic training

* Trained: final adapter (initialised from Stage 1), MVPG (semantic prompt
  branch, prototype-cosine map, 4D correlation branch) and the SAM mask
  decoder (initialised from SAM ViT-H). Frozen: DINOv2 and the SAM prompt
  encoder. The SAM image encoder is not used. Training the adapter together
  with the MVPG and the decoder follows the first arXiv version of FS-DINO;
  later versions keep the adapter frozen in this stage.
* Loss: dice + BCE on the 37 x 37 prior plus dice + BCE on the 256 x 256
  decoder logits, query mask resized with nearest neighbour to each grid.
  AdamW (lr 1e-4, weight decay 1e-4), batch 2, no learning-rate schedule.
* Episodes come from fixed, seeded plans on a scene-disjoint split (see
  `docs/data.md`). With K > 1 supports the prompts of the K supports are
  averaged (semantic prompt, dense prompt, prior logits) and the decoder runs
  once, the same rule used at HQ training and inference
  (`fs_dino_hq/modeling/kshot.py`).
* Differences from FS-DINO:
  * the decoder uses SAM's frozen dense positional encoding (the paper
    describes a learnable Gaussian positional encoding);
  * the support prototype is the masked average over foreground support
    features (the paper's Eq. 2 averages the whole support feature map);
  * 4D correlation (`Correlation4DBranchV2`): the paper reshapes the
    37^2 x 37^2 cosine volume so that support tokens are channels and applies
    the 1x1 (1369 -> 512), stride-2 3x3 (512 -> 361) and depthwise
    convolutions over the query plane. This implementation folds the query
    tokens into channels instead and applies convolutions with the same kernel
    shapes over the support plane, so its 19 x 19 "query grid" is a learned
    ordering of query tokens rather than a geometric downsampling. The reduced
    volume is averaged over the support dimensions before the prototype-cosine
    map is added (the paper adds it per channel to the 4D-derived volume).
    The Stage 2 checkpoints depend on this layout, so it is kept as trained.

### Adapter convention

The two transformer layers of the adapter are pre-norm without dropout in the
distillation student. The Stage 2 and HQ models build the adapter with the
same convention (`norm_first=True`, dropout 0) and record it in every
checkpoint as `adapter_norm_first`; loaders build the adapter from that flag
and the strict loader refuses a mismatch. Loading the same weights into a
post-norm layer (PyTorch's default) computes a different function.

### Checkpoint selection

During training, full validation runs every 5 epochs, so a checkpoint saved
by another rule (best fast validation, periodic snapshot, last epoch) may never
have been scored on the full validation plan. `select-stage2` therefore scores
every saved candidate (deduplicated by a hash of its weights) on the complete
selection plan and picks the lowest total validation loss (prior + decoder;
ties: lower decoder loss, higher IoU, lower epoch). The result is the best
among the saved checkpoints, not among all epochs. `train-hq` refuses a Stage 2
input that does not come with an OK selection report and matching hashes,
unless `--allow-unselected-stage2` is given (recorded in the run).

## HQ branch

* Trained: only the HQ modules (`hf_token`, `hf_mlp`, `adapter_proj`,
  `early_proj`, `decoder_mask_proj`, `dino_l3_proj`), 1,145,056 parameters.
  Everything from Stage 2 is frozen. `hf_token`, `hf_mlp`, `adapter_proj` and
  `decoder_mask_proj` are initialised from `sam_hq_vit_h.pth` (HQ-SAM's
  `embedding_encoder` and `embedding_maskfeature` have the same shapes);
  `early_proj` and `dino_l3_proj` start from random initialisation.
* Feature sources differ from HQ-SAM: the early/local feature is the
  distilled early adapter's output projected by `early_proj`, the global
  feature is the Stage 2 adapter embedding projected by `adapter_proj`, and
  the decoder feature is projected by `decoder_mask_proj`; the three are
  summed. `dino_l3_proj` (a projection of raw DINOv2 block-3 features) is
  constructed, stored and counted as trainable but receives no gradient in
  this configuration; it is kept for checkpoint compatibility, so 325,056
  parameters are effectively trained.
* Two decoder passes (`predict_dual_canonical`): a pass without the HQ token
  gives the canonical Stage 2 output ("base"); a second pass with the HQ token
  gives the HQ residual. The output is `corrected = base + hq` (residual
  scale fixed at 1). In HQ-SAM the HQ token shares the decoder pass with the
  SAM mask token; with the separate pass the Stage 2 output cannot change when
  the HQ modules are trained.
* Loss: dice + BCE on the corrected logits over the full 256 x 256 grid,
  AdamW (lr 1e-4, weight decay 1e-4), batch 2, five-shot episodes. HQ-SAM's
  training code computes its mask losses on sampled points; that variant is
  not part of this release.
* Checkpoint selection: lowest loss on the full validation plan, evaluated
  every 5 epochs.

## Preprocessing (all stages)

Images are letterboxed: the longest side is resized to 518 pixels with
bilinear interpolation (masks: nearest neighbour), and the result is pasted
at the top-left corner of a black 518 x 518 canvas before ImageNet
normalisation. Model outputs are 256 x 256 logits aligned with the padded
square; predictions are cropped to the content box and resized bilinearly to
the original image size. A model must be used with the preprocessing it was
trained with; the letterbox is not interchangeable with a square stretch.

## Reproducibility

Seeds control the episode plans, the order of episodes, DataLoader worker
seeding and the random initialisation of new modules. Each run's
`config.json` records the SHA-256 of its input files; the Stage 2 and HQ
trainers also record environment versions (Python, NumPy, PyTorch, CUDA, GPU
model, cuDNN flags) and per-file code hashes, and the HQ trainer records a
hash of the initial HQ state (`initial_hq_state_sha256`). GPU training is not bitwise
reproducible (cuDNN determinism is requested, but not all kernels have
deterministic implementations).

The random initialisation of new modules depends on the construction order
of the model, which is kept fixed: DINOv2 (from the `torch.hub` code), the
full SAM ViT-H model and then the project modules are built in the same order
as for the runs in `docs/results.md`, even where a component is unused. The
`torch.hub` DINOv2 code (pinned to the commit whose files are identical to the
code used for those runs), the `segment_anything` version (pinned) and the
PyTorch version also influence it.

## References

[1] M. Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193.
[2] A. Kirillov et al. Segment Anything. ICCV 2023. arXiv:2304.02643.
[3] L. Ke et al. Segment Anything in High Quality. NeurIPS 2023. arXiv:2306.01567.
[4] W. Zhuo, Z. Tang, W. Xue, H. Ding, L. Shen. DINOv2-powered Few-Shot Semantic Segmentation: A Unified Framework via Cross-Model Distillation and 4D Correlation Mining. arXiv:2504.15669v1, 2025.

Note on [4]: later arXiv versions of the paper were revised and retitled; the
published version is W. Zhuo, Z. Tang, W. Xue, H. Ding, J. Ji, L. Shen,
"UINO-FSS: Unifying Representation Learning and Few-Shot Segmentation via
Hierarchical Distillation and Mamba-HyperCorrelation", IEEE Transactions on
Image Processing, 2026 (doi:10.1109/TIP.2026.3723708). Several statements
above (the Stage 2 freezing, Eq. 2, the loss, the positional encoding and the
4D correlation) describe arXiv v1 and differ from the published version.
