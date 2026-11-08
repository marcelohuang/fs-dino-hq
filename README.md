# FS-DINO-HQ

Few-shot segmentation from a handful of annotated support images. A frozen
DINOv2 ViT-B/14 encodes the support and query images; an FS-DINO style meta
visual prompt generator turns the annotated supports into prompts for the SAM
mask decoder; an HQ-SAM style branch adds a residual correction to the
decoder output. The SAM image encoder is used only as a teacher to distil the
two adapters that map DINOv2 features into SAM's feature space. The code was
developed for segmenting rebar in construction-site photographs with five
annotated supports.

```
support images + masks ─┐
                        ├─ DINOv2 ─ prompt generator ─┐
query image ────────────┤                             ├─ SAM mask decoder ─ Stage 2 base ─┐
                        └─ DINOv2 ─ final adapter ────┘                                  ├─ corrected mask
                          DINOv2 ─ early adapter ─ HQ branch ─ residual ────────────────┘
```

Training has three stages: adapter distillation from SAM (Stage 1),
episodic training of the prompt generator, final adapter and mask decoder
(Stage 2), and training of the HQ branch with everything else frozen. See
[docs/implementation_notes.md](docs/implementation_notes.md) for the relation
to FS-DINO, SAM, HQ-SAM and DINOv2, and for the conventions checkpoints
depend on.

## Installation

Python 3.10 or newer and PyTorch 2.1 or newer (a CUDA GPU for training).

```bash
git clone <repository-url> fs-dino-hq
cd fs-dino-hq
pip install -e .            # add [plot] for loss curves, [dev] for the tests
```

`segment-anything` is installed from its GitHub repository at a pinned commit.
DINOv2 is built from its `torch.hub` definition at a pinned commit
(`fs_dino_hq/modeling/backbones.py`), downloaded once to `$TORCH_HOME/hub`; for
offline machines set `FS_DINO_HQ_DINOV2_REPO` to a local clone of
https://github.com/facebookresearch/dinov2 checked out at that commit. Tested with Python 3.11,
PyTorch 2.11, NumPy 2.2 and Pillow 11.3.

## Weights

Upstream weights (download separately):

| File | Source | Needed for |
|---|---|---|
| `dinov2_vitb14_pretrain.pth` | https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth | everything |
| `sam_vit_h_4b8939.pth` | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth | training (Stages 1, 2 and HQ) |
| `sam_hq_vit_h.pth` | https://huggingface.co/lkeab/hq-sam (file `sam_hq_vit_h.pth`) | HQ training |

Trained weights for the configurations in `configs/` are not distributed at
this time. Inference needs one trained HQ checkpoint (it contains the Stage 2
weights and the early adapter) and the DINOv2 weights.

## Inference

Python:

```python
from PIL import Image
from fs_dino_hq.inference import Predictor
from fs_dino_hq.data.labelme import labelme_json_to_mask

predictor = Predictor.from_checkpoint("runs/hq_five_shot/hq_best.pt",
                                      "weights/dinov2_vitb14_pretrain.pth")
supports = [f"data/supports/img_{i}.png" for i in (3, 17, 21, 40, 52)]
masks = [labelme_json_to_mask(s.replace(".png", ".json")) for s in supports]
result = predictor.predict(Image.open("data/queries/site_07.jpg"), supports, masks)
result.probability   # float32 [H, W], query resolution
result.mask          # uint8 [H, W], probability >= 0.5
```

Command line, for a folder of queries with one fixed support set:

```bash
fs-dino-hq predict \
    --hq-ckpt runs/hq_five_shot/hq_best.pt \
    --dino-ckpt weights/dinov2_vitb14_pretrain.pth \
    --support-dir data/supports/ --support-set data/supports/support_set.json \
    --query-dir data/queries/ --output-dir outputs/predictions/ --save-base
```

Inputs, preprocessing and outputs are described in [docs/data.md](docs/data.md)
and [docs/evaluation.md](docs/evaluation.md). `examples/synthetic_demo.py`
runs the full interface on generated images with a randomly initialised model
to show the call sequence (its outputs are meaningless).

## Training

No training or evaluation data is distributed. Stage 1 needs a pool of
unlabelled photographs; Stage 2 and the HQ branch need a LabelMe dataset in
class folders whose file names encode the scene ([docs/data.md](docs/data.md)).

```bash
export DATA_DIR=/path/to/data WEIGHTS_DIR=/path/to/weights OUTPUT_DIR=/path/to/runs
fs-dino-hq make-distill-split ...                   # distillation split, see docs/training.md
fs-dino-hq distill --config configs/distill_final_adapter.json
fs-dino-hq distill --config configs/distill_early_adapter.json
fs-dino-hq make-episodes ...                        # episode plans, see docs/training.md
fs-dino-hq train-stage2 --config configs/stage2_five_shot.json
fs-dino-hq select-stage2 --run-dir $OUTPUT_DIR/stage2_five_shot
fs-dino-hq train-hq --config configs/hq_five_shot.json
fs-dino-hq plot-history $OUTPUT_DIR/hq_five_shot
```

[docs/training.md](docs/training.md) gives the data layout, every stage's
inputs and outputs, checkpoint selection, resuming, and the expected run times.

## Evaluation

```bash
fs-dino-hq evaluate --pred-dir outputs/predictions/pred_mask --gt-dir data/queries/ \
    --out-csv outputs/predictions/metrics.csv --out-json outputs/predictions/metrics.json
fs-dino-hq evaluate --pred-dir outputs/predictions/base_mask --gt-dir data/queries/ \
    --out-csv outputs/predictions/metrics_base.csv --out-json outputs/predictions/metrics_base.json
fs-dino-hq compare --base outputs/predictions/metrics_base.csv --new outputs/predictions/metrics.csv
```

Metrics (IoU, Dice, boundary F, thin-structure recall, fragmentation) and the
paired bootstrap are defined in [docs/evaluation.md](docs/evaluation.md).

## Results

On 46 annotated construction-site photographs with five fixed supports, the
released configuration reached a mean IoU of 0.409 (0.405 without the HQ
branch). These numbers come from single training runs evaluated on a private,
reused evaluation set with partial annotations;
[docs/results.md](docs/results.md) gives the full tables, the limitations and
what was re-verified with this code.

## Repository layout

```
src/fs_dino_hq/
  modeling/     DINOv2/SAM loading, adapters, prompt generator, HQ decoder, K-shot aggregation
  data/         letterbox preprocessing, LabelMe masks, episode plans and datasets
  training/     distillation, Stage 2, checkpoint selection, HQ training, history plots
  inference/    Predictor API, predict command, visualisations
  evaluation/   metrics, evaluate and compare commands
configs/        configurations of the reported runs
docs/           data, training, evaluation, results, implementation notes
examples/       synthetic end-to-end example
tests/          CPU tests (stub backbones, synthetic fixtures)
tools/slurm/    optional job-script template
```

## License

This repository does not yet carry a license.
`src/fs_dino_hq/modeling/hq_decoder.py` contains code adapted from Segment
Anything and HQ-SAM under the Apache License 2.0; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgements

This work builds on DINOv2 (Oquab et al.), Segment Anything (Kirillov et al.),
HQ-SAM (Ke et al.) and FS-DINO (Zhuo et al.; published as UINO-FSS); references
are listed in [docs/implementation_notes.md](docs/implementation_notes.md).
The Stage 2 model and training recipe were ported from an earlier, unpublished
implementation of FS-DINO Stage 2.
