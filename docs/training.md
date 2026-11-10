# Training

The full pipeline has five steps. Each produces the input of the next; the
checkpoint dependencies are summarised in
[implementation_notes.md](implementation_notes.md#pipeline-and-checkpoint-dependencies).

| Step | Command | Config | Output used later |
|---|---|---|---|
| 1a | `fs-dino-hq distill` | `configs/distill_final_adapter.json` | `final_adapter_best.pt` |
| 1b | `fs-dino-hq distill` | `configs/distill_early_adapter.json` | `early_adapter_best.pt` |
| 2 | `fs-dino-hq train-stage2` | `configs/stage2_five_shot.json` | `stage2_*.pt` candidates |
| 3 | `fs-dino-hq select-stage2` | (reads the Stage 2 run) | `stage2_selected.pt` |
| 4 | `fs-dino-hq train-hq` | `configs/hq_five_shot.json` | `hq_best.pt` (the model used by `predict`) |

Every command prints its options with `--help`. Values in a `--config` file
are defaults; options given on the command line override them.

## Directory layout used by the configs

The configs refer to three environment variables:

```
$WEIGHTS_DIR/  dinov2_vitb14_pretrain.pth  sam_vit_h_4b8939.pth  sam_hq_vit_h.pth
$DATA_DIR/
  unlabelled/images/          unlabelled photographs for Stage 1
  unlabelled/split/           train.txt, val.txt (fs-dino-hq make-distill-split)
  synthetic/{beams,columns,slabs,walls}/   LabelMe dataset for Stage 2 and HQ
  episodes/                   plans and manifests (fs-dino-hq make-episodes)
$OUTPUT_DIR/                  one directory per run (created by the commands)
```

```bash
export WEIGHTS_DIR=/path/to/weights DATA_DIR=/path/to/data OUTPUT_DIR=/path/to/runs
```

The formats are described in [data.md](data.md). A different layout only
needs edited configs or command-line overrides.

## Data preparation

Distillation split (images only; exclude every image used for evaluation;
`all_images.txt` lists every unlabelled image, see [data.md](data.md#unlabelled-images-for-stage-1-distillation)):

```bash
fs-dino-hq make-distill-split --manifest all_images.txt --image-root $DATA_DIR/unlabelled/images \
    --exclude-dir /path/to/eval_queries --exclude-dir /path/to/eval_supports \
    --exclude-sequences-of-excluded --out-dir $DATA_DIR/unlabelled/split
```

Episode plans (CPU, a few minutes; the four runs share one split):

```bash
P="--synthetic-root $DATA_DIR/synthetic --out-dir $DATA_DIR/episodes --seed 42 --val-frac 0.05"
fs-dino-hq make-episodes $P --n-shot 1 --cycles 8 --suffix _1shot
fs-dino-hq make-episodes $P --n-shot 5 --cycles 2 --suffix _5shot --no-verify-images
fs-dino-hq make-episodes $P --n-shot 5 --cycles 8 --suffix _5shot_8cyc --no-verify-images
```

`docs/data.md` lists the SHA-256 of the plans used for the reported runs.
`make-episodes` refuses to write into a directory whose manifests describe a
different split.

## Step 1: adapter distillation

```bash
fs-dino-hq distill --config configs/distill_final_adapter.json
fs-dino-hq distill --config configs/distill_early_adapter.json
```

The two runs are independent and can run in parallel on two GPUs. Each trains
at 126, 252 and 518 pixels (8, 8 and 14 epochs; `--stages`), with the SAM
ViT-H teacher at 1024 pixels.

* Outputs: `history.csv` (one row per epoch plus one per stage start),
  `<adapter>_stage<res>_{best,last}.pt`, `<adapter>_best.pt` (lowest held-out
  MSE at 518 pixels over all stages), `c2f_last.pt` (resume state),
  `results.json` (scores of the selected and final-stage checkpoints),
  `config.json`; the early adapter also writes `pca_teacher_1280to256.pt`.
* `--resume` continues from `c2f_last.pt` at the next epoch of the same stage;
  without it, a run directory that already contains `c2f_last.pt` is refused.
* `--smoke` runs every stage for one epoch on 8 training and 4 validation
  images. Use a separate directory, for example
  `fs-dino-hq distill --config configs/distill_early_adapter.json --smoke --save-dir $OUTPUT_DIR/smoke_distill_early`.
* The early adapter's PCA basis is fitted on 400 training images with a GPU
  eigendecomposition, which is not guaranteed to give the same basis on
  another machine. To continue or repeat a run with an existing basis, pass
  `--pca-cache <run>/pca_teacher_1280to256.pt`.
* Run time on one NVIDIA L40S: about 32 minutes per epoch (dominated by the
  1024-pixel SAM teacher), about 16 hours per adapter.

## Step 2: Stage 2

```bash
fs-dino-hq train-stage2 --config configs/stage2_five_shot.json
```

* Inputs: the final adapter from step 1a, the SAM ViT-H mask decoder, the
  episode plans.
* Each epoch trains on 2,000 episodes in plan order and validates on the first
  300 episodes of the selection plan ("fast"); every 5 epochs and at the end
  it validates on all 1,421 episodes of every plan in `val_plans` ("full").
* Outputs: `history_fast.csv`, `history_full.csv`, `stage2_epoch_<E>.pt`
  (every `save_every` epochs and `save_epochs`), `stage2_best_full_val.pt`,
  `stage2_best_fast_val.pt`, `stage2_best.pt`, `stage2_last.pt` (optimizer,
  random-number state and histories for `--resume`), `config.json`,
  `freeze_audit.json`, `adapter_init_report.json`, `frozen_check.json`.
* `--resume` continues from `stage2_last.pt`; a run directory that already
  contains `stage2_last.pt` is refused without it.
* `--smoke` runs one epoch of 4 training episodes (unless `--epochs` is given)
  and checks preprocessing, gradient paths and (for K > 1) the K-shot
  aggregation; the full validation of each epoch still covers every validation
  plan (about 2,800 episodes for the five-shot config). Use a separate
  directory, for example
  `fs-dino-hq train-stage2 --config configs/stage2_five_shot.json --smoke --save-dir $OUTPUT_DIR/smoke_stage2`.
* `configs/stage2_one_shot.json` trains the one-shot variant compared in
  `docs/results.md`.
* Run time on one L40S: about 44 hours for the five-shot configuration
  including validation, about 15 hours for the one-shot configuration.

## Step 3: Stage 2 checkpoint selection

```bash
fs-dino-hq select-stage2 --run-dir $OUTPUT_DIR/stage2_five_shot
```

Scores every saved `stage2_*.pt` (deduplicated by a hash of the weights) on
the complete selection plan (`select_on` of the run, here `val5`) and on the
other validation plans as diagnostics, and selects the lowest total loss.
Outputs in the run directory: `stage2_selected.pt` (copy of the winner),
`stage2_selected.json`, `stage2_selection_report.json` (all candidates and
their scores, status `OK` or `FAILED`) and `stage2_selection_cache.json`.
Scores are cached per candidate, so an interrupted selection resumes; a rerun
only recomputes what is missing. The dataset root, the validation plans and the
backbone checkpoints are read from the run's `config.json`; if they have moved,
pass `--synthetic-root`, `--dino-ckpt`, `--sam-ckpt` and the plans
(`--select-plan val5=<path>:5 --diagnostic-plans val1=<path>:1`). Run time: about
8 minutes per candidate for the five-shot plan on one L40S (about 3 to 4 hours
for the 22 distinct states of the five-shot run).

## Step 4: HQ branch

```bash
fs-dino-hq train-hq --config configs/hq_five_shot.json
```

* Inputs: `--stage2-run` (uses `stage2_selected.pt` after checking the
  selection report and hashes), the early adapter from step 1b, the SAM ViT-H
  and HQ-SAM ViT-H weights. Any other Stage 2 checkpoint needs
  `--stage2-ckpt FILE --allow-unselected-stage2` (on the command line it
  replaces the config's `stage2_run`) and is recorded as such in
  `stage2_input_provenance.json`.
* 50 epochs of 2,000 five-shot episodes at batch size 2: 1,000 optimizer
  updates per epoch, 50,000 in total. Full validation every 5 epochs;
  `hq_best.pt` is the checkpoint with the lowest full-validation loss
  (`checkpoint_selection_report.json`).
* Training loss: BCE + Dice at sampled points against the 1024 x 1024 mask
  (`"loss": "point_sampled"` in the config). `--loss full_grid` trains with
  dice + BCE on the 256 x 256 grid instead, the control of `docs/results.md`.
  Validation uses the full-grid loss in both cases, so the training and
  validation curves of a point-sampled run are on different scales.
* Outputs: `loss_history_fast.csv` (per epoch: cumulative optimizer updates,
  training loss and its BCE and Dice terms, fast-validation loss),
  `loss_history_full.csv`, `train_steps.csv` (one row per optimizer update),
  `hq_epoch_<E>.pt` for every full-validation epoch, `hq_best_full_val.pt`,
  `hq_best_fast_val.pt`, `hq_last.pt`, `hq_best.pt`, and the provenance files
  listed by `fs-dino-hq train-hq --help`. `config.json` records the hashes of
  all input files and the hash of the initial HQ state.
* `--smoke` runs one training step and checks that only HQ parameters change.
* Run time on one L40S: about 11 hours.

The HQ trainer has no `--resume`; a run directory that already contains
`hq_best_full_val.pt` is refused.

## Training curves

```bash
pip install -e ".[plot]"
fs-dino-hq plot-history $OUTPUT_DIR/stage2_five_shot   # also: distill and HQ run directories
```

Writes `<run>.png` (optionally `--format png pdf`) and a `<run>_series.csv`
into the run directory (or `--out`), with the selected epoch marked. The
horizontal axis is the epoch; for HQ runs its label gives the number of
optimizer updates per epoch, and the training loss is drawn with its BCE and
Dice terms in a panel of its own.

## Hardware and software

The reported runs used one NVIDIA L40S (48 GB) per run, 6 to 7 data-loading
workers and 48 GB of host memory, with Python 3.11, PyTorch 2.11 (CUDA 12.8),
NumPy 2.2 and Pillow 11.3. Training is not bitwise reproducible on GPU; the
seeds fix the episode order, data-loader seeding and the initialisation of new
modules (see [implementation_notes.md](implementation_notes.md#reproducibility)).

`tools/slurm/train_template.sh` is a generic single-GPU Slurm template.
