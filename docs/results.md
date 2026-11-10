# Results and their scope

All numbers below come from single training runs of the configurations in
`configs/` (seed 42) and from one fixed evaluation protocol. They are
reported to document what this code produced; they are not benchmark
claims. The data behind them are not distributed with this repository, so the
real-photo numbers cannot be reproduced without access to the same images.

## Evaluation protocol

* Queries: 46 construction-site photographs of rebar (40 phone images at
  1280 x 720, 6 action-camera images at 5120 x 2880 or 5171 x 3878) with
  LabelMe polygon annotations (labels `spacing` and `intersection`, merged
  into one foreground class).
* Supports: the same 5 annotated photographs, in a fixed order, for every
  query (5-shot); no query is used as a support.
* Output: the corrected prediction at threshold 0.5, at the query's native
  resolution. "Stage 2 base" is the same model's output without the HQ
  branch (`base_prob/`, `base_mask/` from `fs-dino-hq predict --save-base`).
* Metrics: `fs_dino_hq.evaluation.metrics` (boundary tolerance 5 px, thin
  width 12 px); means and medians over the 46 images.
* Paired comparisons: per-query differences, 10,000 bootstrap resamples over
  queries, percentile 95 % interval (`fs-dino-hq compare`, seed 42). The
  interval reflects query sampling for the given checkpoints, not variation
  between training runs, which was not measured.

## Limitations of this evaluation

* One reused evaluation set. The same 46 queries were used to compare the
  variants explored during development, including the choice of the five-shot
  Stage 2 configuration reported here. Results on this set are therefore
  optimistic as an estimate for new images.
* Partial annotations. Bars that are visible but not annotated count as
  background, so correct predictions on them are counted as false positives.
* Stage 1 split isolation. The unlabelled photographs used to distil the
  adapters were split so that no evaluation image (and no exact duplicate) is
  in the distillation set, but 48 of the 5,796 distillation training frames
  come from the capture sequences of evaluation images (7 sequences; 9 of
  these frames are near duplicates of an evaluation image). Distillation uses
  no labels; the effect on the reported numbers was not measured.
  `fs-dino-hq make-distill-split --exclude-sequences-of-excluded` produces the
  stricter split (no such frames); it was not used for the runs below.
* Training data. Stage 2 and HQ were trained on synthetic images only; all
  real-photo numbers are therefore transfer results.
* Single runs. No seed replication; GPU training is not bitwise
  reproducible. Differences between two configurations below are differences
  between two single runs; whether they exceed the variation between training
  seeds was not measured.
* One support set. Every result uses the same five supports; the sensitivity
  to the choice of supports was not measured.
* Dependence on the supports. In every training episode the supports and the
  query show the same kind of object, so training never requires the supports
  to decide what the foreground is, and how much the output of the trained
  model changes with the supports has not been measured. On the synthetic
  validation set the Stage 2 model scored nearly the same with one support as
  with five (decoder IoU 0.790 and 0.791 at the last epoch).
* Checkpoint choice. The HQ checkpoint is the one with the lowest validation
  loss among ten validated epochs. That loss compares the logits with the
  query mask resized to 256 x 256 by nearest-neighbour index, which places the
  target about half a cell off the logit grid. Ranking the saved checkpoints
  of the reported run by IoU at native resolution on the same validation
  episodes prefers epoch 40 instead of 45; on the 46 queries epoch 40 reaches
  an IoU of 0.4160, 0.0011 [0.0004, 0.0018] below epoch 45.

## Reported configuration (five-shot Stage 2, HQ branch with the point-sampled loss)

Training-side selections, made on held-out validation data (never on the 46
evaluation queries): 395 unlabelled construction-site photographs for Stage 1,
1,421 synthetic images from 9 held-out scenes for Stage 2 and HQ:

| Stage | Selected checkpoint | Selection value |
|---|---|---|
| Stage 1, final adapter | stage 518 px, epoch 14 of 14 (30 overall) | held-out MSE 0.00323, cosine 0.820 to the SAM embedding (395 images, 518 px) |
| Stage 1, early adapter | stage 518 px, epoch 14 of 14 (30 overall) | cosine 0.855 to the PCA-reduced SAM block-7 feature (395 images, 518 px) |
| Stage 2, five-shot | epoch 183 of 200 (best of 22 distinct saved states) | five-shot validation loss 1.2001 (prior 0.9711 + decoder 0.2289), decoder IoU 0.792, 1,421 episodes |
| HQ (`configs/hq_five_shot.json`) | epoch 45 of 50 (45,000 optimizer updates) | lowest full-validation loss, 0.2794 (dice + BCE on the 256 x 256 grid, 1,421 episodes; validated every 5 epochs) |

Real-photo evaluation (46 queries, 5 supports):

| Output | IoU | Dice | Boundary F | Thin recall | Fragmentation (lower is better) |
|---|---|---|---|---|---|
| Corrected (model output) | 0.4171 (median 0.4397) | 0.5812 | 0.4724 | 0.6245 | 22.46 (median 19.75) |
| Stage 2 base (no HQ branch) | 0.4050 (median 0.4235) | 0.5687 | 0.4600 | 0.5870 | 21.13 (median 18.0) |

Effect of the HQ branch (corrected minus Stage 2 base, same checkpoint, paired over the 46 queries):

| Metric | Difference | 95 % interval | Queries better / equal / worse |
|---|---|---|---|
| IoU | +0.0120 | [+0.0074, +0.0173] | 34 / 0 / 12 |
| Dice | +0.0125 | [+0.0075, +0.0179] | 34 / 0 / 12 |
| Boundary F | +0.0124 | [+0.0055, +0.0193] | 32 / 0 / 14 |
| Thin recall | +0.0376 | [+0.0306, +0.0450] | 44 / 0 / 2 |
| Fragmentation | +1.33 | [-0.46, +3.15] | 19 / 1 / 26 |

Most of the performance comes from the Stage 2 model; the HQ branch adds
about 0.012 IoU, mainly by recovering thin structures.

## Effect of the HQ training loss

Control run: the same Stage 2 checkpoint, early adapter, initial HQ weights,
episodes, seed and budget, trained with dice + BCE on the 256 x 256 logit grid
(`--loss full_grid`) instead of the point-sampled loss. Its selected
checkpoint is epoch 40 (full-validation loss 0.2171).

| HQ training loss | IoU | Dice | Boundary F | Thin recall | Fragmentation |
|---|---|---|---|---|---|
| Point-sampled, 1024 x 1024 target (reported) | 0.4171 | 0.5812 | 0.4724 | 0.6245 | 22.46 |
| Full grid, 256 x 256 target (control) | 0.4086 | 0.5725 | 0.4630 | 0.5993 | 18.14 |
| Difference [95 % interval], better / equal / worse | +0.0084 [+0.0036, +0.0137], 31 / 0 / 15 | +0.0087 [+0.0037, +0.0141], 31 / 0 / 15 | +0.0094 [+0.0020, +0.0169], 25 / 0 / 21 | +0.0252 [+0.0183, +0.0323], 39 / 0 / 7 | +4.31 [+2.57, +6.03], 6 / 2 / 38 |

The two losses differ in more than the sampling of points: the point-sampled
loss reads a 1024 x 1024 target at the positions at which it reads the
logits, whereas the full-grid loss uses a 256 x 256 target that is coarser
and displaced by about half a cell relative to the logit grid (see
[implementation_notes.md](implementation_notes.md#hq-branch)); the Dice
smoothing constants also differ. The gain can therefore not be attributed to
point sampling.

Most of it appears to come from the alignment of the target. Translating the
saved probability maps by half a logit cell towards the upper left, the
correction that the displaced target predicts, and thresholding them again
gives (same 46 queries; a test on saved outputs, no retraining):

| Output | IoU as predicted | IoU translated by -0.5 cell | Difference [95 % interval] |
|---|---|---|---|
| Full-grid control | 0.4086 | 0.4152 | +0.0066 [+0.0010, +0.0128] |
| Point-sampled (reported) | 0.4171 | 0.3989 | -0.0181 [-0.0232, -0.0130] |
| Stage 2 base | 0.4050 | 0.4089 | +0.0039 [-0.0012, +0.0096] |

The reported model is centred (any translation lowers its IoU), whereas the
control and the Stage 2 base, both trained on the displaced 256 x 256
target, improve when moved back. Against the translated control the reported
model leads by +0.0018 [0.0000, +0.0036] instead of +0.0084. A control
trained with a centred full-grid target would be the direct test; it has not
been run.

The reported model also recovers more of the annotated bars (higher recall
on thin structures) and produces more small separate components (higher
fragmentation). The validation losses of the two runs are on the same scale
but do not rank the two losses: the control is trained on the validation
target, the reported model on the 1024 x 1024 one.

## Five-shot versus one-shot Stage 2 training

Two Stage 2 runs differing only in the number of supports per training
episode (`configs/stage2_five_shot.json`, `configs/stage2_one_shot.json`;
same queries, budget and initialisation), each followed by its own HQ run
with the full-grid HQ loss (`--loss full_grid`), the configuration in use
when this comparison was made. To compare at matched selection
opportunity, each run used the checkpoint with the lowest five-shot validation
loss among the epochs saved by both runs (50, 100, 150, 200): epoch 200 for
five-shot training (loss 1.2181) and epoch 100 for one-shot training
(loss 1.2968). These are not the checkpoints of
the reported configuration (five-shot epoch 183 is selected among all its
saved states). Real-photo evaluation, five-shot minus one-shot, 46 queries:

| Metric | Stage 2 base: one-shot / five-shot | Difference [95 % interval] | Corrected: one-shot / five-shot | Difference [95 % interval] |
|---|---|---|---|---|
| IoU | 0.3367 / 0.3836 | +0.0470 [+0.0281, +0.0666] | 0.3735 / 0.4006 | +0.0270 [+0.0131, +0.0419] |
| Dice | 0.4941 / 0.5448 | +0.0507 [+0.0297, +0.0730] | 0.5350 / 0.5632 | +0.0282 [+0.0129, +0.0458] |
| Boundary F | 0.4179 / 0.4385 | +0.0205 [+0.0015, +0.0405] | 0.4320 / 0.4525 | +0.0205 [+0.0046, +0.0376] |
| Thin recall | 0.4276 / 0.5299 | +0.1023 [+0.0682, +0.1362] | 0.5346 / 0.5939 | +0.0593 [+0.0273, +0.0935] |
| Fragmentation | 25.80 / 19.59 | -6.21 [-9.18, -3.57] | 24.10 / 15.57 | -8.53 [-12.26, -5.01] |

To reproduce the comparison:

1. Score the one-shot run's saved checkpoints on the five-shot validation plan
   (training validates that run on the one-shot plan only):
   `fs-dino-hq select-stage2 --run-dir $OUTPUT_DIR/stage2_one_shot --diagnostic-plans val5=$DATA_DIR/episodes/val_episode_plan_5shot.csv:5`.
   The five-shot losses are `candidates[].diagnostics.val5.loss` in that run's
   `stage2_selection_report.json`; for the five-shot run they are
   `candidates[].loss` in its own report. Take the lowest among epochs 50,
   100, 150 and 200 of each run.
2. Train the HQ branch on each matched checkpoint, which is not a selection
   artifact:
   `fs-dino-hq train-hq --config configs/hq_five_shot.json --loss full_grid --stage2-ckpt $OUTPUT_DIR/stage2_one_shot/stage2_epoch_100.pt --allow-unselected-stage2 --save-dir $OUTPUT_DIR/hq_one_shot_e100`
   (and likewise `stage2_five_shot/stage2_epoch_200.pt`).
3. Predict and evaluate both as in `docs/evaluation.md` (with `--save-base`)
   and compare the per-image metric files with `fs-dino-hq compare`.

## Verification

This code is a restructured version of the research code that produced the
numbers above. The following checks compared the two on the same inputs (one
NVIDIA L40S or CPU; PyTorch 2.11):

| Check | Result |
|---|---|
| `fs-dino-hq predict --save-base` with the reported HQ checkpoint on the 46 evaluation queries (GPU) | output probability maps and masks, corrected and Stage 2 base, pixel-identical to the research code's outputs for all 46 queries |
| `fs-dino-hq evaluate` on those masks | every per-image metric identical; means and medians as in the tables above |
| `fs-dino-hq compare` (seed 42, 10,000 resamples): corrected vs Stage 2 base, and reported model vs full-grid control | differences and 95 % intervals identical to the tables |
| One HQ training step with the point-sampled loss from the reported weights, same episodes and point-generator state | identical inputs, logits, sampled points, point labels and loss terms; gradients of all trained tensors identical on CPU, equal to 2e-5 (relative) between two GPU processes |
| `fs-dino-hq train-hq` for three short epochs from the inputs of the reported run, with each loss | the initial HQ state of the reported run and its control (hash in `implementation_notes.md`); per-update losses, validation losses and every saved checkpoint tensor identical on CPU, within 3e-6 on GPU; on GPU the first updates of the point-sampled run equal those recorded for the reported run |
| `fs-dino-hq select-stage2` re-scoring two saved Stage 2 checkpoints (epochs 170 and 183) on the full validation plans (GPU) | validation losses identical to the recorded selection; epoch 183 selected |
| `fs-dino-hq train-stage2` (five-shot) for 2 short epochs on real data (CPU) | loss histories and every saved checkpoint tensor identical |
| `fs-dino-hq make-episodes` and `fs-dino-hq distill` (checked for the first version of this code; these commands have not changed since) | manifests and plans byte-identical (SHA-256 in `docs/data.md`); loss histories and checkpoint tensors of 2 short epochs identical on CPU |
| Unit tests (CPU, stub backbones, synthetic fixtures) | pass |

Not verified: complete training runs with this code (the reported numbers come
from the research code; full GPU runs are not bitwise reproducible), and other
GPU types or software versions.
