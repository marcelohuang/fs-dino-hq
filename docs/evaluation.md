# Inference and evaluation

## Prediction

`fs-dino-hq predict` (or `fs_dino_hq.inference.Predictor`) runs one forward
pass per query with a fixed list of K annotated supports.

Processing, identical to training:

1. Every image is letterboxed to 518 x 518: the longest side is resized to 518
   pixels (bilinear), the image is pasted at the top-left corner of a black
   canvas and normalised with the ImageNet mean and standard deviation.
   Support masks are letterboxed with nearest-neighbour resizing.
2. For each support, the prompt generator computes a semantic prompt, a
   prototype-cosine map and the 4D-correlation prior and dense prompt against
   the query. The prompts of the K supports are averaged; the mask decoder runs
   once.
3. Two decoder passes give the Stage 2 base logits (no HQ token) and the HQ
   residual logits; the output logits are their sum (256 x 256, aligned with
   the padded square).
4. The sigmoid probability map is cropped to the letterbox content box,
   converted to 8 bits and resized bilinearly to the query size, so
   probabilities are multiples of 1/255. The mask is `probability >= 0.5`
   (`--threshold`).

The order of the supports affects the result only through floating-point
summation order. Use the same `--support-set` file to reproduce a run.

Output files (`--output-dir`):

| Path | Content |
|---|---|
| `pred_prob/<stem>.png` | 8-bit probability, value = floor(255 x p) |
| `pred_mask/<stem>.png` | mask, 0 or 255 |
| `base_prob/`, `base_mask/` | Stage 2 base output (`--save-base`), same encoding |
| `visualizations/` | overlays, side-by-side panels and a support sheet (`--save-visualizations`) |
| `predictions.csv` | per query: foreground fraction; IoU and Dice of the output and of the base when the query has an annotation |
| `inference_config.json` | arguments, supports, checkpoint SHA-256, checkpoint metadata, package version |

An output directory that already holds predictions is refused unless
`--overwrite` is given, which removes the earlier outputs first. Images are
read with their EXIF orientation applied, the frame in which LabelMe shows and
annotates them; a query whose annotation is empty gets no IoU/Dice.

The HQ residual alone is not a segmentation and is not written; it is
available as `Prediction.logits["hq_logits"]` for analysis.

## Metrics

`fs-dino-hq evaluate --pred-dir <masks> --gt-dir <annotations>` compares every
`<stem>.png` with `<stem>.json` at the annotation's resolution (predictions of
another size are resized with nearest neighbour). Predictions are read as 8-bit
grayscale and a pixel value above 127 is foreground: use 0/255 masks such as
`pred_mask/` from `predict` (a 0/1 mask is refused with an error).
Images without an annotation file, or with an empty annotation, are skipped
and listed in the summary.

| Metric | Definition |
|---|---|
| IoU, Dice | area overlap of the binary masks; 0 when both are empty |
| Boundary precision / recall / F | a boundary pixel is a mask pixel with a background 4-neighbour (pixels on the image border count as boundary); precision is the share of predicted boundary pixels within 5 px (Euclidean, `--tolerance`) of the annotated boundary, recall the converse, F their harmonic mean; 0 if either boundary is empty |
| Thin recall | recall over annotated pixels whose distance to the nearest background pixel is at most 6 px (`--thin-width 12` / 2): all pixels of bars up to about 12 px wide plus the outer 6 px of wider structures; NaN if there are none |
| Fragmentation | number of 8-connected predicted components divided by the number of annotated components (lower is better; an empty prediction scores 0); NaN if the annotation has no component |

Outputs: a per-image CSV and a JSON summary with the mean and median of each
metric over the evaluated images (NaN values excluded per metric), written to
`<pred-dir>/metrics.csv` and `<pred-dir>/metrics.json` unless `--out-csv` and
`--out-json` are given.

## Comparing two evaluations

`fs-dino-hq compare --base A/metrics.csv --new B/metrics.csv` pairs images by
id, computes per-image differences (new - base), and reports for each metric
the mean difference, a 95 % percentile interval from 10,000 bootstrap
resamples over images (seed 42, `numpy.random.default_rng`), and the counts of
images where the new evaluation is better, equal or worse (for fragmentation,
lower is better). Images with a NaN value are left out of that metric only.
The interval describes sampling over images for two fixed sets of
predictions; it says nothing about variation between training runs.
