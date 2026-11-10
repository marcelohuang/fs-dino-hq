# Data

No images, annotations or trained weights are distributed with this
repository. This document describes the formats the code expects and the data
used for the runs in `docs/results.md`.

## Annotations

Masks are read from [LabelMe](https://github.com/wkentaro/labelme) JSON files
stored next to their images with the same stem (`img_001.png` +
`img_001.json`). Fields used:

```json
{
  "imageHeight": 720,
  "imageWidth": 1280,
  "shapes": [
    {"label": "spacing", "points": [[x1, y1], [x2, y2], [x3, y3], ...]},
    ...
  ]
}
```

Every shape with at least three points is filled as a polygon into a binary
mask of size `imageHeight x imageWidth`; shapes with fewer points are skipped.
All labels are merged into one foreground class unless a label filter is given
(`--label-filter`). There are no ignore regions: unannotated pixels are
background. The mask size is taken from the JSON, so it must match the image.

## Inference: supports and queries

`fs-dino-hq predict` reads

* `--support-dir`: annotated support images (image + LabelMe JSON, any size,
  `.png`/`.jpg`/`.jpeg`). The image is found from the JSON stem.
* `--support-set` (optional): a JSON list of support stems, for example
  `["img_003", "img_017", "img_021", "img_040", "img_052"]`. It selects and
  orders the supports; the same supports are used for every query. Without
  it all annotated images in `--support-dir` are used, sorted by name.
  The reported model was trained with five supports per episode.
* `--query-dir`: query images; an annotation next to a query is optional and
  only used to report IoU/Dice.

The supports should show the target structure with complete masks: every
annotated pixel is treated as foreground and everything else as background.

## Training data for Stage 2 and HQ

Stage 2 and the HQ branch are trained on episodes drawn from a LabelMe dataset
organised by class folder:

```
<synthetic_root>/
  beams/    real-<scene>-<camera>-<id>.png + .json   (scene, camera, id: integers)
  columns/  ...
  slabs/    ...
  walls/    ...
```

The runs in `docs/results.md` used a synthetic rebar dataset with these four
folders (25,087 image/annotation pairs at about 2400 x 1800 pixels; labels
`spacing` and `intersection`; one pair excluded because its mask is empty).
It is not distributed with this repository. The folder name is used to pair
supports with queries of the same structural component; the segmentation
target is always "all annotated rebar".

### Episode plans

`fs-dino-hq make-episodes` checks the dataset, holds out whole scenes for
validation, and writes fixed episode plans. The scene is parsed from the file
stem: by default every stem must match
`^real[-_](?P<scene>\d+)[-_](?P<cam>\d+)[-_](?P<uid>\d+)$` (case-insensitive);
otherwise the command writes `scene_parse_failures.json` and stops. Other naming
schemes need a `--scene-regex` with a named group `scene`; `--group-mode image`
splits by image instead (not scene-disjoint). A plan is a CSV file with
one row per episode:

```
cycle, component, query_image, query_json, query_group,
support_1_image ... support_K_image, support_1_json ... support_K_json,
support_1_group ... support_K_group, episode_idx
```

(paths relative to `<synthetic_root>`). In the training plan every training
image is the query exactly once per cycle, and its supports come from other
scenes of the same component; validation supports are drawn from the training
scenes; a query is never its own support. The trainers consume plan rows in
order (2,000 per epoch) and wrap around at the end.

Plans used for the reported runs (seed 42, validation fraction 0.05; 23,665
training and 1,421 validation images, 9 of 50 scenes held out):

| File | Command options | Used by | SHA-256 |
|---|---|---|---|
| `train_episode_plan_1shot.csv` | `--n-shot 1 --cycles 8` | Stage 2, one-shot | `dd6d9994861c07d8c2391d9afc389c736a59db9227b82faf37f4c062a783d78b` |
| `val_episode_plan_1shot.csv` | `--n-shot 1` | Stage 2 validation (both runs) | `b3da89c61440993a6cf98c8345f02ca4c6c5bd1a2dcb0a06738b0ce58d0d10ef` |
| `train_episode_plan_5shot_8cyc.csv` | `--n-shot 5 --cycles 8` | Stage 2, five-shot | `83b278240ad45e781041c8e745c46dad2104ed096ff7a6a48d604eb0ba42bd7f` |
| `train_episode_plan_5shot.csv` | `--n-shot 5 --cycles 2` | HQ | `289dd66c3d0254e2c41fb8b16261466fcbaf4c59d34673c5083aca3a57e9d071` |
| `val_episode_plan_5shot.csv` | `--n-shot 5` | Stage 2 (five-shot) and HQ validation | `bc6671236cc2c01486647b8782a4781f2d0499bc8a6046cf3fd5e0bbd45f1356` |
| `synthetic_train_manifest.csv` | (any of the above) | split record | `4e62e6049ca59f78a6da78689fb4a51b2692841784918cd28263c8c7df0631e3` |
| `synthetic_val_manifest.csv` | (any of the above) | split record | `7de33e099fadc3a794644fb00eefe8e5f084bf5f793f6672979027c57ad627db` |

With the same dataset files these commands regenerate files with these hashes
(see the Verification section of `docs/results.md`). The one-shot and
five-shot training plans list the same queries in the same order; the
two-cycle five-shot plan is the first 47,330 rows of the eight-cycle plan. The
HQ run consumes 100,000 episodes and therefore wraps around the two-cycle plan;
using the eight-cycle plan instead would give a different episode sequence.

## Unlabelled images for Stage 1 distillation

Stage 1 needs only images. `fs-dino-hq distill` reads a manifest (one path per
line, relative to `--image-root`) for training and one for held-out
validation. `fs-dino-hq make-distill-split` splits a list of all unlabelled
images (`--manifest`, one path per line relative to `--image-root`; stems must
be unique) into `train.txt` and `val.txt`. It keeps frames of one capture
sequence on the same side of the split (the sequence is the file stem without
a trailing `[_-]<digits>` frame counter; `--group-regex` changes the rule), and
removes images that are also used for evaluation (`--exclude-dir` for folders
of LabelMe annotations, `--exclude-list` for lists of file names). With
`--exclude-sequences-of-excluded` it also removes every frame of a capture
sequence that contains an excluded image. One way to build the list:

```bash
(cd $DATA_DIR/unlabelled/images && find . -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
    | sed 's|^\./||' | sort) > all_images.txt
```

The reported adapters were distilled on 5,796 training and 395 validation
unlabelled construction-site photographs (29 evaluation-related images
removed). That split did not use `--exclude-sequences-of-excluded`; see the
limitation in `docs/results.md`.

## Evaluation data of the reported results

46 annotated query photographs and a pool of annotated support photographs
from construction sites, described in `docs/results.md`. They are not
distributed.
