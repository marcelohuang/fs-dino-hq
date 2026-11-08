# Third-party notices

## Code adapted in this repository

`src/fs_dino_hq/modeling/hq_decoder.py` contains code adapted from:

* Segment Anything, `segment_anything/modeling/mask_decoder.py`
  (`MaskDecoder.predict_masks`), https://github.com/facebookresearch/segment-anything.
  Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
  Licensed under the Apache License, Version 2.0.
* HQ-SAM, `segment_anything/modeling/mask_decoder_hq.py` (`MaskDecoderHQ`),
  https://github.com/SysCV/sam-hq.
  Copyright (c) Meta Platforms, Inc. and affiliates. Modified by HQ-SAM team.
  All rights reserved. Licensed under the Apache License, Version 2.0.

The file states its modifications in its header. The model dimensions in
`build_sam_prompt_decoder` (`src/fs_dino_hq/modeling/backbones.py`) follow
`segment_anything/build_sam.py` (same license). A copy of the Apache License,
Version 2.0 is in `LICENSES/Apache-2.0.txt`.

## Dependencies and external resources (not redistributed here)

| Resource | Obtained from | License stated by the provider |
|---|---|---|
| `segment-anything` Python package (code) | https://github.com/facebookresearch/segment-anything | Apache-2.0 |
| SAM ViT-H weights `sam_vit_h_4b8939.pth` | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth | Apache-2.0 |
| DINOv2 code (`torch.hub`, commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`) | https://github.com/facebookresearch/dinov2 | Apache-2.0 for the DINOv2 code used here (`dinov2_vitb14`) |
| DINOv2 ViT-B/14 weights `dinov2_vitb14_pretrain.pth` | https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth | Apache-2.0 |
| HQ-SAM ViT-H weights `sam_hq_vit_h.pth` | https://huggingface.co/lkeab/hq-sam | Apache-2.0 |

At the pinned commit the DINOv2 repository also contains Cell-DINO and
XRay-DINO code and models under other licenses (`LICENSE_CELL_DINO_CODE`,
`LICENSE_CELL_DINO_MODELS`, `LICENSE_XRAY_DINO_MODEL`); its `hubconf.py`
imports their hub modules when any model is loaded through `torch.hub`. This
project does not use those models. Check the providers' current terms before
use. Models trained with this code
contain weights derived from the SAM and HQ-SAM checkpoints.
