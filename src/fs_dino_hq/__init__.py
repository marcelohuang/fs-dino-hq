"""FS-DINO-HQ: few-shot segmentation from annotated support images with DINOv2
features, an FS-DINO style meta visual prompt generator, the SAM mask decoder
and an HQ-SAM style refinement branch.

Subpackages:
  modeling     model components and the full model
  data         letterbox preprocessing, LabelMe masks, episode datasets and plans
  training     Stage 1 distillation, Stage 2 training and checkpoint selection,
               HQ training, training-history plots
  inference    ``Predictor`` (Python API) and the ``predict`` command
  evaluation   segmentation metrics, ``evaluate`` and ``compare`` commands
"""
__version__ = "0.1.0"
