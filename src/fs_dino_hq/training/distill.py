"""Stage 1: coarse-to-fine distillation of the final or the early adapter.

    fs-dino-hq distill --config final_adapter.json
    fs-dino-hq distill --config early_adapter.json
    fs-dino-hq distill --config final_adapter.json --resume      # continue a stopped run
    fs-dino-hq distill --config final_adapter.json --smoke --save-dir /tmp/c2f_smoke

The student is the bottleneck adapter (``fs_dino_hq.modeling.distill_adapter``)
on DINOv2 block-3 tokens (block index 2, raw block output as everywhere else in
this package); the teacher is a frozen SAM ViT-H encoder feature on the fixed
64x64 grid:

  --adapter final   teacher = SAM final image embedding [256, 64, 64]
                    (the embedding the SAM mask decoder consumes); checkpoint
                    key "adapter_state". Initialises the Stage 2 adapter.
  --adapter early   teacher = PCA(1280 -> 256) of the SAM block-7 output, the
                    first global-attention block, which HQ-SAM uses as its
                    early-layer feature; checkpoint key "early_adapter_state".
                    Feeds the HQ branch. FS-DINO distils only the final
                    embedding; the early adapter is an extension.

Schedule: the student input resolution grows over stages (default
126 -> 252 -> 518, token grids 9x9 -> 18x18 -> 37x37); the adapter weights are
carried from stage to stage, while the optimiser and the warm-up + cosine
schedule are re-created per stage. The teacher view is always 1024 px, so the
target grid is identical in every stage and the student is bilinearly
upsampled to 64x64 before the loss (FS-DINO's "upsample student" alignment).
The loss is a plain MSE on raw features.

Every epoch logs the stage-resolution train/val loss and a fixed-resolution
diagnostic (--diag-res, default 518) on the held-out split, both over the
full 64x64 grid and restricted to the non-padded content box, so stages are
comparable. ``c2f_last.pt`` is written after every epoch and --resume
continues from it (same stage, next epoch, optimiser/scheduler/RNG restored);
a --save-dir that already holds ``c2f_last.pt`` is refused without --resume.
The early adapter's PCA basis is hashed into the run state
(``config["pca_basis_sha256"]``) and a resume toward a different basis is
refused.

Outputs in --save-dir (``<name>`` is ``final_adapter`` or ``early_adapter``):
  config.json, history.csv, c2f_last.pt, results.json,
  <name>_stage<res>_last.pt / <name>_stage<res>_best.pt (minimum stage val loss),
  <name>_best.pt (minimum diagnostic MSE over all stages; the file used downstream),
  pca_teacher_1280to256.pt (early adapter only: the PCA basis of the teacher).
See docs/training.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

from ..config import parse_with_config
from ..data.distill_data import DINO_PATCH, TEACHER_GRID, LetterboxDistillDataset, read_manifest
from ..modeling.backbones import load_dinov2_vitb14, load_sam_vit_h
from ..modeling.distill_adapter import BottleneckAdapter
from ..utils.provenance import file_sha256
from ..utils.seeding import seed_everything

STATE_KEY = {"final": "adapter_state", "early": "early_adapter_state"}
LOSS = "mse"                 # recorded in config.json / results.json; the only loss
ADAPTER_OUT = 256            # adapter output channels = SAM embedding width = PCA dimension
PCA_DIM = ADAPTER_OUT
PCA_FILE = "pca_teacher_1280to256.pt"
PCA_GEOMETRY = "letterbox_1024_sam_native_pad"
LAST_FILE = "c2f_last.pt"
REQUIRED = ("adapter", "train_manifest", "val_manifest", "dino_ckpt", "sam_ckpt", "save_dir")


# ── Feature extractors ───────────────────────────────────────────────────────

class PersistentHook:
    """Capture a module's output on every forward; robust to tuple/list outputs."""

    def __init__(self, module: nn.Module) -> None:
        self.output: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._fn)

    def _fn(self, module, inp, output) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        self.output = output

    def remove(self) -> None:
        self._handle.remove()


class DinoL3Extractor:
    """DINOv2 block-``block_index`` tokens without the final norm, as [B, 768, H, W].

    Mirrors ``DINOv2Layer3Extractor`` in ``fs_dino_hq.modeling.model`` so the
    distilled adapter sees exactly the feature the downstream model feeds it.
    """

    def __init__(self, dino_model: nn.Module, block_index: int) -> None:
        self.dino = dino_model
        self.block_index = block_index
        self._hook = PersistentHook(dino_model.blocks[block_index])

    @torch.no_grad()
    def __call__(self, dino_input: torch.Tensor) -> torch.Tensor:
        self.dino.forward_features(dino_input)          # triggers the hook
        raw = self._hook.output                         # [B, N, C], CLS may be present
        B, N, C = raw.shape
        # Strip CLS if (N-1) is a perfect square (dinov2_vitb14 has no register tokens)
        if int(math.sqrt(N - 1)) ** 2 == (N - 1):
            raw = raw[:, 1:, :]
            N -= 1
        hw = int(math.sqrt(N))
        return raw.transpose(1, 2).reshape(B, C, hw, hw).contiguous()


class SamEarlyExtractor:
    """SAM image-encoder block-``early_block`` output as BCHW [B, 1280, 64, 64].

    The block output is BHWC [B, 64, 64, 1280]; the full encoder forward runs
    and a hook reads the block.
    """

    def __init__(self, sam_image_encoder: nn.Module, early_block: int) -> None:
        self.encoder = sam_image_encoder
        self.early_block = early_block
        self._hook = PersistentHook(sam_image_encoder.blocks[early_block])

    @torch.no_grad()
    def __call__(self, sam_input: torch.Tensor) -> torch.Tensor:
        self.encoder(sam_input)                          # full forward triggers hook
        raw = self._hook.output                          # [B,64,64,1280] (BHWC)
        return raw.permute(0, 3, 1, 2).contiguous()      # [B,1280,64,64]


# ── PCA teacher compression (early adapter) ──────────────────────────────────

def fit_pca_teacher(sam_early: SamEarlyExtractor, fit_loader: DataLoader, device: torch.device,
                    pca_dim: int) -> Dict:
    """Fit PCA(C -> pca_dim) on all 64x64 SAM early-block feature vectors of the loader's images.

    Mean and covariance are accumulated in float64, image by image. Returns a
    dict with ``W`` [pca_dim, C] float32 (top eigenvectors as rows), ``mean``
    [C] float32, ``evr`` (explained-variance ratio of the kept components),
    ``in_dim`` and ``out_dim``. The eigenvector signs (and the basis within
    near-degenerate eigenvalues) depend on the eigensolver, so a refit on
    another device need not reproduce a basis exactly; reuse a saved basis
    with --pca-cache to continue a teacher.
    """
    n_total = 0
    C = None
    sum_x: Optional[torch.Tensor] = None
    xtx: Optional[torch.Tensor] = None
    print(f"[PCA] fitting PCA on SAM block-{sam_early.early_block} features (all tokens) ...", flush=True)

    for bi, batch in enumerate(fit_loader):
        sam_img = batch["sam_image"].to(device)
        feat = sam_early(sam_img)                         # [B,C,64,64]
        B, Cc, H, W = feat.shape
        if C is None:
            C = Cc
            sum_x = torch.zeros(C, dtype=torch.float64, device=device)
            xtx = torch.zeros(C, C, dtype=torch.float64, device=device)
        feat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)   # [B, 4096, C]
        for img_i in range(B):
            X = feat[img_i].double()                           # [HW, C]
            sum_x += X.sum(dim=0)
            xtx += X.t() @ X
            n_total += X.shape[0]
        if (bi + 1) % 20 == 0:
            print(f"[PCA]   processed {bi + 1} batches, {n_total} vectors", flush=True)

    if n_total == 0 or sum_x is None or xtx is None:
        raise RuntimeError("PCA fit got no data")
    mean = sum_x / n_total                                # [C]
    cov = xtx / n_total - torch.outer(mean, mean)         # [C, C]
    cov = 0.5 * (cov + cov.t())                           # symmetrize

    evals, evecs = torch.linalg.eigh(cov)                 # ascending
    idx = torch.argsort(evals, descending=True)
    top = idx[:pca_dim]
    W = evecs[:, top].t().contiguous()                    # [pca_dim, C]
    sel_var = evals[top].clamp_min(0).sum()
    tot_var = evals.clamp_min(0).sum().clamp_min(1e-12)
    evr = float((sel_var / tot_var).item())

    print(f"[PCA] fit done: {n_total} vectors, dim {C}->{pca_dim}, "
          f"explained-variance-ratio sum = {evr:.4f}", flush=True)
    return {
        "W": W.float().cpu(),
        "mean": mean.float().cpu(),
        "evr": evr,
        "in_dim": int(C),
        "out_dim": int(pca_dim),
    }


def build_pca_conv(pca: Dict) -> nn.Conv2d:
    """Frozen 1x1 conv implementing y = W @ (x - mean)  (bias = -(W @ mean))."""
    W = pca["W"]                                          # [out, in]
    mean = pca["mean"]                                    # [in]
    out_dim, in_dim = W.shape
    conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=True)
    with torch.no_grad():
        conv.weight.copy_(W.view(out_dim, in_dim, 1, 1))
        conv.bias.copy_(-(W @ mean))
    for p in conv.parameters():
        p.requires_grad_(False)
    conv.eval()
    return conv


# ── Schedule ─────────────────────────────────────────────────────────────────

def build_warmup_cosine(optimizer, epochs: int, warmup_epochs: int):
    """Linear warm-up (1 percent -> 100 percent) then cosine, stepped per epoch. Returns (scheduler, warmup)."""
    warmup = max(0, min(warmup_epochs, epochs - 1))
    if warmup > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                 total_iters=warmup),
                        CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))],
            milestones=[warmup],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    return scheduler, warmup


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fs-dino-hq distill", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="JSON file whose keys are option names with underscores; command-line values win")
    p.add_argument("--adapter", choices=["final", "early"], default=None,
                   help="which adapter to distil (required)")
    p.add_argument("--train-manifest", default=None, help="text file of training images (required)")
    p.add_argument("--val-manifest", default=None, help="text file of held-out images (required)")
    p.add_argument("--image-root", default=None, help="folder the manifest entries are relative to")
    p.add_argument("--dino-ckpt", default=None, help="dinov2_vitb14_pretrain.pth (required)")
    p.add_argument("--sam-ckpt", default=None, help="sam_vit_h_4b8939.pth (required)")
    p.add_argument("--save-dir", default=None, help="output folder (required)")
    p.add_argument("--stages", default="126:8,252:8,518:14",
                   help="comma list of <dino_res>:<epochs>, in training order")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=1, help="per stage")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--diag-res", type=int, default=518, help="fixed resolution of the held-out diagnostic")
    p.add_argument("--dino-block-index", type=int, default=2, help="0-based DINOv2 block (2 = block 3)")
    p.add_argument("--sam-early-block", type=int, default=7,
                   help="0-based SAM ViT-H block of the early teacher (7 = first global-attention block)")
    p.add_argument("--pca-fit-images", type=int, default=400,
                   help="early adapter: training images (seeded subset) used to fit the PCA teacher")
    p.add_argument("--pca-cache", default=None,
                   help=f"early adapter: existing {PCA_FILE} to reuse instead of fitting one "
                        f"(otherwise <save-dir>/{PCA_FILE} is reused if present, or fitted on the train split); "
                        f"refused when <save-dir> already holds a different basis")
    p.add_argument("--resume", action="store_true",
                   help=f"continue from <save-dir>/{LAST_FILE} if it exists (a --save-dir holding "
                        f"{LAST_FILE} is refused without --resume)")
    p.add_argument("--stop-after-stage", type=int, default=0,
                   help="exit after this many stages (0 = run all stages)")
    p.add_argument("--max-train-images", type=int, default=0, help="use only the first N training images")
    p.add_argument("--max-val-images", type=int, default=0, help="use only the first N held-out images")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: 1 epoch per stage, 8 train / 4 val images, batch size <= 2")
    p.add_argument("--print-every", type=int, default=50)
    return p


def parse_stages(spec: str) -> List[Tuple[int, int]]:
    """'126:8,252:8,518:14' -> [(126, 8), (252, 8), (518, 14)]; resolutions must be multiples of 14."""
    out = []
    for item in spec.split(","):
        res, ep = item.split(":")
        out.append((int(res), int(ep)))
    if not out or any(r % DINO_PATCH for r, _ in out) or any(e < 1 for _, e in out):
        raise ValueError(f"bad --stages {spec!r}: need <res>:<epochs> items with res a multiple of "
                         f"{DINO_PATCH} and epochs >= 1")
    return out


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parse_with_config(parser, argv)
    missing = [r for r in REQUIRED if not getattr(args, r)]
    if missing:
        parser.error("required (command line or config): "
                     + ", ".join("--" + r.replace("_", "-") for r in missing))
    try:
        stages = parse_stages(args.stages)
    except ValueError as e:
        parser.error(str(e))
    if args.diag_res % DINO_PATCH:
        parser.error(f"--diag-res must be a multiple of {DINO_PATCH}")
    if args.pca_fit_images < 1:
        parser.error("--pca-fit-images must be >= 1")
    if args.smoke:
        args.stages = ",".join(f"{r}:1" for r, _ in stages)
        args.max_train_images = args.max_train_images or 8
        args.max_val_images = args.max_val_images or 4
        args.pca_fit_images = min(args.pca_fit_images, 4)
        args.batch_size = min(args.batch_size, 2)
    return args


# ── Teacher, loss and metrics ────────────────────────────────────────────────

class Teacher:
    """Frozen SAM-side target on the 64x64 grid."""

    def __init__(self, kind: str, sam, device, early_block: int = 7, pca: Optional[Dict] = None):
        self.kind = kind
        self.encoder = sam.image_encoder
        if kind == "early":
            self.early = SamEarlyExtractor(sam.image_encoder, early_block)
            self.pca_conv = build_pca_conv(pca).to(device)
        self.out_channels = ADAPTER_OUT

    @torch.no_grad()
    def __call__(self, sam_image: torch.Tensor) -> torch.Tensor:
        if self.kind == "final":
            return self.encoder(sam_image)
        return self.pca_conv(self.early(sam_image))


def distill_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Plain MSE on raw features over the full 64x64 grid."""
    return F.mse_loss(student, teacher)


@torch.no_grad()
def metrics(student: torch.Tensor, teacher: torch.Tensor, content: torch.Tensor) -> Dict[str, float]:
    """MSE, normalised MSE and cosine over the full grid and over the content box only."""
    cos = F.cosine_similarity(student, teacher, dim=1, eps=1e-8)          # [B, 64, 64]
    sq = (student - teacher).pow(2).mean(1)                                # [B, 64, 64]
    nsq = (F.normalize(student, dim=1) - F.normalize(teacher, dim=1)).pow(2).mean(1)
    c = content[:, 0]
    n_c = c.sum().clamp_min(1.0)
    return {"mse": sq.mean().item(), "normalized_mse": nsq.mean().item(), "cosine": cos.mean().item(),
            "mse_content": (sq * c).sum().item() / n_c.item(),
            "normalized_mse_content": (nsq * c).sum().item() / n_c.item(),
            "cosine_content": (cos * c).sum().item() / n_c.item(),
            "student_norm": student.norm(dim=1).mean().item(), "teacher_norm": teacher.norm(dim=1).mean().item()}


def restore_stage_best(ckpt: Dict, save_dir: Path, name: str, res: int) -> float:
    """Stage-best loss to continue from when resuming inside a stage.

    Checkpoints carry it as ckpt["best_stage"]. For a checkpoint without that
    key, the stage-best file already on disk is used
    (<name>_stage<res>_best.pt stores the stage validation metrics under
    "stage_val"), so a resumed run never overwrites a better stage-best
    checkpoint with a worse epoch. Returns +inf only when neither exists.
    """
    if "best_stage" in ckpt:
        return float(ckpt["best_stage"])
    cand = Path(save_dir) / f"{name}_stage{res}_best.pt"
    if cand.is_file():
        prev = torch.load(cand, map_location="cpu", weights_only=False)
        if isinstance(prev.get("stage_val"), dict) and "loss" in prev["stage_val"]:
            return float(prev["stage_val"]["loss"])
    return math.inf


def checkpoint_provenance(ck: Dict) -> Dict:
    """Which stage / epoch a saved adapter checkpoint comes from."""
    return {k: ck.get(k) for k in ("stage", "res", "epoch_in_stage", "epoch", "selected_by")}


def adapter_sha256(adapter: nn.Module) -> str:
    """SHA-256 over the adapter state (sorted keys, raw tensor bytes); history.csv logs its first 16 hex digits."""
    h = hashlib.sha256()
    for k, v in sorted(adapter.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ── Loops ────────────────────────────────────────────────────────────────────

def run_epoch(adapter, dino_l3, teacher, loader, device, optimizer=None, print_every=50, tag=""):
    """One pass over ``loader``; trains when ``optimizer`` is given. Returns image-weighted mean metrics + loss."""
    train = optimizer is not None
    adapter.train(train)
    sums: Dict[str, float] = {}
    n, loss_sum = 0, 0.0
    for step, batch in enumerate(loader, 1):
        xd = batch["dino_image"].to(device, non_blocking=True)
        xs = batch["sam_image"].to(device, non_blocking=True)
        content = batch["content_mask"].to(device)
        with torch.no_grad():
            feat = dino_l3(xd)                                   # [B, 768, g, g]
            target = teacher(xs)                                 # [B, 256, 64, 64]
        with torch.set_grad_enabled(train):
            student = F.interpolate(adapter(feat), size=(TEACHER_GRID, TEACHER_GRID),
                                    mode="bilinear", align_corners=False)
            loss = distill_loss(student, target)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        bs = xd.shape[0]
        m = metrics(student.detach(), target, content)
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + v * bs
        loss_sum += loss.item() * bs
        n += bs
        if step == 1:
            g = feat.shape[-1]
            if g != xd.shape[-1] // DINO_PATCH:
                raise RuntimeError(f"token grid {g} does not match input {xd.shape[-1]}")
            print(f"[{tag}] dino_input={tuple(xd.shape[-2:])} tokens={tuple(feat.shape[-2:])} "
                  f"student={tuple(student.shape[-3:])} teacher={tuple(target.shape[-3:])}", flush=True)
        if train and (step % print_every == 0 or step == 1):
            print(f"[{tag} step {step:05d}] loss={loss.item():.6f} cos={m['cosine']:.4f} "
                  f"cos_content={m['cosine_content']:.4f} |s|={m['student_norm']:.3f} |t|={m['teacher_norm']:.3f}",
                  flush=True)
    out = {k: v / max(n, 1) for k, v in sums.items()}
    out["loss"] = loss_sum / max(n, 1)
    return out


def make_loader(ds, batch_size, shuffle, num_workers, seed, epoch):
    """Loader with its own generator seeded with seed*1000 + epoch (data order independent of global RNG)."""
    gen = torch.Generator(); gen.manual_seed(seed * 1000 + epoch)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle, generator=gen,
                      persistent_workers=False)


def pca_basis_sha256(pca: Dict) -> str:
    """SHA-256 over the PCA basis that defines the early teacher (``W`` and ``mean``, float32 bytes).

    Hashes the tensors rather than the file, so the same basis saved to
    another file name gives the same value.
    """
    h = hashlib.sha256()
    for k in ("W", "mean"):
        t = torch.as_tensor(pca[k]).detach().cpu().float().contiguous()
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()


def load_or_fit_pca(args, save_dir: Path, sam, train_entries: List[str], first_res: int,
                    device: torch.device) -> Tuple[Dict, str]:
    """PCA basis of the early teacher: --pca-cache, else <save-dir>/pca_teacher_1280to256.pt, else a fit.

    A fit uses a seeded subset (``random.Random(seed)``) of the training images
    and is saved to <save-dir>; a --pca-cache file is copied there unchanged
    when the run folder has none, and refused when the run folder already
    holds a different basis, so the run folder always holds the basis its
    adapter was distilled toward.
    """
    pca_path = save_dir / PCA_FILE
    if args.pca_cache:
        pca = torch.load(args.pca_cache, map_location="cpu", weights_only=True)
        if not pca_path.exists():
            shutil.copyfile(args.pca_cache, pca_path)
        else:
            held = torch.load(pca_path, map_location="cpu", weights_only=True)
            if pca_basis_sha256(held) != pca_basis_sha256(pca):
                raise SystemExit(f"ERROR: {pca_path} already holds a different PCA basis than --pca-cache "
                                 f"{args.pca_cache}; use a new --save-dir, or remove {pca_path} if no run "
                                 f"in that folder was distilled toward it")
        return pca, args.pca_cache
    if pca_path.is_file():
        return torch.load(pca_path, map_location="cpu", weights_only=True), str(pca_path)
    fit_ds = LetterboxDistillDataset(train_entries, args.image_root, first_res, max_images=args.max_train_images)
    idx = list(range(len(fit_ds))); random.Random(args.seed).shuffle(idx)
    idx = idx[:args.pca_fit_images]
    # No generator: creating the iterator draws one value from the global torch RNG
    # (after the adapter is built, so the adapter initialisation is unaffected).
    fit_loader = DataLoader(Subset(fit_ds, idx), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    pca = fit_pca_teacher(SamEarlyExtractor(sam.image_encoder, args.sam_early_block), fit_loader, device,
                          PCA_DIM)
    pca["fit_images"] = [fit_ds.entries[i] for i in idx]
    pca["geometry"] = PCA_GEOMETRY
    torch.save(pca, pca_path)
    return pca, str(pca_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed, deterministic_cudnn=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    last_path = save_dir / LAST_FILE
    if last_path.is_file() and not args.resume:
        # starting over would overwrite <name>_best.pt, history.csv and the resumable state
        raise SystemExit(f"ERROR: {last_path} exists; pass --resume to continue or choose a new --save-dir")
    stages = parse_stages(args.stages)
    state_key = STATE_KEY[args.adapter]
    name = f"{args.adapter}_adapter"
    print(f"[INFO] adapter={args.adapter} stages={stages} loss={LOSS} device={device}", flush=True)

    # ── data ─────────────────────────────────────────────────────────────
    train_entries = read_manifest(args.train_manifest)
    val_entries = read_manifest(args.val_manifest)
    train_ds = LetterboxDistillDataset(train_entries, args.image_root, stages[0][0], max_images=args.max_train_images)
    val_ds = LetterboxDistillDataset(val_entries, args.image_root, stages[0][0], max_images=args.max_val_images)
    diag_ds = LetterboxDistillDataset(val_entries, args.image_root, args.diag_res, max_images=args.max_val_images)
    print(f"[INFO] train images={len(train_ds)} val images={len(val_ds)}", flush=True)
    # checked before the backbones are loaded: an empty held-out set has no
    # diagnostic, and training drops the last incomplete batch
    if len(val_ds) == 0:
        raise SystemExit(f"ERROR: no held-out images in --val-manifest {args.val_manifest}")
    if len(train_ds) < args.batch_size:
        raise SystemExit(f"ERROR: {len(train_ds)} training images in --train-manifest {args.train_manifest}; "
                         f"need at least --batch-size {args.batch_size}")

    # ── models ───────────────────────────────────────────────────────────
    # Construction order after seeding fixes the adapter initialisation:
    # DINOv2 hub model -> full SAM ViT-H -> adapter -> (early) PCA fit.
    dino = load_dinov2_vitb14(args.dino_ckpt, device)
    sam = load_sam_vit_h(args.sam_ckpt, device)
    dino_l3 = DinoL3Extractor(dino, args.dino_block_index)
    adapter = BottleneckAdapter(in_channels=768, out_channels=ADAPTER_OUT).to(device)

    pca = None
    pca_sha = None
    if args.adapter == "early":
        pca, pca_src = load_or_fit_pca(args, save_dir, sam, train_entries, stages[0][0], device)
        pca_sha = pca_basis_sha256(pca)
        print(f"[INFO] PCA teacher: {pca_src} evr={float(pca['evr']):.4f} basis_sha={pca_sha[:16]}", flush=True)
    teacher = Teacher(args.adapter, sam, device, args.sam_early_block, pca)

    # ── resume state ─────────────────────────────────────────────────────
    history: List[Dict] = []
    best_diag = math.inf
    best_stage_resume = math.inf
    start_stage, start_epoch = 0, 0
    opt_state = sched_state = None
    if args.resume and last_path.is_file():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        prev = ck["config"]
        if (prev["stages"], prev["adapter"], prev.get("loss", LOSS)) != (args.stages, args.adapter, LOSS):
            raise SystemExit(f"{last_path} was written with stages={prev['stages']!r} adapter={prev['adapter']!r}; "
                             f"this run has stages={args.stages!r} adapter={args.adapter!r}")
        prev_pca_sha = prev.get("pca_basis_sha256")      # absent in states written before it was recorded
        if prev_pca_sha is not None and prev_pca_sha != pca_sha:
            raise SystemExit(f"{last_path} was distilled toward PCA basis {prev_pca_sha[:16]}, but this run "
                             f"loaded basis {(pca_sha or 'none')[:16]} ({PCA_FILE} in --save-dir or --pca-cache); "
                             f"resume with the original basis")
        adapter.load_state_dict(ck[state_key], strict=True)
        history, best_diag = ck["history"], ck["best_diag"]
        start_stage, start_epoch = ck["stage_idx"], ck["epoch_in_stage"] + 1
        opt_state, sched_state = ck["optimizer_state"], ck["scheduler_state"]
        torch.set_rng_state(ck["rng"]["torch"]); random.setstate(ck["rng"]["python"]); np.random.set_state(ck["rng"]["numpy"])
        if start_epoch >= stages[start_stage][1]:
            start_stage, start_epoch, opt_state, sched_state = start_stage + 1, 0, None, None
        else:
            best_stage_resume = restore_stage_best(ck, save_dir, name, stages[start_stage][0])
        print(f"[RESUME] from {last_path}: stage {start_stage} epoch {start_epoch} "
              f"adapter_sha={adapter_sha256(adapter)[:16]} stage_best={best_stage_resume:.6f}", flush=True)
    elif args.resume:
        print(f"[RESUME] {last_path} not found: starting from scratch", flush=True)

    config = {**vars(args), "loss": LOSS, "pca_dim": PCA_DIM,
              "stages_parsed": stages, "state_key": state_key,
              "n_train": len(train_ds), "n_val": len(val_ds), "teacher_grid": TEACHER_GRID,
              "dino_view": "letterbox longest-side, top-left, black pad before ImageNet normalisation "
                           "(LetterboxPreprocess)",
              "sam_view": "letterbox longest-side to 1024, SAM normalisation, zero pad after (Sam.preprocess)",
              "student_alignment": "bilinear upsample of adapter output to 64x64 (align_corners=False)",
              "optimizer": "AdamW, re-created at each stage; linear warm-up then cosine within the stage",
              "sha256": {k: file_sha256(v) for k, v in (("dino_ckpt", args.dino_ckpt), ("sam_ckpt", args.sam_ckpt),
                                                        ("train_manifest", args.train_manifest),
                                                        ("val_manifest", args.val_manifest)) if v},
              "pca_evr": float(pca["evr"]) if pca else None,
              "pca_basis_sha256": pca_sha}
    (save_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    hist_path = save_dir / "history.csv"

    def save_adapter(path: Path, extra: Dict):
        torch.save({state_key: adapter.state_dict(), "config": config, **extra}, path)

    def write_history():
        if history:
            with open(hist_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)

    def diag_eval(tag: str) -> Dict[str, float]:
        loader = make_loader(diag_ds, args.batch_size, False, args.num_workers, args.seed, 0)
        return run_epoch(adapter, dino_l3, teacher, loader, device, tag=tag)

    # ── stages ───────────────────────────────────────────────────────────
    global_epoch = sum(e for _, e in stages[:start_stage]) + start_epoch
    for si in range(start_stage, len(stages)):
        res, n_epochs = stages[si]
        train_ds.set_dino_res(res); val_ds.set_dino_res(res)
        sha_in = adapter_sha256(adapter)
        optimizer = AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler, warmup = build_warmup_cosine(optimizer, n_epochs, args.warmup_epochs)
        first_epoch = start_epoch if si == start_stage else 0
        if opt_state is not None and si == start_stage:
            optimizer.load_state_dict(opt_state); scheduler.load_state_dict(sched_state)
            opt_state = sched_state = None
        else:
            pre = diag_eval(f"diag@{args.diag_res} before stage {res}")
            history.append({"stage": si, "res": res, "epoch_in_stage": 0, "global_epoch": global_epoch,
                            "event": "stage_start", "adapter_sha16": sha_in[:16],
                            **{f"diag_{k}": v for k, v in pre.items()}, "lr": args.lr,
                            "train_loss": "", "val_loss": "", "val_cosine": "", "val_cosine_content": "", "seconds": 0.0})
            print(f"[STAGE {si} res={res}] start: adapter_sha={sha_in[:16]} "
                  f"diag@{args.diag_res} mse={pre['mse']:.6f} cos_content={pre['cosine_content']:.4f} "
                  f"(weights carried from the previous stage)", flush=True)
        best_stage = best_stage_resume if (si == start_stage and first_epoch > 0) else math.inf
        for ep in range(first_epoch, n_epochs):
            t0 = time.time()
            tl = make_loader(train_ds, args.batch_size, True, args.num_workers, args.seed, global_epoch)
            tr = run_epoch(adapter, dino_l3, teacher, tl, device, optimizer, args.print_every,
                           tag=f"s{si} r{res} e{ep + 1}/{n_epochs}")
            vl = make_loader(val_ds, args.batch_size, False, args.num_workers, args.seed, 0)
            va = run_epoch(adapter, dino_l3, teacher, vl, device, tag="val")
            # at the diagnostic resolution the val pass is the diagnostic
            dg = va if res == args.diag_res else diag_eval(f"diag@{args.diag_res}")
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            global_epoch += 1
            row = {"stage": si, "res": res, "epoch_in_stage": ep + 1, "global_epoch": global_epoch, "event": "epoch",
                   "adapter_sha16": adapter_sha256(adapter)[:16],
                   **{f"diag_{k}": v for k, v in dg.items()}, "lr": lr,
                   "train_loss": tr["loss"], "val_loss": va["loss"], "val_cosine": va["cosine"],
                   "val_cosine_content": va["cosine_content"], "seconds": round(time.time() - t0, 1)}
            history.append(row); write_history()
            print(f"[STAGE {si} res={res} epoch {ep + 1}/{n_epochs}] train={tr['loss']:.6f} val={va['loss']:.6f} "
                  f"val_cos={va['cosine']:.4f}/{va['cosine_content']:.4f}(content) "
                  f"diag@{args.diag_res} mse={dg['mse']:.6f} mse_content={dg['mse_content']:.6f} "
                  f"cos_content={dg['cosine_content']:.4f} lr={lr:.2e} {row['seconds']}s", flush=True)
            extra = {"epoch": global_epoch, "stage": si, "res": res, "epoch_in_stage": ep + 1,
                     "val_mse": va["mse"], "val_cosine": va["cosine"], "diag": dg, "stage_val": va}
            save_adapter(save_dir / f"{name}_stage{res}_last.pt", extra)
            if va["loss"] < best_stage:
                best_stage = va["loss"]; save_adapter(save_dir / f"{name}_stage{res}_best.pt", extra)
            if dg["mse"] < best_diag:
                best_diag = dg["mse"]
                save_adapter(save_dir / f"{name}_best.pt", {**extra, "selected_by": f"min diag@{args.diag_res} mse"})
            torch.save({state_key: adapter.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(), "stage_idx": si, "epoch_in_stage": ep,
                        "history": history, "best_diag": best_diag, "best_stage": best_stage, "config": config,
                        "rng": {"torch": torch.get_rng_state(), "python": random.getstate(),
                                "numpy": np.random.get_state()}},
                       last_path)
        start_epoch = 0
        if args.stop_after_stage and si + 1 >= args.stop_after_stage:
            print(f"[INFO] --stop-after-stage {args.stop_after_stage}: exiting after stage {si}", flush=True)
            return 0

    # ── final report: evaluate the checkpoints on disk ──────────────────
    results = evaluate_run_checkpoints(save_dir, name, state_key, stages, args, adapter, dino_l3, teacher,
                                       diag_ds, device)
    results.update({"adapter": args.adapter, "stages": stages, "loss": LOSS,
                    "best_diag_mse_during_training": best_diag, "history_rows": len(history)})
    (save_dir / "results.json").write_text(json.dumps(results, indent=2))
    print("[DONE] " + json.dumps({k: results[k] for k in ("selected_best", "stage_final_best", "stage_final_last")},
                                 default=str), flush=True)
    return 0


def evaluate_run_checkpoints(save_dir: Path, name: str, state_key: str, stages, args, adapter, dino_l3,
                             teacher, diag_ds, device) -> Dict:
    """Score the saved checkpoints of a run on the held-out diagnostic set.

    Reports, each with its provenance (stage, resolution, epoch): the globally
    selected <name>_best.pt (minimum diagnostic MSE during training), the best
    and the last checkpoint of the final (highest-resolution) stage, and the
    last checkpoint of every stage. Loads each checkpoint into ``adapter``.
    """
    last_res = stages[-1][0]
    loader = make_loader(diag_ds, args.batch_size, False, args.num_workers, args.seed, 0)

    def score(path: Path) -> Dict:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        adapter.load_state_dict(ck[state_key], strict=True)
        adapter.eval()
        return {"path": str(path), **checkpoint_provenance(ck),
                "diag": run_epoch(adapter, dino_l3, teacher, loader, device, tag=f"eval {path.name}")}

    results: Dict = {"diag_res": args.diag_res, "n_diag_images": len(diag_ds)}
    results["selected_best"] = score(save_dir / f"{name}_best.pt")
    results["stage_final_best"] = score(save_dir / f"{name}_stage{last_res}_best.pt")
    results["stage_final_last"] = score(save_dir / f"{name}_stage{last_res}_last.pt")
    results["stage_last_checkpoints"] = {str(res): score(save_dir / f"{name}_stage{res}_last.pt") for res, _ in stages}
    sel = results["selected_best"]
    results["selection_note"] = (f"{name}_best.pt = stage {sel['stage']} ({sel['res']} px) epoch {sel['epoch_in_stage']}, "
                                 f"chosen by minimum diag@{args.diag_res} MSE during training; "
                                 f"stage_final_best is chosen by the {last_res}-px validation loss")
    results["checkpoints"] = sorted(p.name for p in save_dir.glob("*.pt"))
    return results


if __name__ == "__main__":
    sys.exit(main())
