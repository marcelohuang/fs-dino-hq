"""Bottleneck adapter trained by Stage 1 distillation (the "student").

Transforms DINOv2 block-3 features (B×768×H×W) into SAM-aligned embeddings
(B×256×H×W) via convolutional reduction, self-attention and a pointwise
projection. The distilled early adapter is used in this form inside the HQ
model; the distilled final adapter is loaded into
``fs_dino_hq.modeling.model.BottleneckAdapter`` (same weights, ``.layer.``
infix removed, ``norm_first=True``, dropout 0).

Architecture (FS-DINO bottleneck adapter):
  1×1 conv: 768 → 256
  3×3 conv: 256 → 256
  1×1 conv: 256 → 128
  2× self-attention blocks at 128 channels (FFN inner dim 512)
  1×1 conv: 128 → 256
"""

import torch
import torch.nn as nn


class SpatialTransformerBlock(nn.Module):
    """Self-attention block operating on spatial feature maps.

    Reshapes B×C×H×W → B×(HW)×C, applies one transformer encoder layer,
    then reshapes back to B×C×H×W.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial feature map.
        Returns:
            (B, C, H, W) after self-attention.
        """
        B, C, H, W = x.shape
        # Flatten spatial dims → sequence
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_seq = self.layer(x_seq)              # (B, H*W, C)
        # Reshape back to spatial
        return x_seq.transpose(1, 2).view(B, C, H, W)


class BottleneckAdapter(nn.Module):
    """Bottleneck adapter: the only trainable module in Stage 1.

    Input:  B × 768 × H × W  (DINOv2 3rd-layer features)
    Output: B × 256 × H × W  (aligned with SAM encoder output channels)
    """

    def __init__(self, in_channels: int = 768, mid_channels: int = 256,
                 attn_channels: int = 128, out_channels: int = 256,
                 nhead: int = 8, ffn_dim: int = 512, num_attn_blocks: int = 2):
        super().__init__()

        # --- Dimension reduction ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, attn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(attn_channels),
            nn.GELU(),
        )

        # --- Self-attention blocks ---
        self.attn_blocks = nn.ModuleList([
            SpatialTransformerBlock(
                d_model=attn_channels, nhead=nhead, ffn_dim=ffn_dim,
            )
            for _ in range(num_attn_blocks)
        ])

        # --- Output projection (no BN on final output) ---
        self.proj_out = nn.Conv2d(attn_channels, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for conv layers; transformer layers use PyTorch defaults."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 768, H, W) DINOv2 3rd-layer spatial features.
        Returns:
            (B, 256, H, W) distilled embedding.
        """
        assert x.dim() == 4, f"Expected 4D input (B,C,H,W), got {x.dim()}D"
        assert x.shape[1] == 768, f"Expected 768 input channels, got {x.shape[1]}"

        x = self.conv1(x)   # (B, 256, H, W)
        x = self.conv2(x)   # (B, 256, H, W)
        x = self.conv3(x)   # (B, 128, H, W)

        for attn in self.attn_blocks:
            x = attn(x)     # (B, 128, H, W)

        x = self.proj_out(x)  # (B, 256, H, W)
        return x
