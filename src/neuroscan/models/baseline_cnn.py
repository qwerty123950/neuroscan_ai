"""
NeuroScan AI — Compact Baseline Convolutional Neural Network (CNN).

A CPU-feasible, non-pretrained baseline architecture for 4-class brain tumor
MRI classification:
- 4 convolutional feature extraction blocks (Conv2d + BatchNorm2d + ReLU + MaxPool2d).
- Adaptive average pooling to decouple spatial input size from classifier width.
- Fully connected classification head with Dropout regularization.
- Outputs 4 raw unnormalized logits (glioma, meningioma, notumor, pituitary).
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d (3x3) -> BatchNorm2d -> ReLU -> MaxPool2d (2x2)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    Compact baseline CNN for 4-class Brain Tumor MRI classification.

    Architecture summary:
    - Input: [Batch, 3, 224, 224]
    - Block 1: 3 -> 32 channels, spatial: 224x224 -> 112x112
    - Block 2: 32 -> 64 channels, spatial: 112x112 -> 56x56
    - Block 3: 64 -> 128 channels, spatial: 56x56 -> 28x28
    - Block 4: 128 -> 256 channels, spatial: 28x28 -> 14x14
    - Global Pooling: AdaptiveAvgPool2d((1, 1)) -> [Batch, 256, 1, 1]
    - Classifier Head:
      - Flatten -> [Batch, 256]
      - Dropout(p=0.4)
      - Linear(256, 128) -> ReLU
      - Dropout(p=0.2)
      - Linear(128, 4) -> [Batch, 4] raw logits

    Parameters: ~422,308 total (~1.6 MB memory footprint), optimized for CPU training.
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Feature Extractor
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),    # -> [B, 32, 112, 112]
            ConvBlock(32, 64),             # -> [B, 64, 56, 56]
            ConvBlock(64, 128),            # -> [B, 128, 28, 28]
            ConvBlock(128, 256),           # -> [B, 256, 14, 14]
        )

        # Global Spatial Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> [B, 256, 1, 1]

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw unnormalized logits of shape [Batch, num_classes]."""
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        """Kaiming normal initialization for Conv layers and Xavier uniform for Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def count_parameters(self) -> Dict[str, int]:
        """Return counts of trainable, non-trainable, and total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "non_trainable": total - trainable,
            "total": total,
        }

