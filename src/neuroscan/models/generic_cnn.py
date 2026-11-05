"""
NeuroScan AI — Generic CNN Architecture from Gómez-Guzmán et al. (2023).

Reproduces the 17-layer Generic Convolutional Neural Network architecture
described in Figure 2 and Section 3 of:
"Classifying Brain Tumors on Magnetic Resonance Imaging by Using
Convolutional Neural Networks", Electronics 2023, 12(4), 955.

Architecture Specifications:
- Input size: [Batch, 3, 256, 256]
- 4 Convolutional blocks:
  - Conv2d(3x3 kernel, stride 1, padding 1)
  - BatchNorm2d
  - ReLU activation
  - MaxPool2d(2x2 window, stride 2)
- Spatial resolution progression:
  - Input: 256x256
  - Block 1: 3 -> 32 channels, 128x128
  - Block 2: 32 -> 64 channels, 64x64
  - Block 3: 64 -> 128 channels, 32x32
  - Block 4: 128 -> 256 channels, 16x16
- Classification Head (Figure 2):
  - Flatten -> [Batch, 256 * 16 * 16 = 65,536]
  - Dropout(p=0.5 per Fig 2 caption; configurable to 0.2 per Table 2)
  - Linear(65536, 4) -> raw logits (CrossEntropyLoss in PyTorch incorporates Softmax)
  - Optional hidden dense stage (if dense_units is configured)
"""
from typing import Dict, Optional

import torch
import torch.nn as nn


class GenericConvBlock(nn.Module):
    """
    Convolutional feature extraction block:
    Conv2d(3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2, stride 2).
    """

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


class PaperGenericCNN(nn.Module):
    """
    Faithful reproduction of the Generic CNN model from Gómez-Guzmán et al. (2023).
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 3,
        dropout: float = 0.5,
        dense_units: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_rate = dropout
        self.dense_units = dense_units

        # 4 Convolutional blocks (Figure 2)
        self.features = nn.Sequential(
            GenericConvBlock(in_channels, 32),    # -> [B, 32, 128, 128]
            GenericConvBlock(32, 64),             # -> [B, 64, 64, 64]
            GenericConvBlock(64, 128),            # -> [B, 128, 32, 32]
            GenericConvBlock(128, 256),           # -> [B, 256, 16, 16]
        )

        # Classification Head (Figure 2: Flatten -> Dropout -> Dense Softmax)
        flattened_dim = 256 * 16 * 16  # 65,536 features for 256x256 input

        if dense_units is not None and dense_units > 0:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened_dim, dense_units),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(dense_units, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout),
                nn.Linear(flattened_dim, num_classes),
            )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw unnormalized logits of shape [Batch, num_classes]."""
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        """Initialize weights with Kaiming normal for Conv and Xavier uniform for Linear."""
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

