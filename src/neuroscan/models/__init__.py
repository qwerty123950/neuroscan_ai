"""
NeuroScan AI — Models Module.
"""
from typing import Any, Dict, Optional

import torch.nn as nn

from neuroscan.models.baseline_cnn import BaselineCNN
from neuroscan.models.generic_cnn import PaperGenericCNN

__all__ = [
    "BaselineCNN",
    "PaperGenericCNN",
    "get_model",
]


def get_model(
    architecture: str = "PaperGenericCNN",
    num_classes: int = 4,
    in_channels: int = 3,
    **kwargs: Any,
) -> nn.Module:
    """
    Factory function to instantiate models by architecture name.
    """
    arch_lower = architecture.lower()
    if arch_lower in ("papergenericcnn", "generic_cnn", "genericcnn"):
        dropout = kwargs.get("dropout", 0.5)
        dense_units = kwargs.get("dense_units", None)
        return PaperGenericCNN(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=dropout,
            dense_units=dense_units,
        )
    elif arch_lower in ("baselinecnn", "baseline_cnn"):
        return BaselineCNN(
            num_classes=num_classes,
            in_channels=in_channels,
        )
    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available architectures: ['PaperGenericCNN', 'BaselineCNN']"
        )
