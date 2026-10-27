"""
NeuroScan AI — Data Configuration Module.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Configuration dataclass for dataset and DataLoader settings."""
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = False
    manifest_path: str = "data/processed/dataset_manifest.csv"
    seed: int = 42


def load_data_config(config_path: Optional[Union[str, Path]] = None) -> DataConfig:
    """
    Load data configuration from a YAML file, falling back to sensible defaults.
    """
    if config_path is None:
        # Check standard location relative to project root
        project_root = Path(__file__).resolve().parents[3]
        default_yaml = project_root / "configs" / "data_config.yaml"
        if default_yaml.is_file():
            config_path = default_yaml

    if config_path is not None:
        path = Path(config_path)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and "data" in data:
                    cfg_dict = data["data"]
                    return DataConfig(
                        img_size=cfg_dict.get("img_size", 224),
                        batch_size=cfg_dict.get("batch_size", 16),
                        num_workers=cfg_dict.get("num_workers", 0),
                        pin_memory=cfg_dict.get("pin_memory", False),
                        manifest_path=cfg_dict.get("manifest_path", "data/processed/dataset_manifest.csv"),
                        seed=cfg_dict.get("seed", 42),
                    )

    return DataConfig()

