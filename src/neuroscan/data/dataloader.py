"""
NeuroScan AI — DataLoader Construction Module.
"""
from pathlib import Path
import random
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from neuroscan.data.config import DataConfig, load_data_config
from neuroscan.data.dataset import BrainTumorMRIDataset, get_default_project_root
from neuroscan.data.transforms import get_eval_transforms, get_train_transforms


class ReproducibleDataLoader(DataLoader):
    """
    DataLoader that guarantees reproducible random augmentation sequences
    by deriving epoch-level seeds from its generator under num_workers=0.
    """

    def __iter__(self):
        if self.generator is not None:
            # Derive deterministic epoch seed from DataLoader generator for main-process transforms
            epoch_seed = int(torch.empty((), dtype=torch.int64).random_(generator=self.generator).item() % (2**32))
            torch.manual_seed(epoch_seed)
            random.seed(epoch_seed)
        return super().__iter__()


def seed_worker(worker_id: int) -> None:
    """
    Seed worker subprocess RNGs appropriately based on PyTorch's initial seed.
    Guarantees deterministic multi-worker augmentation streams if num_workers > 0.
    """
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_dataloaders(
    config: Optional[DataConfig] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct training and evaluation DataLoaders driven strictly by the verified manifest.

    Guarantees:
    - Training loader uses only the 5,600 included Training records (shuffle=True).
    - Evaluation loader uses only the 1,486 included Testing records (shuffle=False).
    - Conservative worker count (default 0) tailored for a laptop with 7.4 GiB RAM.
    - Deterministic evaluation transform pipeline (no random augmentations).
    - Excluded cross-split duplicate test images are strictly barred from entering evaluation.
    - Shuffling and random augmentations are reproducible across runs with the same seed.
    """
    if config is None:
        config = load_data_config()

    root = Path(project_root).resolve() if project_root else get_default_project_root()

    # Preprocessing pipelines
    train_transform = get_train_transforms(img_size=config.img_size)
    eval_transform = get_eval_transforms(img_size=config.img_size)

    # Dataset instances
    train_dataset = BrainTumorMRIDataset(
        split="Training",
        manifest_path=config.manifest_path,
        project_root=root,
        transform=train_transform,
    )
    eval_dataset = BrainTumorMRIDataset(
        split="Testing",
        manifest_path=config.manifest_path,
        project_root=root,
        transform=eval_transform,
    )

    # Shuffling generator with reproducible seed
    train_generator = None
    if config.seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(config.seed)

    train_loader = ReproducibleDataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        generator=train_generator,
        worker_init_fn=seed_worker if config.seed is not None else None,
    )

    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, eval_loader

