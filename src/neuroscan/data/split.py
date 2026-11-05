"""
NeuroScan AI — Stratified Dataset Splitting Module.

Provides deterministic, stratified splitting of the training dataset
for model validation and development, while keeping the clean testing
set strictly quarantined for final benchmark evaluation.
"""
from collections import defaultdict
from pathlib import Path
import random
from typing import List, Optional, Tuple, Union

from neuroscan.data.config import DataConfig, load_data_config
from neuroscan.data.dataset import BrainTumorMRIDataset, get_default_project_root
from neuroscan.data.transforms import get_eval_transforms, get_train_transforms


def get_stratified_split_indices(
    dataset: BrainTumorMRIDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Compute deterministic stratified train and validation indices from a dataset.

    Guarantees:
    - Enforces that input dataset split is 'Training' to prevent evaluation set contamination.
    - Preserves class proportions in both train and validation subsets.
    - Zero path/index overlap between the two splits.
    - Deterministic across runs using an isolated random.Random(seed) instance.

    Returns:
        (train_indices, val_indices) sorted lists of integer sample indices.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0.0 and 1.0, got {val_ratio}")

    if dataset.split != "Training":
        raise ValueError(
            f"Validation splits must be drawn strictly from the 'Training' split, "
            f"got dataset with split='{dataset.split}'. The clean Testing set must remain isolated."
        )

    # Group sample indices by class label
    class_to_indices = defaultdict(list)
    for idx, (_, label_idx, _) in enumerate(dataset.samples):
        class_to_indices[label_idx].append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for label_idx in sorted(class_to_indices.keys()):
        indices = list(class_to_indices[label_idx])
        indices.sort()  # Ensure deterministic input order
        rng.shuffle(indices)

        n_train = int(round(len(indices) * (1.0 - val_ratio)))
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:])

    train_indices.sort()
    val_indices.sort()

    # Integrity assertions
    assert len(set(train_indices).intersection(set(val_indices))) == 0, "Train and validation indices overlap!"
    assert len(train_indices) + len(val_indices) == len(dataset), "Total split indices do not match dataset length!"

    return train_indices, val_indices


def create_train_val_datasets(
    config: Optional[DataConfig] = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[BrainTumorMRIDataset, BrainTumorMRIDataset]:
    """
    Factory creating stratified training (80%) and validation (20%) datasets.

    Guarantees:
    - Both datasets drawn from the 5,600 included Training images.
    - Training dataset uses training transforms (with modest augmentation).
    - Validation dataset uses deterministic evaluation transforms.
    - Clean evaluation set (1,486 images) remains untouched.
    """
    if config is None:
        config = load_data_config()

    root = Path(project_root).resolve() if project_root else get_default_project_root()

    # 1. Base full training dataset (untransformed for indexing)
    base_train = BrainTumorMRIDataset(
        split="Training",
        manifest_path=config.manifest_path,
        project_root=root,
    )

    # 2. Compute stratified indices
    train_idx, val_idx = get_stratified_split_indices(
        dataset=base_train,
        val_ratio=val_ratio,
        seed=seed,
    )

    # 3. Construct partitioned datasets with appropriate transforms
    normalize_mode = getattr(config, "normalize_mode", "rescale_only")
    brightness_mode = getattr(config, "brightness_mode", "paper_literal")
    train_dataset = BrainTumorMRIDataset(
        split="Training",
        manifest_path=config.manifest_path,
        project_root=root,
        transform=get_train_transforms(
            img_size=config.img_size,
            normalize_mode=normalize_mode,
            brightness_mode=brightness_mode,
        ),
        sample_indices=train_idx,
    )

    val_dataset = BrainTumorMRIDataset(
        split="Training",
        manifest_path=config.manifest_path,
        project_root=root,
        transform=get_eval_transforms(img_size=config.img_size, normalize_mode=normalize_mode),
        sample_indices=val_idx,
    )

    return train_dataset, val_dataset

