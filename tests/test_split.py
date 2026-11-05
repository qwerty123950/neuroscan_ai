"""
NeuroScan AI — Unit Tests for Stratified Validation Splitting.
"""
from collections import Counter
import unittest

import torch
from torchvision import transforms

from neuroscan.data.dataset import BrainTumorMRIDataset, get_default_project_root
from neuroscan.data.split import create_train_val_datasets, get_stratified_split_indices


class TestDatasetSplitting(unittest.TestCase):
    """Test suite for deterministic stratified train/val splitting."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = get_default_project_root()
        cls.train_dataset = BrainTumorMRIDataset(
            split="Training",
            project_root=cls.project_root,
        )

    def test_rejection_of_testing_split_for_validation(self):
        """Verify that attempting to split the clean Testing dataset raises ValueError."""
        test_dataset = BrainTumorMRIDataset(
            split="Testing",
            project_root=self.project_root,
        )
        with self.assertRaises(ValueError) as ctx:
            get_stratified_split_indices(dataset=test_dataset, val_ratio=0.2)
        self.assertIn("must be drawn strictly from the 'Training' split", str(ctx.exception))

    def test_stratified_split_counts_and_balance(self):
        """Assert exact 80/20 counts (4,480 train / 1,120 val) and exact per-class balance (1,120/280)."""
        train_idx, val_idx = get_stratified_split_indices(
            dataset=self.train_dataset,
            val_ratio=0.2,
            seed=42,
        )

        self.assertEqual(len(train_idx), 4480, "Train split must contain exactly 4,480 samples (80% of 5,600).")
        self.assertEqual(len(val_idx), 1120, "Val split must contain exactly 1,120 samples (20% of 5,600).")

        # Zero index overlap
        self.assertTrue(set(train_idx).isdisjoint(set(val_idx)), "Train and validation indices must not overlap.")

        # Class counts in train
        train_labels = [self.train_dataset.samples[i][1] for i in train_idx]
        train_class_counts = Counter(train_labels)
        for label_idx in range(4):
            self.assertEqual(
                train_class_counts[label_idx],
                1120,
                f"Class {label_idx} in training split must have exactly 1,120 samples."
            )

        # Class counts in val
        val_labels = [self.train_dataset.samples[i][1] for i in val_idx]
        val_class_counts = Counter(val_labels)
        for label_idx in range(4):
            self.assertEqual(
                val_class_counts[label_idx],
                280,
                f"Class {label_idx} in validation split must have exactly 280 samples."
            )

    def test_create_train_val_datasets_factory(self):
        """Verify factory returns datasets with correct sample counts, transforms, and zero leakage."""
        train_ds, val_ds = create_train_val_datasets(
            val_ratio=0.2,
            seed=42,
            project_root=self.project_root,
        )

        self.assertEqual(len(train_ds), 4480)
        self.assertEqual(len(val_ds), 1120)

        # Verify transforms
        train_has_flip = any(isinstance(t, transforms.RandomHorizontalFlip) for t in train_ds.transform.transforms)
        val_has_flip = any(isinstance(t, transforms.RandomHorizontalFlip) for t in val_ds.transform.transforms)
        self.assertTrue(train_has_flip, "Train dataset must use training transforms with augmentation.")
        self.assertFalse(val_has_flip, "Validation dataset must use deterministic evaluation transforms.")

        # Verify zero path overlap between train and val
        train_paths = {s[0] for s in train_ds.samples}
        val_paths = {s[0] for s in val_ds.samples}
        self.assertTrue(train_paths.isdisjoint(val_paths), "Train and validation image paths must be completely disjoint.")

    def test_deterministic_split_reproducibility(self):
        """Verify that fixed seed produces identical splits and different seeds produce different splits."""
        t1, v1 = get_stratified_split_indices(self.train_dataset, val_ratio=0.2, seed=42)
        t2, v2 = get_stratified_split_indices(self.train_dataset, val_ratio=0.2, seed=42)
        t3, v3 = get_stratified_split_indices(self.train_dataset, val_ratio=0.2, seed=999)

        self.assertEqual(t1, t2)
        self.assertEqual(v1, v2)
        self.assertNotEqual(t1, t3)
        self.assertNotEqual(v1, v3)


if __name__ == "__main__":
    unittest.main()

