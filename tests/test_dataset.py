"""
NeuroScan AI — Dataset and DataLoader Unit Tests.
"""
import csv
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from neuroscan.data.config import DataConfig
from neuroscan.data.dataloader import create_dataloaders
from neuroscan.data.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    BrainTumorMRIDataset,
    get_default_project_root,
)
from neuroscan.data.transforms import get_eval_transforms, get_train_transforms


class TestBrainTumorMRIDataset(unittest.TestCase):
    """Unit tests for BrainTumorMRIDataset, transforms, and DataLoaders."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = get_default_project_root()
        cls.manifest_path = cls.project_root / "data" / "processed" / "dataset_manifest.csv"

        # Load excluded test paths from manifest for leakage assertion tests
        cls.excluded_rel_paths = set()
        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == "Testing" and row["eval_status"] == "exclude":
                    cls.excluded_rel_paths.add(row["rel_path"])

    def test_class_to_idx_mapping(self):
        """Verify deterministic class-to-index mapping."""
        expected = {
            "glioma": 0,
            "meningioma": 1,
            "notumor": 2,
            "pituitary": 3,
        }
        self.assertEqual(CLASS_TO_IDX, expected)
        self.assertEqual(len(IDX_TO_CLASS), 4)
        for name, idx in expected.items():
            self.assertEqual(IDX_TO_CLASS[idx], name)

    def test_split_sample_counts_and_exclusion(self):
        """Assert exact sample counts (5,600 training, 1,486 eval) and zero leaked test samples."""
        train_dataset = BrainTumorMRIDataset(
            split="Training",
            manifest_path=self.manifest_path,
            project_root=self.project_root,
        )
        eval_dataset = BrainTumorMRIDataset(
            split="Testing",
            manifest_path=self.manifest_path,
            project_root=self.project_root,
        )

        self.assertEqual(len(train_dataset), 5600, "Training set must contain exactly 5,600 samples.")
        self.assertEqual(len(eval_dataset), 1486, "Clean evaluation set must contain exactly 1,486 samples.")
        self.assertEqual(len(self.excluded_rel_paths), 114, "Manifest must contain exactly 114 excluded test paths.")

        # Ensure no excluded test image is present in eval_dataset
        eval_rel_paths = {sample[2] for sample in eval_dataset.samples}
        leaked_in_eval = eval_rel_paths.intersection(self.excluded_rel_paths)
        self.assertEqual(
            len(leaked_in_eval),
            0,
            f"Found leaked test images in evaluation dataset: {leaked_in_eval}"
        )

    def test_tensor_shapes_types_and_rgb_conversion(self):
        """Verify output tensor shapes, float32 dtype, label range, and 3-channel RGB conversion."""
        transform = get_eval_transforms(img_size=224)
        eval_dataset = BrainTumorMRIDataset(
            split="Testing",
            manifest_path=self.manifest_path,
            project_root=self.project_root,
            transform=transform,
        )

        # Inspect first sample
        img_tensor, label = eval_dataset[0]

        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertEqual(img_tensor.shape, torch.Size([3, 224, 224]))
        self.assertEqual(img_tensor.dtype, torch.float32)
        self.assertIsInstance(label, int)
        self.assertTrue(0 <= label <= 3)

        # Specifically verify an image originally stored in single-channel grayscale (mode 'L')
        # Find first sample that was mode 'L'
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            grayscale_sample = None
            for row in reader:
                if row["split"] == "Testing" and row["eval_status"] == "include":
                    abs_p = self.project_root / row["rel_path"]
                    with Image.open(abs_p) as im:
                        if im.mode == "L":
                            grayscale_sample = row["rel_path"]
                            break

        if grayscale_sample:
            # Find this sample in dataset
            match_idx = next(i for i, s in enumerate(eval_dataset.samples) if s[2] == grayscale_sample)
            gray_tensor, _ = eval_dataset[match_idx]
            self.assertEqual(gray_tensor.shape, torch.Size([3, 224, 224]), "Grayscale image must be converted to 3 RGB channels.")

    def test_deterministic_eval_transforms(self):
        """Verify that evaluation transform is strictly deterministic across repeated invocations."""
        eval_transform = get_eval_transforms(img_size=224)
        dummy_img = Image.new("RGB", (300, 400), color=(128, 64, 32))

        t1 = eval_transform(dummy_img)
        t2 = eval_transform(dummy_img)

        self.assertTrue(torch.equal(t1, t2), "Evaluation transform must be bit-identical across runs.")

    def test_invalid_split_raises_value_error(self):
        """Verify that an invalid split string raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            BrainTumorMRIDataset(
                split="Validation",
                manifest_path=self.manifest_path,
                project_root=self.project_root,
            )
        self.assertIn("Invalid split", str(ctx.exception))

    def test_missing_manifest_raises_file_not_found(self):
        """Verify that a missing manifest path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            BrainTumorMRIDataset(
                split="Training",
                manifest_path="nonexistent_manifest.csv",
                project_root=self.project_root,
            )

    def test_malformed_manifest_raises_value_error(self):
        """Verify that a manifest with missing required columns raises ValueError."""
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write("rel_path,label\nfoo.jpg,glioma\n")
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                BrainTumorMRIDataset(
                    split="Training",
                    manifest_path=tmp_path,
                    project_root=self.project_root,
                )
            self.assertIn("missing required columns", str(ctx.exception))
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_dataloaders_construction_and_batch_smoke_check(self):
        """Construct DataLoaders and perform a CPU-only single batch smoke test."""
        config = DataConfig(
            img_size=224,
            batch_size=16,
            num_workers=0,
            pin_memory=False,
            manifest_path="data/processed/dataset_manifest.csv",
            seed=42,
        )

        train_loader, eval_loader = create_dataloaders(
            config=config,
            project_root=self.project_root,
        )

        self.assertEqual(len(train_loader.dataset), 5600)
        self.assertEqual(len(eval_loader.dataset), 1486)
        self.assertTrue(train_loader.drop_last is False)

        # Smoke check 1: Pull one batch from train_loader
        train_iter = iter(train_loader)
        batch_images, batch_labels = next(train_iter)

        self.assertEqual(batch_images.shape, torch.Size([16, 3, 224, 224]))
        self.assertEqual(batch_images.dtype, torch.float32)
        self.assertEqual(batch_labels.shape, torch.Size([16]))
        self.assertEqual(batch_labels.dtype, torch.int64)

        # Smoke check 2: Pull one batch from eval_loader
        eval_iter = iter(eval_loader)
        eval_images, eval_labels = next(eval_iter)

        self.assertEqual(eval_images.shape, torch.Size([16, 3, 224, 224]))
        self.assertEqual(eval_images.dtype, torch.float32)
        self.assertEqual(eval_labels.shape, torch.Size([16]))
        self.assertEqual(eval_labels.dtype, torch.int64)

    def test_security_valid_path_accepted(self):
        """Verify that a legitimate image path inside data/raw/ is accepted."""
        dataset = BrainTumorMRIDataset(
            split="Training",
            manifest_path=self.manifest_path,
            project_root=self.project_root,
        )
        self.assertGreater(len(dataset), 0)

    def test_security_path_traversal_rejected(self):
        """Verify that a path traversal attempt (e.g. ../../../../etc/hosts) is rejected with ValueError."""
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write(
                "rel_path,split,label,file_sha256,pixel_sha256,is_duplicate,is_cross_split_duplicate,eval_status,exclusion_reason\n"
                "../../../../etc/hosts,Training,glioma,hash1,hash2,False,False,include,none\n"
            )
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                BrainTumorMRIDataset(
                    split="Training",
                    manifest_path=tmp_path,
                    project_root=self.project_root,
                )
            self.assertIn("Path traversal detected", str(ctx.exception))
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_security_absolute_path_outside_dataset_rejected(self):
        """Verify that an absolute path outside data/raw/ (e.g. /etc/hosts) is rejected with ValueError."""
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write(
                "rel_path,split,label,file_sha256,pixel_sha256,is_duplicate,is_cross_split_duplicate,eval_status,exclusion_reason\n"
                "/etc/hosts,Training,glioma,hash1,hash2,False,False,include,none\n"
            )
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                BrainTumorMRIDataset(
                    split="Training",
                    manifest_path=tmp_path,
                    project_root=self.project_root,
                )
            self.assertIn("Path traversal detected", str(ctx.exception))
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_security_symlink_escaping_dataset_rejected(self):
        """Verify that a symlink inside data/raw/ pointing outside data/raw/ is rejected with ValueError."""
        raw_root = (self.project_root / "data" / "raw").resolve()
        with tempfile.TemporaryDirectory(dir=raw_root) as tmp_dir:
            symlink_path = Path(tmp_dir) / "escape_symlink.jpg"
            try:
                os.symlink("/etc/hosts", symlink_path)
            except (OSError, NotImplementedError):
                self.skipTest("Symlinks not supported in environment.")

            rel_symlink = symlink_path.relative_to(self.project_root).as_posix()

            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp_csv:
                tmp_csv.write(
                    "rel_path,split,label,file_sha256,pixel_sha256,is_duplicate,is_cross_split_duplicate,eval_status,exclusion_reason\n"
                    f"{rel_symlink},Training,glioma,hash1,hash2,False,False,include,none\n"
                )
                tmp_csv_path = Path(tmp_csv.name)

            try:
                with self.assertRaises(ValueError) as ctx:
                    BrainTumorMRIDataset(
                        split="Training",
                        manifest_path=tmp_csv_path,
                        project_root=self.project_root,
                    )
                self.assertIn("Path traversal detected", str(ctx.exception))
            finally:
                tmp_csv_path.unlink(missing_ok=True)

    def test_reproducibility_same_seed_identical_first_epoch(self):
        """Verify that two independently created training loaders with identical seeds produce identical batches."""
        cfg1 = DataConfig(batch_size=8, seed=42)
        cfg2 = DataConfig(batch_size=8, seed=42)

        loader1, _ = create_dataloaders(config=cfg1, project_root=self.project_root)
        loader2, _ = create_dataloaders(config=cfg2, project_root=self.project_root)

        imgs1, lbls1 = next(iter(loader1))
        imgs2, lbls2 = next(iter(loader2))

        self.assertTrue(torch.equal(lbls1, lbls2), "Sample ordering must be identical for identical seed.")
        self.assertTrue(torch.equal(imgs1, imgs2), "Augmented image tensors must be identical for identical seed.")

    def test_reproducibility_different_seeds_produce_different_outputs(self):
        """Verify that different seeds produce different sample ordering and random augmentations."""
        cfg1 = DataConfig(batch_size=8, seed=42)
        cfg2 = DataConfig(batch_size=8, seed=999)

        loader1, _ = create_dataloaders(config=cfg1, project_root=self.project_root)
        loader2, _ = create_dataloaders(config=cfg2, project_root=self.project_root)

        imgs1, lbls1 = next(iter(loader1))
        imgs2, lbls2 = next(iter(loader2))

        self.assertFalse(
            torch.equal(lbls1, lbls2) and torch.equal(imgs1, imgs2),
            "Different seeds should produce different training streams."
        )

    def test_eval_loader_unaffected_by_training_randomness(self):
        """Verify evaluation outputs remain bit-identical regardless of training iterations."""
        cfg = DataConfig(batch_size=8, seed=42)
        train_loader, eval_loader = create_dataloaders(config=cfg, project_root=self.project_root)

        # Baseline eval batch
        eval_imgs1, eval_lbls1 = next(iter(eval_loader))

        # Consume multiple training batches that advance random streams
        train_iter = iter(train_loader)
        for _ in range(5):
            _ = next(train_iter)

        # Second eval batch from a fresh eval loader
        _, eval_loader2 = create_dataloaders(config=cfg, project_root=self.project_root)
        eval_imgs2, eval_lbls2 = next(iter(eval_loader2))

        self.assertTrue(torch.equal(eval_lbls1, eval_lbls2))
        self.assertTrue(torch.equal(eval_imgs1, eval_imgs2))


if __name__ == "__main__":
    unittest.main()

