"""
NeuroScan AI — Unit Tests for Training Pipeline, Checkpointing, and Early Stopping.
"""
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

from PIL import Image
import torch
import torch.nn as nn

from neuroscan.data.transforms import get_train_transforms
from neuroscan.training.train import (
    EarlyStopping,
    load_training_config,
    run_training,
    save_checkpoint,
    set_all_seeds,
)


class TestTrainingComponents(unittest.TestCase):
    """Test suite for early stopping, checkpoint persistence, config validation, and dry run."""

    def test_early_stopping_max_mode_trigger(self):
        """Verify EarlyStopping in max mode triggers early_stop after patience epochs of stagnation."""
        stopper = EarlyStopping(patience=3, mode="max", min_delta=0.01)

        # Epoch 1: Initial baseline
        improved = stopper.step(score=0.80, epoch=1)
        self.assertTrue(improved)
        self.assertEqual(stopper.best_score, 0.80)
        self.assertEqual(stopper.counter, 0)
        self.assertFalse(stopper.early_stop)

        # Epoch 2: Improvement below min_delta (0.805 - 0.80 = 0.005 < 0.01)
        improved = stopper.step(score=0.805, epoch=2)
        self.assertFalse(improved)
        self.assertEqual(stopper.counter, 1)
        self.assertFalse(stopper.early_stop)

        # Epoch 3: Regression
        improved = stopper.step(score=0.79, epoch=3)
        self.assertFalse(improved)
        self.assertEqual(stopper.counter, 2)
        self.assertFalse(stopper.early_stop)

        # Epoch 4: Stagnation reaching patience (counter = 3)
        improved = stopper.step(score=0.80, epoch=4)
        self.assertFalse(improved)
        self.assertEqual(stopper.counter, 3)
        self.assertTrue(stopper.early_stop)

    def test_early_stopping_reset_on_improvement(self):
        """Verify EarlyStopping resets stagnation counter when significant improvement occurs."""
        stopper = EarlyStopping(patience=3, mode="max", min_delta=0.01)
        stopper.step(score=0.80, epoch=1)
        stopper.step(score=0.79, epoch=2)
        self.assertEqual(stopper.counter, 1)

        # Epoch 3: Jump to 0.85 (0.05 > 0.01)
        improved = stopper.step(score=0.85, epoch=3)
        self.assertTrue(improved)
        self.assertEqual(stopper.best_score, 0.85)
        self.assertEqual(stopper.counter, 0)
        self.assertFalse(stopper.early_stop)

    def test_checkpoint_roundtrip_persistence(self):
        """Verify save_checkpoint writes all required states and reproduction metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "checkpoints" / "test_model.pt"

            model = nn.Sequential(nn.Linear(4, 2), nn.ReLU())
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            val_metrics = {
                "loss": 0.45,
                "accuracy": 0.88,
                "macro_f1": 0.875,
                "per_class": {},
            }
            config = {"experiment": {"name": "test_exp", "seed": 42}}

            save_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=5,
                best_val_macro_f1=0.875,
                val_metrics=val_metrics,
                seed=42,
                config=config,
            )

            self.assertTrue(ckpt_path.is_file(), "Checkpoint file must exist on disk.")

            # Load and verify contents
            loaded = torch.load(ckpt_path, weights_only=False)
            self.assertEqual(loaded["epoch"], 5)
            self.assertEqual(loaded["best_val_macro_f1"], 0.875)
            self.assertEqual(loaded["seed"], 42)
            self.assertIn("model_state_dict", loaded)
            self.assertIn("optimizer_state_dict", loaded)
            self.assertIn("class_to_idx", loaded)
            self.assertIn("idx_to_class", loaded)
            self.assertEqual(loaded["class_to_idx"]["glioma"], 0)
            self.assertEqual(loaded["class_to_idx"]["pituitary"], 3)
            self.assertEqual(loaded["config"]["experiment"]["name"], "test_exp")

    def test_train_config_loading_and_paper_parameters(self):
        """Verify train_config.yaml loads with paper hyperparameters."""
        config = load_training_config()
        self.assertEqual(config["experiment"]["seed"], 42)
        self.assertEqual(config["experiment"]["device"], "cpu")

        # Paper hyperparameters
        self.assertEqual(config["data"]["img_size"], 256)
        self.assertEqual(config["data"]["batch_size"], 16)
        self.assertEqual(config["data"]["normalize_mode"], "rescale_only")
        self.assertEqual(config["model"]["architecture"], "PaperGenericCNN")
        self.assertEqual(config["training"]["epochs"], 10)
        self.assertEqual(config["training"]["optimizer"]["name"], "Adam")
        self.assertEqual(config["training"]["optimizer"]["learning_rate"], 0.0001)

        # Clean test quarantine policy
        self.assertTrue(config["evaluation_policy"]["clean_test_quarantine"])

    def test_dry_run_pipeline_execution(self):
        """
        Verify that dry-run sets up PaperGenericCNN, validates dataset quarantine,
        and performs zero training epochs or parameter updates.
        """
        report = run_training(dry_run=True)

        self.assertEqual(report["status"], "success")
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["architecture"], "PaperGenericCNN")
        self.assertEqual(report["num_parameters"], 651044)
        self.assertEqual(report["train_samples"], 4480)
        self.assertEqual(report["val_samples"], 1120)
        self.assertEqual(report["testing_samples_accessed"], 0)
        self.assertTrue(report["testing_split_quarantined"])
        self.assertEqual(report["optimizer"], "Adam")
        self.assertEqual(report["epochs_run"], 0)
        self.assertEqual(report["optimization_steps_run"], 0)
        self.assertEqual(report["dummy_forward_shape"], [2, 4])
        self.assertTrue(report["reproducibility"]["model_seeded_before_dataloader_iter"])

    def test_brightness_mode_validation(self):
        """Verify brightness_mode raises ValueError on invalid mode and succeeds on valid modes."""
        with self.assertRaises(ValueError):
            get_train_transforms(brightness_mode="invalid_mode")

        t_paper = get_train_transforms(brightness_mode="paper_literal")
        t_jitter = get_train_transforms(brightness_mode="safe_jitter")
        t_none = get_train_transforms(brightness_mode="none")
        self.assertIsNotNone(t_paper)
        self.assertIsNotNone(t_jitter)
        self.assertIsNotNone(t_none)

    @patch("neuroscan.data.transforms.random.uniform")
    def test_paper_literal_brightness_bounds_mock(self, mock_uniform):
        """Verify paper_literal samples factor strictly from [0.0, 0.7] using a deterministic mock."""
        mock_uniform.return_value = 0.0
        t = get_train_transforms(brightness_mode="paper_literal")
        img = Image.new("RGB", (256, 256), color=(100, 100, 100))

        out_zero = t(img)
        mock_uniform.assert_called_once_with(0.0, 0.7)
        self.assertTrue(torch.all(out_zero == 0.0), "Factor 0.0 must yield all-zero pixel tensor.")

        mock_uniform.reset_mock()
        mock_uniform.return_value = 0.7
        out_scaled = t(img)
        mock_uniform.assert_called_once_with(0.0, 0.7)
        self.assertFalse(torch.all(out_scaled == 0.0), "Factor 0.7 must preserve non-zero scaled pixels.")


if __name__ == "__main__":
    unittest.main()
