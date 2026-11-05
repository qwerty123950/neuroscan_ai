"""
NeuroScan AI — Unit Tests for Evaluation Metrics and Safe Zero-Division.
"""
import unittest

import torch

from neuroscan.evaluation.metrics import (
    DEFAULT_CLASS_NAMES,
    MetricTracker,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_metrics_from_confusion_matrix,
)


class TestMetrics(unittest.TestCase):
    """Test suite for confusion matrix, precision, recall, F1, and zero-division safety."""

    def test_known_confusion_matrix_calculation(self):
        """Verify metric calculation on a known, manually calculated 4x4 confusion matrix."""
        # Rows: True [glioma, meningioma, notumor, pituitary]
        # Cols: Pred [glioma, meningioma, notumor, pituitary]
        matrix = [
            [10,  0,  0,  0],  # Class 0: TP=10, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
            [ 0,  8,  2,  0],  # Class 1: TP=8,  FP=2, FN=2 -> P=0.8, R=0.8, F1=0.8
            [ 0,  2,  8,  0],  # Class 2: TP=8,  FP=2, FN=2 -> P=0.8, R=0.8, F1=0.8
            [ 0,  0,  0, 10],  # Class 3: TP=10, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
        ]
        metrics = compute_metrics_from_confusion_matrix(matrix, class_names=DEFAULT_CLASS_NAMES)

        self.assertEqual(metrics["total_samples"], 40)
        self.assertAlmostEqual(metrics["accuracy"], 0.9, places=5)
        self.assertAlmostEqual(metrics["macro_precision"], 0.9, places=5)
        self.assertAlmostEqual(metrics["macro_recall"], 0.9, places=5)
        self.assertAlmostEqual(metrics["macro_f1"], 0.9, places=5)

        # Per-class checks
        self.assertAlmostEqual(metrics["per_class"]["glioma"]["f1"], 1.0, places=5)
        self.assertAlmostEqual(metrics["per_class"]["meningioma"]["f1"], 0.8, places=5)
        self.assertAlmostEqual(metrics["per_class"]["notumor"]["f1"], 0.8, places=5)
        self.assertAlmostEqual(metrics["per_class"]["pituitary"]["f1"], 1.0, places=5)

    def test_zero_division_no_predictions_for_class(self):
        """Verify that a class receiving 0 predictions results in 0.0 precision/f1 without NaN or errors."""
        matrix = [
            [10, 0,  0,  0],
            [ 5, 0,  5,  0],  # Meningioma: 10 true samples, but 0 predicted as meningioma (col 1 is 0)
            [ 0, 0, 10,  0],
            [ 0, 0,  0, 10],
        ]
        metrics = compute_metrics_from_confusion_matrix(matrix, class_names=DEFAULT_CLASS_NAMES)

        # Meningioma: TP=0, FP=0, FN=10
        men = metrics["per_class"]["meningioma"]
        self.assertEqual(men["precision"], 0.0)
        self.assertEqual(men["recall"], 0.0)
        self.assertEqual(men["f1"], 0.0)
        self.assertEqual(men["support"], 10)

        # Macro-F1 must be a valid, finite float in [0.0, 1.0]
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertLessEqual(metrics["macro_f1"], 1.0)
        self.assertFalse(torch.isnan(torch.tensor(metrics["macro_f1"])))

    def test_zero_division_all_empty(self):
        """Verify that an empty confusion matrix (0 samples) produces 0.0 for all metrics cleanly."""
        matrix = [[0] * 4 for _ in range(4)]
        metrics = compute_metrics_from_confusion_matrix(matrix, class_names=DEFAULT_CLASS_NAMES)

        self.assertEqual(metrics["total_samples"], 0)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["macro_f1"], 0.0)
        self.assertEqual(metrics["macro_precision"], 0.0)
        self.assertEqual(metrics["macro_recall"], 0.0)

    def test_compute_confusion_matrix_with_logits_tensor(self):
        """Verify confusion matrix computation directly from 2D raw logits tensor."""
        targets = torch.tensor([0, 1, 2, 3])
        # Logits giving correct class the highest value
        logits = torch.tensor([
            [5.0, 1.0, 0.0, 0.0],
            [1.0, 6.0, 2.0, 0.0],
            [0.0, 1.0, 4.0, 2.0],
            [0.0, 0.0, 1.0, 7.0],
        ])
        matrix = compute_confusion_matrix(targets, logits, num_classes=4)
        expected = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        self.assertEqual(matrix, expected)

    def test_metric_tracker_batch_accumulation(self):
        """Verify MetricTracker accumulates loss and predictions across multiple mini-batches."""
        tracker = MetricTracker(class_names=DEFAULT_CLASS_NAMES)

        # Batch 1: 2 samples, loss 0.5
        b1_targets = torch.tensor([0, 1])
        b1_preds = torch.tensor([0, 1])
        tracker.update(loss=0.5, preds=b1_preds, targets=b1_targets, batch_size=2)

        # Batch 2: 2 samples, loss 0.3
        b2_targets = torch.tensor([2, 3])
        b2_preds = torch.tensor([2, 0])  # One error: sample 3 predicted as 0
        tracker.update(loss=0.3, preds=b2_preds, targets=b2_targets, batch_size=2)

        metrics = tracker.compute()
        # Average loss: (0.5 * 2 + 0.3 * 2) / 4 = 0.4
        self.assertAlmostEqual(metrics["loss"], 0.4, places=5)
        self.assertEqual(metrics["total_samples"], 4)
        # 3 correct out of 4: accuracy = 0.75
        self.assertAlmostEqual(metrics["accuracy"], 0.75, places=5)

    def test_mismatched_lengths_raise_value_error(self):
        """Verify ValueError is raised when targets and preds lengths do not match."""
        with self.assertRaises(ValueError):
            compute_confusion_matrix([0, 1], [0])


if __name__ == "__main__":
    unittest.main()

