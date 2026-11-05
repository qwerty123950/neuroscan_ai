"""
NeuroScan AI — Unit Tests for Model Architectures.
"""
import unittest

import torch

from neuroscan.models import BaselineCNN, PaperGenericCNN, get_model


class TestModelArchitectures(unittest.TestCase):
    """Test suite for BaselineCNN and PaperGenericCNN architectures."""

    def test_baseline_cnn_forward_and_params(self):
        """Verify BaselineCNN forward pass and parameter counts."""
        model = BaselineCNN(num_classes=4, in_channels=3)
        model.eval()
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, torch.Size([2, 4]))
        param_counts = model.count_parameters()
        self.assertEqual(param_counts["total"], 422308)

    def test_paper_generic_cnn_forward_and_params(self):
        """Verify PaperGenericCNN forward pass on 256x256 inputs and parameter counts."""
        model = PaperGenericCNN(num_classes=4, in_channels=3, dropout=0.5)
        model.eval()
        dummy_input = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, torch.Size([2, 4]))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all().item())

        param_counts = model.count_parameters()
        total = param_counts["total"]
        # Expected: 651,044 parameters
        self.assertEqual(total, 651044)
        self.assertEqual(param_counts["trainable"], total)

    def test_paper_generic_cnn_eval_determinism(self):
        """Verify PaperGenericCNN in eval() mode is deterministic."""
        model = PaperGenericCNN(num_classes=4, in_channels=3, dropout=0.5)
        model.eval()
        dummy_input = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out1 = model(dummy_input)
            out2 = model(dummy_input)
        self.assertTrue(torch.equal(out1, out2))

    def test_paper_generic_cnn_train_dropout_stochasticity(self):
        """Verify PaperGenericCNN in train() mode has active dropout."""
        model = PaperGenericCNN(num_classes=4, in_channels=3, dropout=0.5)
        model.train()
        dummy_input = torch.randn(2, 3, 256, 256)
        out1 = model(dummy_input)
        out2 = model(dummy_input)
        self.assertFalse(torch.equal(out1, out2))

    def test_model_factory_get_model(self):
        """Verify get_model factory instantiates correct models."""
        m1 = get_model("PaperGenericCNN")
        self.assertIsInstance(m1, PaperGenericCNN)

        m2 = get_model("BaselineCNN")
        self.assertIsInstance(m2, BaselineCNN)

        with self.assertRaises(ValueError):
            get_model("NonExistentModel")


if __name__ == "__main__":
    unittest.main()
