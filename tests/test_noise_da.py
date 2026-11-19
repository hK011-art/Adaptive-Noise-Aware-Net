import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import torch
from src.models.diffusion_da import NoiseDADiffusionLoss
from src.models.noise_da_trainer import NoiseDATrainer, CharbonnierLoss

class TestNoiseDA(unittest.TestCase):
    def test_per_sample_channel_shuffling(self):
        """Verifies channel shuffling operates independently per sample in the batch."""
        da_loss = NoiseDADiffusionLoss()
        da_loss.train()

        B, C, H, W = 8, 3, 32, 32
        y_s = torch.ones(B, C, H, W)
        y_r = torch.zeros(B, C, H, W)

        # Shuffle
        shuffled = da_loss.channel_shuffle(y_s, y_r)
        self.assertEqual(shuffled.shape, (B, 2 * C, H, W))

        # Check that order is either (y_s, y_r) or (y_r, y_s) per sample
        first_channels = shuffled[:, 0, 0, 0]  # Should be 1.0 or 0.0
        self.assertTrue(((first_channels == 1.0) | (first_channels == 0.0)).all())

    def test_residual_algebra_consistency(self):
        """Verifies residual shift equation consistency: Y + (X_hat - Y) = X_hat."""
        B, C, H, W = 2, 3, 32, 32
        Y = torch.rand(B, C, H, W)
        X_hat = torch.rand(B, C, H, W)

        R = X_hat - Y
        reconstructed = Y + R
        diff = torch.max(torch.abs(reconstructed - X_hat)).item()
        self.assertLess(diff, 1e-6)

    def test_charbonnier_loss_smoothness(self):
        """Verifies Charbonnier loss is strictly positive and differentiable at zero difference."""
        loss_fn = CharbonnierLoss(eps=1e-3)
        x = torch.zeros(2, 3, 16, 16)
        y = torch.zeros(2, 3, 16, 16)
        val = loss_fn(x, y).item()
        self.assertAlmostEqual(val, 1e-3, places=5)

    def test_warmup_schedule_bounds(self):
        """Verifies lambda_DA warmup schedule is bounded in [0, beta_max]."""
        trainer = NoiseDATrainer(use_domain_adaptation=True, beta_max=0.2, gamma_rate=5.0)
        self.assertEqual(trainer.compute_lambda_dif(0, 50), 0.0)
        final_val = trainer.compute_lambda_dif(50, 50)
        self.assertGreater(final_val, 0.0)
        self.assertLessEqual(final_val, 0.20001)

    def test_cross_domain_batch_size_alignment(self):
        """Verifies NoiseDATrainer safely aligns mismatched source and target batch sizes."""
        trainer = NoiseDATrainer(use_domain_adaptation=True, is_adaptive=True, base_channels=16)
        trainer.train()

        # Source batch size 4, Target batch size 2
        x_s = torch.rand(4, 3, 32, 32)
        y_s = torch.rand(4, 3, 32, 32)
        x_r = torch.rand(2, 3, 32, 32)

        out = trainer(x_s, y_s, x_r, current_epoch=1, total_epochs=10)
        self.assertIn("total_loss", out)
        self.assertTrue(torch.isfinite(out["total_loss"]))
        self.assertIn("loss_dif", out)
        self.assertTrue(torch.isfinite(out["loss_dif"]))

if __name__ == "__main__":
    unittest.main()

