import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import torch
import torch.nn as nn
from src.models.noise_estimator import SpatialNoiseEstimator, NoiseEncoder, compute_noise_strength_target
from src.models.adaptive_unet import FeatureModulation, AdaptiveNoiseAwareUNet, AdaptiveDenoiser

class TestModels(unittest.TestCase):
    def setUp(self):
        self.B, self.H, self.W = 2, 64, 64
        self.clean = torch.rand(self.B, 3, self.H, self.W)
        self.noisy = torch.clamp(self.clean + torch.randn(self.B, 3, self.H, self.W) * 0.1, 0.0, 1.0)

    def test_noise_strength_target_formula(self):
        """Verifies degradation target formula: M = sqrt(1/3 * sum (Y-X)^2)."""
        m_target = compute_noise_strength_target(self.noisy, self.clean)
        self.assertEqual(m_target.shape, (self.B, 1, self.H, self.W))
        self.assertTrue((m_target >= 0.0).all())

        # Exact manual calculation check
        diff = self.noisy - self.clean
        expected = torch.sqrt(torch.mean(diff ** 2, dim=1, keepdim=True))
        diff_err = torch.max(torch.abs(m_target - expected)).item()
        self.assertLess(diff_err, 1e-6)

    def test_spatial_noise_estimator(self):
        """Verifies Scout CNN resolution preservation and non-negative output."""
        scout = SpatialNoiseEstimator(in_channels=3, hidden_channels=32)
        m_hat = scout(self.noisy)
        self.assertEqual(m_hat.shape, (self.B, 1, self.H, self.W))
        self.assertTrue((m_hat >= 0.0).all(), "Estimated degradation map must be strictly non-negative!")

    def test_noise_encoder_and_latent_z(self):
        """Verifies NoiseEncoder maps spatial map M_hat to compact latent z in R^32."""
        encoder = NoiseEncoder(in_channels=1, embedding_dim=32)
        m_dummy = torch.rand(self.B, 1, self.H, self.W)
        z = encoder(m_dummy)
        self.assertEqual(z.shape, (self.B, 32))
        self.assertTrue(torch.isfinite(z).all())

    def test_film_zero_initialization(self):
        """Verifies FiLM weights are zero-initialized and produce exact identity at init."""
        film = FeatureModulation(feature_channels=64, embedding_dim=32)
        feat = torch.randn(self.B, 64, 32, 32)
        z = torch.randn(self.B, 32)

        out = film(feat, z)
        diff = torch.max(torch.abs(out - feat)).item()
        self.assertLess(diff, 1e-6, "FiLM layer must perform exact identity mapping at initialization!")

    def test_adaptive_denoiser_conditioning_modes(self):
        """Verifies normal, zero, wrong, shuffled, and oracle conditioning modes execute cleanly."""
        denoiser = AdaptiveDenoiser(base_channels=32, embedding_dim=32)
        denoiser.eval()

        with torch.no_grad():
            # Normal
            x_norm, _, _ = denoiser.forward_with_conditioning_mode(self.noisy, mode="normal")
            self.assertEqual(x_norm.shape, (self.B, 3, self.H, self.W))

            # Zero
            x_zero, _, _ = denoiser.forward_with_conditioning_mode(self.noisy, mode="zero")
            self.assertEqual(x_zero.shape, (self.B, 3, self.H, self.W))

            # Shuffled
            x_shuf, _, _ = denoiser.forward_with_conditioning_mode(self.noisy, mode="shuffled")
            self.assertEqual(x_shuf.shape, (self.B, 3, self.H, self.W))

            # Wrong
            x_wrng, _, _ = denoiser.forward_with_conditioning_mode(self.noisy, mode="wrong")
            self.assertEqual(x_wrng.shape, (self.B, 3, self.H, self.W))

            # Oracle
            x_orcl, _, _ = denoiser.forward_with_conditioning_mode(self.noisy, x_clean=self.clean, mode="oracle")
            self.assertEqual(x_orcl.shape, (self.B, 3, self.H, self.W))

if __name__ == "__main__":
    unittest.main()

