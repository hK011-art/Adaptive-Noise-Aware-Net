import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import torch
import numpy as np
from src.noise.snic_noise import SNICNoiseGenerator

class TestNoisePhysics(unittest.TestCase):
    def setUp(self):
        self.noise_gen = SNICNoiseGenerator(sensor_type="iphone11_main")

    def test_poisson_sampling_intensity_range(self):
        """Verifies Poisson noise across extreme dynamic ranges without NaNs or negative outputs."""
        for intensity in [0.0, 1e-4, 0.1, 0.5, 1.0]:
            clean = torch.full((2, 3, 32, 32), intensity, dtype=torch.float32)
            noisy, res = self.noise_gen.add_noise(clean, iso=1600.0)

            self.assertFalse(torch.isnan(noisy).any(), f"NaN detected at intensity {intensity}!")
            self.assertFalse(torch.isinf(noisy).any(), f"Inf detected at intensity {intensity}!")
            self.assertTrue((noisy >= 0.0).all() and (noisy <= 1.0).all(), f"Values out of [0, 1] at intensity {intensity}!")

    def test_iso_variance_scaling(self):
        """Verifies that empirical noise variance strictly increases with higher ISO levels."""
        clean = torch.full((4, 3, 64, 64), 0.5, dtype=torch.float32)
        
        torch.manual_seed(42)
        _, res_low = self.noise_gen.add_noise(clean, iso=100.0)
        _, res_high = self.noise_gen.add_noise(clean, iso=3200.0)

        var_low = torch.var(res_low).item()
        var_high = torch.var(res_high).item()

        self.assertGreater(var_high, var_low, f"High ISO variance ({var_high:.2e}) should exceed Low ISO ({var_low:.2e})!")

    def test_read_noise_zero_intensity(self):
        """Verifies that dark exposures still exhibit electronic read noise variance."""
        dark = torch.zeros(4, 3, 64, 64, dtype=torch.float32)
        _, res = self.noise_gen.add_noise(dark, iso=800.0)
        var = torch.var(res).item()
        self.assertGreater(var, 0.0, "Zero-intensity exposures must still exhibit read noise variance!")

if __name__ == "__main__":
    unittest.main()

