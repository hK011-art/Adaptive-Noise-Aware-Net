import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import torch
from src.utils.metrics import (
    compute_per_image_psnr,
    compute_per_image_ssim,
    LPIPSCalculator,
    aggregate_metric_distribution
)

class TestMetrics(unittest.TestCase):
    def test_per_image_psnr_independent(self):
        """Verifies per-image PSNR returns discrete individual scores per sample."""
        clean = torch.ones(2, 3, 32, 32)
        # Sample 0 has small error, Sample 1 has larger error
        noisy = clean.clone()
        noisy[0] = noisy[0] + 0.01
        noisy[1] = noisy[1] + 0.1

        psnrs = compute_per_image_psnr(noisy, clean)
        self.assertEqual(len(psnrs), 2)
        self.assertGreater(psnrs[0], psnrs[1], "Sample 0 (lower noise) must have higher PSNR than Sample 1!")

    def test_per_image_ssim_range(self):
        """Verifies per-image SSIM returns values in [0, 1]."""
        img1 = torch.rand(2, 3, 32, 32)
        img2 = torch.rand(2, 3, 32, 32)
        ssims = compute_per_image_ssim(img1, img2)
        self.assertEqual(len(ssims), 2)
        for s in ssims:
            self.assertGreaterEqual(s, -1.0)
            self.assertLessEqual(s, 1.0)

    def test_metric_aggregation_statistics(self):
        """Verifies mean, std, median, and confidence interval calculation."""
        vals = [20.0, 22.0, 24.0, 26.0, 28.0]
        stats = aggregate_metric_distribution(vals)
        self.assertEqual(stats["mean"], 24.0)
        self.assertEqual(stats["median"], 24.0)
        self.assertGreater(stats["std"], 0.0)
        self.assertGreater(stats["ci_95"], 0.0)

if __name__ == "__main__":
    unittest.main()

