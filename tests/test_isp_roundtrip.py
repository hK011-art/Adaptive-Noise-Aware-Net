import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import torch
import numpy as np
from src.noise.isp_pipeline import ISPPipeline

class TestISPRoundtrip(unittest.TestCase):
    """
    Validates round-trip numerical fidelity of simplified camera-space ISP:
    sRGB -> Linear RGB -> Simplified Camera RGB -> Linear RGB -> sRGB
    Measures MAE, RMSE, PSNR, and Max Absolute Error.
    """
    def test_isp_roundtrip_zero_noise(self):
        isp = ISPPipeline(gamma=2.2, use_exact_srgb=True)

        # Test tensor spanning the full dynamic range [0.0, 1.0]
        B, C, H, W = 4, 3, 64, 64
        torch.manual_seed(42)
        srgb_orig = torch.rand(B, C, H, W)

        # Round trip
        cam_space = isp.srgb_to_camera_space(srgb_orig)
        srgb_recon = isp.camera_space_to_srgb(cam_space)

        # Numerical metrics
        diff = torch.abs(srgb_orig - srgb_recon)
        mae = torch.mean(diff).item()
        rmse = torch.sqrt(torch.mean(diff ** 2)).item()
        max_err = torch.max(diff).item()
        psnr = 10.0 * np.log10(1.0 / (rmse ** 2)) if rmse > 0 else 100.0

        # Invertibility assertions: reconstruction error must be negligible (< 1e-4)
        self.assertLess(mae, 1e-4, f"MAE too high: {mae}")
        self.assertLess(rmse, 1e-4, f"RMSE too high: {rmse}")
        self.assertLess(max_err, 1e-3, f"Max error too high: {max_err}")
        self.assertGreater(psnr, 70.0, f"Round-trip PSNR must exceed 70 dB, got {psnr:.2f} dB")

    def test_isp_boundary_conditions(self):
        """Validates boundary values (0.0 and 1.0) survive round trip cleanly."""
        isp = ISPPipeline(gamma=2.2, use_exact_srgb=True)
        black = torch.zeros(1, 3, 16, 16)
        white = torch.ones(1, 3, 16, 16)

        black_recon = isp.camera_space_to_srgb(isp.srgb_to_camera_space(black))
        white_recon = isp.camera_space_to_srgb(isp.srgb_to_camera_space(white))

        self.assertTrue(torch.allclose(black, black_recon, atol=1e-5), "Black level failed round trip!")
        self.assertTrue(torch.allclose(white, white_recon, atol=1e-5), "White level failed round trip!")

if __name__ == "__main__":
    unittest.main()
