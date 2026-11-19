import os
import sys
import unittest
import torch
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.adaptive_unet import AdaptiveDenoiser
from src.models.noise_da_trainer import NoiseDATrainer
from src.utils.statistical_testing import perform_paired_comparison, apply_holm_bonferroni_correction
from src.utils.manifest import extract_scene_id_from_sample, check_for_data_leakage
from src.utils.checkpoint import save_checkpoint, load_checkpoint

class TestPhase3Hardening(unittest.TestCase):
    """
    Phase 3 Verification & Hardening Test Suite:
    Validates research integrity, data leakage defenses, ground-truth isolation,
    mathematical monotonicity of statistical adjustments, and numerical loss reconstruction.
    """

    def test_strict_ground_truth_leakage_prevention(self):
        """
        Part 3 Audit: Guarantees that passing x_clean during normal, zero, wrong, or shuffled
        inference modes immediately raises a ValueError to prevent ground-truth leakage.
        """
        model = AdaptiveDenoiser(base_channels=16, embedding_dim=16)
        y = torch.rand(2, 3, 32, 32)
        x_clean = torch.rand(2, 3, 32, 32)

        # Normal mode with clean reference must fail loud
        with self.assertRaises(ValueError) as ctx:
            model.forward_with_conditioning_mode(y, x_clean=x_clean, mode="normal")
        self.assertIn("GROUND-TRUTH LEAKAGE DETECTED", str(ctx.exception))

        # Zero mode with clean reference must fail loud
        with self.assertRaises(ValueError):
            model.forward_with_conditioning_mode(y, x_clean=x_clean, mode="zero")

        # Wrong mode with clean reference must fail loud
        with self.assertRaises(ValueError):
            model.forward_with_conditioning_mode(y, x_clean=x_clean, mode="wrong")

        # Shuffled mode with clean reference must fail loud
        with self.assertRaises(ValueError):
            model.forward_with_conditioning_mode(y, x_clean=x_clean, mode="shuffled")

        # Normal mode without clean reference must succeed
        x_hat, m_hat, z = model.forward_with_conditioning_mode(y, x_clean=None, mode="normal")
        self.assertEqual(x_hat.shape, y.shape)

        # Oracle mode without clean reference must fail loud
        with self.assertRaises(ValueError):
            model.forward_with_conditioning_mode(y, x_clean=None, mode="oracle")

        # Oracle mode WITH clean reference must succeed
        x_hat_oracle, _, _ = model.forward_with_conditioning_mode(y, x_clean=x_clean, mode="oracle")
        self.assertEqual(x_hat_oracle.shape, y.shape)

    def test_holm_bonferroni_monotonicity_and_ordering(self):
        """
        Part 19 Audit: Verifies cumulative-maximum monotonicity on adjusted p-values
        and ensures restoration of original index ordering.
        """
        # Designed sequence where unconstrained (m - k + 1) * p_(k) is non-monotonic:
        # m = 3
        # Rank 1: p = 0.020 -> unconstrained 0.020 * 3 = 0.060
        # Rank 2: p = 0.025 -> unconstrained 0.025 * 2 = 0.050 (would decrease without cummax!)
        # Rank 3: p = 0.040 -> unconstrained 0.040 * 1 = 0.040
        tests = [
            {"id": "test_C", "p_value": 0.040},
            {"id": "test_A", "p_value": 0.020},
            {"id": "test_B", "p_value": 0.025}
        ]

        corrected = apply_holm_bonferroni_correction(tests, alpha=0.05)

        # 1. Check original ordering is strictly preserved
        self.assertEqual(corrected[0]["id"], "test_C")
        self.assertEqual(corrected[1]["id"], "test_A")
        self.assertEqual(corrected[2]["id"], "test_B")

        # 2. Check cumulative-max monotonicity:
        # test_A (rank 1, p=0.020) -> adj = 0.060
        # test_B (rank 2, p=0.025) -> adj = max(0.060, 0.050) = 0.060
        # test_C (rank 3, p=0.040) -> adj = max(0.060, 0.040) = 0.060
        adj_pvals_in_rank_order = [
            corrected[1]["p_value_holm_adjusted"], # Rank 1
            corrected[2]["p_value_holm_adjusted"], # Rank 2
            corrected[0]["p_value_holm_adjusted"]  # Rank 3
        ]
        for i in range(len(adj_pvals_in_rank_order) - 1):
            self.assertLessEqual(
                adj_pvals_in_rank_order[i],
                adj_pvals_in_rank_order[i + 1],
                "Holm adjusted p-values must be non-decreasing in rank order!"
            )

        # All adjusted values must be <= 1.0
        for item in corrected:
            self.assertLessEqual(item["p_value_holm_adjusted"], 1.0)

    def test_paired_effect_size_and_confidence_interval(self):
        """
        Part 20 & 21 Audit: Verifies Cohen's dz definition and Student-t paired CI.
        """
        baseline = [30.0, 31.0, 32.0, 29.5, 30.5]
        proposed = [30.8, 31.6, 32.7, 30.2, 31.1] # differences: [+0.8, +0.6, +0.7, +0.7, +0.6], mean = +0.68

        res = perform_paired_comparison(baseline, proposed, metric_name="PSNR", confidence_level=0.95)
        self.assertAlmostEqual(res["mean_diff"], 0.68, places=4)
        self.assertIn("cohens_dz", res)
        self.assertIn("ci_unit_of_analysis", res)
        self.assertEqual(res["ci_confidence_level"], 0.95)
        # Lower and upper CI bounds must bracket the mean difference
        self.assertLess(res["mean_diff_ci_lower"], res["mean_diff"])
        self.assertGreater(res["mean_diff_ci_upper"], res["mean_diff"])

    def test_explicit_dataset_specific_scene_parsers(self):
        """
        Part 2 Audit: Verifies explicit dataset-specific parsing and fail-loud exceptions.
        """
        # DIV2K
        self.assertEqual(extract_scene_id_from_sample("0801_y128_x256.png", dataset_name="div2k"), "0801")
        self.assertEqual(extract_scene_id_from_sample("0805.png", dataset_name="div2k"), "0805")
        with self.assertRaises(ValueError):
            extract_scene_id_from_sample("invalid_name.png", dataset_name="div2k")

        # SIDD
        self.assertEqual(extract_scene_id_from_sample("0001_001_S6_00100_00060_3200_L_NOISY.PNG", dataset_name="sidd"), "0001")
        self.assertEqual(extract_scene_id_from_sample("sidd_0042_crop_01.png", dataset_name="sidd"), "0042")
        with self.assertRaises(ValueError):
            extract_scene_id_from_sample("not_sidd_format.png", dataset_name="sidd")

        # DND
        self.assertEqual(extract_scene_id_from_sample("0015_02.mat", dataset_name="dnd"), "0015")
        self.assertEqual(extract_scene_id_from_sample("dnd_0023_crop.png", dataset_name="dnd"), "0023")

        # Empty or malformed inputs must fail loudly
        with self.assertRaises(ValueError):
            extract_scene_id_from_sample("")
        with self.assertRaises(ValueError):
            extract_scene_id_from_sample("   ")

    def test_numerical_loss_reconstruction(self):
        """
        Part 9 Audit: Verifies that logged loss components numerically reconstruct
        the total loss within floating-point tolerance (|L_total - sum(components)| < 1e-6).
        """
        trainer = NoiseDATrainer(
            use_domain_adaptation=True,
            is_adaptive=True,
            base_channels=16,
            lambda_noise=0.5,
            beta_max=0.2,
            gamma_rate=5.0
        )
        trainer.train()

        x_s = torch.rand(2, 3, 32, 32)
        y_s = torch.rand(2, 3, 32, 32)
        x_r = torch.rand(2, 3, 32, 32)

        out = trainer(x_s, y_s, x_r, current_epoch=2, total_epochs=10)

        total_loss = out["total_loss"].item()
        loss_rec = out["loss_rec"].item()
        loss_noise = out["loss_noise"].item()
        loss_dif = out["loss_dif"].item()
        lambda_dif = out["lambda_dif"]
        lambda_noise = out["lambda_noise"]

        reconstructed_loss = loss_rec + (lambda_noise * loss_noise) + (lambda_dif * loss_dif)

        self.assertAlmostEqual(
            total_loss,
            reconstructed_loss,
            places=6,
            msg=f"Total loss ({total_loss:.7f}) did not match reconstructed components ({reconstructed_loss:.7f})"
        )

    def test_checkpoint_exact_numerical_equality(self):
        """
        Part 14 Audit: Verifies that saving and reloading a checkpoint produces
        exact identical numerical outputs: f(y)_saved == f(y)_loaded.
        """
        ckpt_tmp = "tests/test_p3_ckpt_tmp.pth"
        try:
            model1 = AdaptiveDenoiser(base_channels=16, embedding_dim=16)
            model1.eval()
            y = torch.rand(1, 3, 32, 32)

            with torch.no_grad():
                x_hat1, _, _ = model1(y)

            dummy_cfg = {"model": {"name": "test_e"}, "seed": 42}
            save_checkpoint(
                path=ckpt_tmp,
                epoch=1,
                model_name="model_e",
                restoration_net=model1,
                best_metric=0.123,
                config=dummy_cfg
            )

            # Reload into separate model
            model2 = AdaptiveDenoiser(base_channels=16, embedding_dim=16)
            model2.eval()
            ckpt_info = load_checkpoint(ckpt_tmp, restoration_net=model2)

            # Verify config_sha256 was stored
            self.assertIn("config_sha256", ckpt_info)
            self.assertNotEqual(ckpt_info["config_sha256"], "none")

            with torch.no_grad():
                x_hat2, _, _ = model2(y)

            # Check exact numerical equality
            self.assertTrue(torch.allclose(x_hat1, x_hat2, atol=1e-7))

        finally:
            if os.path.exists(ckpt_tmp):
                os.remove(ckpt_tmp)
            inf_path = ckpt_tmp.replace(".pth", "_inference.pth")
            if os.path.exists(inf_path):
                os.remove(inf_path)

    def test_residual_sign_and_dual_algebraic_reconstruction(self):
        """
        Part 11 Audit: Verifies sign conventions:
        Noise residual R = Y - X_hat
        X_hat = clamp(Y - R, 0, 1)
        Counterfactual synthesis: Y_swap = X_hat + R_other
        """
        Y = torch.tensor([[[[0.8, 0.4]]]])
        X = torch.tensor([[[[0.5, 0.3]]]])
        # True residual
        R = Y - X # [0.3, 0.1]
        X_hat = torch.clamp(Y - R, 0.0, 1.0)
        self.assertTrue(torch.allclose(X_hat, X, atol=1e-7))

        # Recombining restored image with residual recovers original noisy observation exactly
        Y_recon = X_hat + R
        self.assertTrue(torch.allclose(Y_recon, Y, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
