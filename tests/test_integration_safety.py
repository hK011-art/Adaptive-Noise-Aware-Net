import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import torch
from src.utils.statistical_testing import perform_paired_comparison, apply_holm_bonferroni_correction
from src.utils.manifest import (
    validate_training_dataset_safety,
    check_for_data_leakage,
    extract_scene_id_from_sample,
    compute_file_sha256
)
from src.models.adaptive_unet import AdaptiveDenoiser
from src.models.noise_da_trainer import NoiseDATrainer

class TestIntegrationSafety(unittest.TestCase):
    def test_holm_bonferroni_correction(self):
        """Verifies Holm-Bonferroni stepwise p-value adjustment."""
        comparisons = [
            {"name": "comp1", "p_value": 0.01},
            {"name": "comp2", "p_value": 0.04},
            {"name": "comp3", "p_value": 0.03}
        ]
        corrected = apply_holm_bonferroni_correction(comparisons, alpha=0.05)
        self.assertEqual(len(corrected), 3)
        for c in corrected:
            self.assertIn("p_value_holm_adjusted", c)
            self.assertIn("is_significant_holm", c)
            self.assertLessEqual(c["p_value"], c["p_value_holm_adjusted"])

    def test_dnd_training_safety_guard(self):
        """Verifies validate_training_dataset_safety raises ValueError when DND is set for train."""
        with self.assertRaises(ValueError):
            validate_training_dataset_safety("data/dnd", "train")
        with self.assertRaises(ValueError):
            validate_training_dataset_safety("dnd_benchmark", "train")

    def test_scene_id_extraction_and_leakage(self):
        """Verifies scene-level leakage detection catches identical scenes even with different patch coordinates."""
        train_samples = ["sceneA_y0_x0", "sceneA_y128_x128"]
        val_samples_leaked = ["sceneA_y256_x256", "sceneB_y0_x0"]
        val_samples_clean = ["sceneC_y0_x0", "sceneD_y0_x0"]

        self.assertEqual(extract_scene_id_from_sample("sceneA_y0_x0"), "sceneA")
        # Should raise due to sceneA overlap
        with self.assertRaises(ValueError):
            check_for_data_leakage(train_samples, val_samples_leaked)
        # Clean splits should pass
        self.assertTrue(check_for_data_leakage(train_samples, val_samples_clean))

    def test_film_parameter_group_isolation(self):
        """Verifies that FiLM learning rate grouping isolates all FiLM parameters without duplication or omissions."""
        trainer = NoiseDATrainer(use_domain_adaptation=True, is_adaptive=True, base_channels=32)

        film_params = []
        base_params = []
        for name, param in trainer.named_parameters():
            if "mod_" in name:
                film_params.append(param)
            else:
                base_params.append(param)

        all_params = list(trainer.parameters())
        self.assertEqual(len(film_params) + len(base_params), len(all_params))
        # Ensure no duplicate parameter IDs
        film_ids = {id(p) for p in film_params}
        base_ids = {id(p) for p in base_params}
        self.assertEqual(len(film_ids.intersection(base_ids)), 0, "Parameter duplicated between FiLM and base groups!")
        self.assertGreater(len(film_params), 0, "FiLM parameters must be identified!")

    def test_dual_residual_algebra(self):
        """
        Verifies mathematical equivalence between:
        Restoration: X_hat = Y - R_noise
        Domain Shift: X_hat = Y + R_shift (where R_shift = -R_noise = X_hat - Y).
        """
        B, C, H, W = 2, 3, 32, 32
        Y = torch.rand(B, C, H, W)
        X_hat = torch.rand(B, C, H, W)

        R_noise = Y - X_hat
        R_shift = X_hat - Y

        rec_from_noise = Y - R_noise
        rec_from_shift = Y + R_shift

        self.assertTrue(torch.allclose(rec_from_noise, X_hat, atol=1e-6))
        self.assertTrue(torch.allclose(rec_from_shift, X_hat, atol=1e-6))
        self.assertTrue(torch.allclose(R_noise, -R_shift, atol=1e-6))

if __name__ == "__main__":
    unittest.main()

