import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import torch
import torch.optim as optim
from src.models.adaptive_unet import AdaptiveDenoiser
from src.utils.checkpoint import save_checkpoint, load_checkpoint

class TestCheckpoints(unittest.TestCase):
    def test_save_and_load_full_checkpoint(self):
        """Verifies full checkpoint saving and loading round trip."""
        tmp_ckpt = "tests/test_ckpt_tmp.pth"
        model = AdaptiveDenoiser(base_channels=16, embedding_dim=16)
        opt = optim.Adam(model.parameters(), lr=1e-4)

        save_checkpoint(
            path=tmp_ckpt,
            epoch=5,
            model_name="Model_E_Test",
            restoration_net=model,
            optimizer=opt,
            best_metric=24.5,
            config={"test": True},
            seed=42,
            dataset_manifest_path="test_manifest.json"
        )

        self.assertTrue(os.path.exists(tmp_ckpt))
        self.assertTrue(os.path.exists(tmp_ckpt.replace(".pth", "_inference.pth")))

        # Load back
        model_new = AdaptiveDenoiser(base_channels=16, embedding_dim=16)
        opt_new = optim.Adam(model_new.parameters(), lr=1e-4)
        meta = load_checkpoint(tmp_ckpt, model_new, optimizer=opt_new)

        self.assertEqual(meta["epoch"], 5)
        self.assertEqual(meta["model_name"], "Model_E_Test")
        self.assertAlmostEqual(meta["best_metric"], 24.5)

        # Cleanup
        if os.path.exists(tmp_ckpt):
            os.remove(tmp_ckpt)
        inf_p = tmp_ckpt.replace(".pth", "_inference.pth")
        if os.path.exists(inf_p):
            os.remove(inf_p)

if __name__ == "__main__":
    unittest.main()

