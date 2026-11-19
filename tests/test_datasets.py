import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import torch
import numpy as np
from torch.utils.data import Dataset
from src.dataset.div2k_dataset import PhotographicDIV2KDataset
from src.dataset.sidd_dataset import GenuineSIDDDataset
from src.dataset.dnd_dataset import HeldOutDNDDataset
from src.utils.manifest import check_for_data_leakage

class ToySyntheticDataset(Dataset):
    """
    STRICTLY FOR UNIT TESTS AND SMOKE TESTING ONLY.
    Isolated deterministic fixture embedded directly in tests.
    """
    def __init__(self, num_samples: int = 16, patch_size: int = 64, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.patch_size = patch_size
        rng = np.random.RandomState(seed)
        
        self.samples = []
        for i in range(num_samples):
            x = np.linspace(0, 1, patch_size)
            y = np.linspace(0, 1, patch_size)
            xx, yy = np.meshgrid(x, y)
            clean = np.stack([
                np.sin(xx * 3.0 + i * 0.1) * 0.5 + 0.5,
                np.cos(yy * 3.0 + i * 0.1) * 0.5 + 0.5,
                (xx + yy) * 0.5
            ], axis=-1).astype(np.float32)
            clean = np.clip(clean, 0.0, 1.0)
            
            noise = rng.normal(0, 0.05, clean.shape).astype(np.float32)
            noisy = np.clip(clean + noise, 0.0, 1.0)
            
            self.samples.append((
                torch.from_numpy(noisy).permute(2, 0, 1),
                torch.from_numpy(clean).permute(2, 0, 1)
            ))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.samples[idx]


class TestDatasets(unittest.TestCase):
    def test_missing_div2k_fails_loudly(self):
        """Verifies missing DIV2K directory raises FileNotFoundError and does NOT create fake data."""
        with self.assertRaises(FileNotFoundError):
            PhotographicDIV2KDataset(root_dir="non_existent_div2k_directory_xyz")

    def test_missing_sidd_fails_loudly(self):
        """Verifies missing SIDD directory raises FileNotFoundError and does NOT create fake gradients."""
        with self.assertRaises(FileNotFoundError):
            GenuineSIDDDataset(root_dir="non_existent_sidd_directory_xyz")

    def test_missing_dnd_fails_loudly(self):
        """Verifies missing DND directory raises FileNotFoundError and does NOT create fake sinusoids."""
        with self.assertRaises(FileNotFoundError):
            HeldOutDNDDataset(root_dir="non_existent_dnd_directory_xyz")

    def test_dnd_training_forbidden(self):
        """Verifies DND cannot be loaded in training mode (strictly held out)."""
        with self.assertRaises(ValueError):
            HeldOutDNDDataset(root_dir="data/dnd", is_train=True)

    def test_toy_fixture_quarantine(self):
        """Verifies test toy dataset functions as an isolated fixture."""
        fixture = ToySyntheticDataset(num_samples=8, patch_size=32)
        self.assertEqual(len(fixture), 8)
        noisy, clean = fixture[0]
        self.assertEqual(noisy.shape, (3, 32, 32))
        self.assertEqual(clean.shape, (3, 32, 32))

    def test_data_leakage_detection(self):
        """Verifies data leakage detector raises ValueError upon scene overlap."""
        train_ids = ["scene_001_p1", "scene_001_p2", "scene_002_p1"]
        val_ids_clean = ["scene_003_p1", "scene_004_p1"]
        val_ids_leaked = ["scene_001_p1", "scene_003_p1"]

        # Should pass without error
        self.assertTrue(check_for_data_leakage(train_ids, val_ids_clean))

        # Should raise ValueError
        with self.assertRaises(ValueError):
            check_for_data_leakage(train_ids, val_ids_leaked)

    def test_sidd_multi_patch_extraction(self):
        """Verifies GenuineSIDDDataset extracts multiple crops per image to reach requested num_samples."""
        import tempfile
        import shutil
        from PIL import Image

        tmp_dir = tempfile.mkdtemp()
        try:
            # Create 2 small images (64x64)
            img1 = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
            img2 = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
            img1.save(os.path.join(tmp_dir, "0001_noisy.png"))
            img2.save(os.path.join(tmp_dir, "0002_noisy.png"))

            ds = GenuineSIDDDataset(root_dir=tmp_dir, num_samples=10, patch_size=32, is_train=True)
            self.assertEqual(len(ds), 10)
            sample_noisy = ds[0]
            self.assertEqual(sample_noisy.shape, (3, 32, 32))
            self.assertEqual(len(ds.get_sample_ids()), 10)

            # Test evaluation mode returns (noisy, clean)
            ds_eval = GenuineSIDDDataset(root_dir=tmp_dir, num_samples=10, patch_size=32, is_train=False)
            self.assertEqual(len(ds_eval), 10)
            eval_noisy, eval_clean = ds_eval[0]
            self.assertEqual(eval_noisy.shape, (3, 32, 32))
            self.assertEqual(eval_clean.shape, (3, 32, 32))
        finally:
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    unittest.main()
