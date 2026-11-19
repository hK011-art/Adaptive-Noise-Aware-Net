import torch
import numpy as np
from torch.utils.data import Dataset

class ToySyntheticDataset(Dataset):
    """
    STRICTLY FOR UNIT TESTS AND SMOKE TESTING ONLY.
    Never import or use this class in production training or benchmark evaluation.
    """
    def __init__(self, num_samples: int = 16, patch_size: int = 64, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.patch_size = patch_size
        rng = np.random.RandomState(seed)
        
        self.samples = []
        for i in range(num_samples):
            # Deterministic geometric patterns for unit testing
            x = np.linspace(0, 1, patch_size)
            y = np.linspace(0, 1, patch_size)
            xx, yy = np.meshgrid(x, y)
            clean = np.stack([
                np.sin(xx * 3.0 + i * 0.1) * 0.5 + 0.5,
                np.cos(yy * 3.0 + i * 0.1) * 0.5 + 0.5,
                (xx + yy) * 0.5
            ], axis=-1).astype(np.float32)
            clean = np.clip(clean, 0.0, 1.0)
            
            # Simple noise for test verification
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

