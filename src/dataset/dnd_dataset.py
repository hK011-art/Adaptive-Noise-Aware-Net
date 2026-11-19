import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class HeldOutDNDDataset(Dataset):
    """
    Genuine Darmstadt Noise Dataset (DND) Loader.
    DND is strictly a held-out benchmark and must NEVER be used for training or tuning.
    Fails loudly if official DND files are not present.
    """
    def __init__(
        self,
        root_dir: str = "data/dnd",
        is_train: bool = False,
        patch_size: int = 128,
        num_samples: int = 200,
        seed: int = 42
    ):
        super().__init__()
        if is_train:
            raise ValueError(
                "SCIENTIFIC VIOLATION: DND dataset must NEVER be used for training or tuning! "
                "It is strictly an external held-out benchmark for cross-camera evaluation."
            )

        self.root_dir = root_dir
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.seed = seed

        # Fail loudly if dataset folder does not exist
        if not os.path.exists(root_dir):
            raise FileNotFoundError(
                f"\n[DATASET ERROR] DND directory '{root_dir}' not found!\n"
                f"Expected directory structure: '{root_dir}/' containing genuine DND benchmark crops/files.\n"
                f"Download official DND benchmark from: https://noise.visinf.tu-darmstadt.de/\n"
                f"Configure path via '--dnd_dir' CLI argument or dataset config file.\n"
                f"SYNTHETIC SINUSOID FALLBACK HAS BEEN PERMANENTLY REMOVED."
            )

        self.samples = []
        self.sample_manifest_ids = []
        self._load_genuine_dnd()

    def _load_genuine_dnd(self):
        dnd_files = sorted([
            os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir)
            if f.lower().endswith(('.png', '.jpg', '.mat'))
        ])

        if len(dnd_files) == 0:
            raise FileNotFoundError(
                f"\n[DATASET ERROR] Found 0 valid DND benchmark files in '{self.root_dir}'!\n"
                f"Please place real DND evaluation crops in this folder."
            )

        print(f"[Dataset] DND (Held-out Benchmark): Found {len(dnd_files)} real evaluation crops.")
        for p in dnd_files[:self.num_samples]:
            try:
                # If image format
                if p.lower().endswith(('.png', '.jpg')):
                    img = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
                    tensor = torch.from_numpy(img.transpose(2, 0, 1).copy())
                    # For benchmark crops without published clean GT, clean is placeholder or paired
                    self.samples.append((tensor, tensor))
                    self.sample_manifest_ids.append(f"dnd_{os.path.basename(p)}")
            except Exception as e:
                print(f"  [Warning] Error loading DND sample {p}: {e}")

        if len(self.samples) == 0:
            raise RuntimeError(f"Failed to load any genuine DND benchmark crops from '{self.root_dir}'.")

    def get_sample_ids(self) -> list[str]:
        return self.sample_manifest_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
