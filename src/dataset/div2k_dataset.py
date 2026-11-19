import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from ..noise.awgn import AWGNGenerator
from ..noise.snic_noise import SNICNoiseGenerator

class PhotographicDIV2KDataset(Dataset):
    """
    Real Photographic DIV2K Dataset Loader with Sensor Noise Synthesis.
    Strictly requires genuine image files on disk; fails loudly if absent.
    Prevents data leakage through strict scene-level partitioning.
    """
    def __init__(
        self,
        root_dir: str = "data/div2k",
        split: str = "train",
        val_ratio: float = 0.2,
        patch_size: int = 128,
        noise_type: str = "snic",
        sensor_type: str = "iphone11_main",
        num_samples: int = 1000,
        iso_list: list[float] = None,
        seed: int = 42
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split.lower()
        self.patch_size = patch_size
        self.noise_type = noise_type.lower()
        self.num_samples = num_samples
        self.iso_list = iso_list if iso_list is not None else [100.0, 200.0, 400.0, 800.0]
        self.seed = seed

        # Fail loudly if dataset folder does not exist
        if not os.path.exists(root_dir):
            raise FileNotFoundError(
                f"\n[DATASET ERROR] DIV2K directory '{root_dir}' not found!\n"
                f"Expected directory structure: '{root_dir}/' containing high-resolution .png or .jpg images.\n"
                f"Download DIV2K dataset from: https://data.vision.ee.ethz.ch/cvl/DIV2K/\n"
                f"Configure the path via the '--data_root' CLI argument or dataset config file.\n"
                f"SILENT SYNTHETIC FALLBACK IS STRICTLY FORBIDDEN FOR SCIENTIFIC RIGOR."
            )

        # Collect genuine images
        all_image_paths = sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if len(all_image_paths) == 0:
            raise FileNotFoundError(
                f"\n[DATASET ERROR] Found 0 valid images in DIV2K directory '{root_dir}'!\n"
                f"Please ensure uncompressed .png or .jpg image files are located in '{root_dir}'."
            )

        # Scene-level train/val partition to prevent patch-level data leakage
        num_images = len(all_image_paths)
        num_val = max(1, int(num_images * val_ratio))
        if self.split == "train":
            self.image_paths = all_image_paths[:-num_val] if num_images > 1 else all_image_paths
        elif self.split == "val":
            self.image_paths = all_image_paths[-num_val:] if num_images > 1 else all_image_paths
        else:
            self.image_paths = all_image_paths

        print(f"[Dataset] DIV2K ({self.split.upper()}): Loaded {len(self.image_paths)} unique scene images from {root_dir}")

        if self.noise_type == "awgn":
            self.noise_gen = AWGNGenerator(sigma_min=0.0, sigma_max=75.0)
        else:
            self.noise_gen = SNICNoiseGenerator(sensor_type=sensor_type)

        self.clean_patches = []
        self._extract_patches()

    def _extract_patches(self):
        loaded_images = []
        for p in self.image_paths:
            try:
                pil_img = Image.open(p).convert("RGB")
                np_img = np.array(pil_img, dtype=np.float32) / 255.0
                loaded_images.append((os.path.basename(p), np_img))
            except Exception as e:
                print(f"  [Warning] Failed to load {p}: {e}")

        if not loaded_images:
            raise RuntimeError(f"Failed to load any valid image from {self.image_paths}")

        rng = np.random.RandomState(self.seed)
        num_patches_per_img = max(1, int(np.ceil(self.num_samples / len(loaded_images))))

        self.sample_manifest_ids = []
        for img_name, img_np in loaded_images:
            H, W, _ = img_np.shape
            if H < self.patch_size or W < self.patch_size:
                continue
            for patch_idx in range(num_patches_per_img):
                y = rng.randint(0, H - self.patch_size + 1)
                x = rng.randint(0, W - self.patch_size + 1)
                patch = img_np[y : y + self.patch_size, x : x + self.patch_size].copy()
                if rng.rand() > 0.5:
                    patch = np.fliplr(patch)
                if rng.rand() > 0.5:
                    patch = np.flipud(patch)
                patch = np.ascontiguousarray(patch)
                tensor_clean = torch.from_numpy(patch).permute(2, 0, 1)
                self.clean_patches.append(tensor_clean)
                self.sample_manifest_ids.append(f"{img_name}_y{y}_x{x}")
                if len(self.clean_patches) >= self.num_samples:
                    break
            if len(self.clean_patches) >= self.num_samples:
                break

    def get_sample_ids(self) -> list[str]:
        return self.sample_manifest_ids

    def __len__(self):
        return len(self.clean_patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean_img = self.clean_patches[idx]
        if self.noise_type == "awgn":
            noisy_img, _ = self.noise_gen.add_noise(clean_img.unsqueeze(0))
            noisy_img = noisy_img.squeeze(0)
        else:
            iso = float(self.iso_list[idx % len(self.iso_list)])
            noisy_img, _ = self.noise_gen.add_noise(clean_img.unsqueeze(0), iso=iso)
            noisy_img = noisy_img.squeeze(0)
        return noisy_img, clean_img
