import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp
from typing import List, Dict, Any, Optional

def compute_per_image_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> List[float]:
    """
    Computes per-image Peak Signal-to-Noise Ratio (PSNR) across a batch [B, C, H, W].
    Returns a list of float PSNR values of length B.
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    B = img1.shape[0]
    # Calculate MSE per sample: average over [C, H, W]
    diff = img1 - img2
    mse_per_image = torch.mean(diff ** 2, dim=[1, 2, 3]).detach().cpu().numpy()

    psnr_list = []
    for mse in mse_per_image:
        if mse <= 1e-12:
            psnr_list.append(100.0)  # Capped max for zero error
        else:
            psnr_list.append(float(10.0 * np.log10((data_range ** 2) / mse)))
    return psnr_list


def gaussian_window(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


_SSIM_WINDOW_CACHE = {}

def get_ssim_window(window_size: int, channel: int, device: torch.device) -> torch.Tensor:
    key = (window_size, channel, device)
    if key not in _SSIM_WINDOW_CACHE:
        _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous().to(device)
        _SSIM_WINDOW_CACHE[key] = window
    return _SSIM_WINDOW_CACHE[key]


def compute_per_image_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> List[float]:
    """
    Computes per-image SSIM across a batch [B, C, H, W].
    Returns a list of float SSIM values of length B.
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    B, channel, _, _ = img1.size()
    window = get_ssim_window(window_size, channel, img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # Average across channels and spatial dims per image [B]
    ssim_per_image = torch.mean(ssim_map, dim=[1, 2, 3]).detach().cpu().numpy()
    return [float(v) for v in ssim_per_image]


class LPIPSCalculator:
    """
    Learned Perceptual Image Patch Similarity (LPIPS) Wrapper.
    FAILS LOUDLY if lpips package is missing when queried, preventing silent 0.0 metrics.
    """
    def __init__(self, device: torch.device, strict: bool = False):
        self.device = device
        self.strict = strict
        self.enabled = False
        self.loss_fn = None
        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net='alex').to(device)
            self.loss_fn.eval()
            self.enabled = True
        except ImportError:
            if strict:
                raise ImportError(
                    "LPIPS package is not installed! Run 'pip install lpips' "
                    "or pass --skip_lpips to evaluation script."
                )

    @torch.no_grad()
    def calculate_per_image(self, img1: torch.Tensor, img2: torch.Tensor) -> Optional[List[float]]:
        if not self.enabled:
            if self.strict:
                raise RuntimeError("LPIPS requested but library is unavailable!")
            return None  # Explicit None indicates metric is unavailable

        img1_scaled = img1 * 2.0 - 1.0
        img2_scaled = img2 * 2.0 - 1.0
        val = self.loss_fn(img1_scaled.to(self.device), img2_scaled.to(self.device))
        return [float(v) for v in val.view(-1).cpu().numpy()]


def aggregate_metric_distribution(values: List[float]) -> Dict[str, float]:
    """Calculates mean, std, median, and 95% confidence intervals from sample list."""
    if not values:
        return {"mean": float('nan'), "std": float('nan'), "median": float('nan'), "ci_95": float('nan')}
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    median = float(np.median(arr))
    ci_95 = float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "ci_95": ci_95
    }

# Backwards compatibility wrappers
def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    return float(np.mean(compute_per_image_psnr(img1, img2, data_range)))

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    return float(np.mean(compute_per_image_ssim(img1, img2, window_size)))
