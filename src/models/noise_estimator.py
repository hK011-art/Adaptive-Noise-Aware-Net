import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialNoiseEstimator(nn.Module):
    """
    Lightweight 4-layer CNN for Spatial Noise-Strength Map Estimation.
    Estimates a 1-channel spatial noise-strength map M_hat from a 3-channel noisy RGB image.
    """
    def __init__(self, in_channels: int = 3, hidden_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)  # Noise-strength is non-negative
        )
        nn.init.constant_(self.net[-2].bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy RGB image tensor [B, 3, H, W]
        Returns:
            m_hat: Estimated spatial noise-strength map [B, 1, H, W]
        """
        return self.net(x)


class NoiseEncoder(nn.Module):
    """
    Degradation Representation Encoder.
    Compresses the 1-channel spatial noise-strength map M_hat into a compact
    latent degradation embedding vector z in R^{embedding_dim}.
    """
    def __init__(self, in_channels: int = 1, embedding_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, embedding_dim)

    def forward(self, m_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m_hat: Spatial noise-strength map [B, 1, H, W]
        Returns:
            z: Compact degradation embedding vector [B, embedding_dim]
        """
        feat = self.encoder(m_hat)
        feat = feat.flatten(1)
        z = self.fc(feat)
        return z


def compute_noise_strength_target(x_noisy: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
    """
    Computes ground-truth 1-channel spatial noise-strength map M_target from
    noisy and clean RGB images:
        M_target(x, y) = sqrt( 1/3 * sum_{c in {R,G,B}} (x_noisy_c - x_clean_c)^2 )
        
    Args:
        x_noisy: Noisy RGB tensor [B, 3, H, W]
        x_clean: Clean RGB reference tensor [B, 3, H, W]
    Returns:
        m_target: 1-channel spatial noise-strength map [B, 1, H, W]
    """
    diff_sq = (x_noisy - x_clean) ** 2
    mean_sq = torch.mean(diff_sq, dim=1, keepdim=True)
    m_target = torch.sqrt(torch.clamp(mean_sq, min=1e-8))
    return m_target

