import torch
import torch.nn as nn
import torch.nn.functional as F
from .noise_estimator import SpatialNoiseEstimator, NoiseEncoder, compute_noise_strength_target

class FeatureModulation(nn.Module):
    """
    Noise-Conditioned FiLM-Style Feature Modulation Layer.
    Transforms intermediate feature tensor F_l using degradation vector z:
        F'_l = (gamma_l(z) + 1.0) * F_l + beta_l(z)
        
    Weights and biases are zero-initialized so modulation begins as an identity mapping:
        gamma = 0, beta = 0  =>  F'_l = F_l.
    """
    def __init__(self, feature_channels: int, embedding_dim: int = 32):
        super().__init__()
        self.fc_gamma = nn.Linear(embedding_dim, feature_channels)
        self.fc_beta = nn.Linear(embedding_dim, feature_channels)

        # Zero-initialize so training starts smoothly from baseline features
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Intermediate feature tensor [B, C, H, W]
            z: Degradation embedding vector [B, embedding_dim]
        Returns:
            x_mod: Modulated feature tensor [B, C, H, W]
        """
        gamma = self.fc_gamma(z).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.fc_beta(z).unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        return (gamma + 1.0) * x + beta


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => LeakyReLU) * 2 with Residual Shortcut"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class AdaptiveNoiseAwareUNet(nn.Module):
    """
    Adaptive Noise-Conditioned Feature Modulation U-Net.
    
    Primary Model E Dataflow:
    - Input: Strictly 3-channel Noisy RGB Image Y [B, 3, H, W]
    - Condition: Degradation embedding vector z [B, 32]
    - Feature Modulation applied at:
        1. Encoder Stage 2 (C = 64)
        2. Bottleneck (C = 256)
        3. Decoder Stage 2 (C = 64)
    - Output: 3-channel restoration residual R_hat_rgb
    - Restored Image: X_hat = clamp(Y - R_hat_rgb, 0.0, 1.0)
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        embedding_dim: int = 32
    ):
        super().__init__()
        c = base_channels

        # Encoder
        self.inc = DoubleConv(in_channels, c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c, c * 2))
        self.mod_enc2 = FeatureModulation(c * 2, embedding_dim)

        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 2, c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 4, c * 8))

        # Bottleneck
        self.bottleneck = DoubleConv(c * 8, c * 8)
        self.mod_bottleneck = FeatureModulation(c * 8, embedding_dim)

        # Decoder
        self.up1 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(c * 4, c * 2)
        self.mod_dec2 = FeatureModulation(c * 2, embedding_dim)

        self.up3 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(c * 2, c)

        # 3-Channel Residual Head
        self.outc = nn.Conv2d(c, out_channels, kernel_size=3, padding=1)

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: Noisy RGB image [B, 3, H, W] (or [B, 4, H, W] for concatenation ablation)
            z: Degradation embedding vector [B, embedding_dim]
        Returns:
            x_hat: Restored clean image [B, 3, H, W]
        """
        # Encoder
        x1 = self.inc(y)
        x2 = self.down1(x1)
        x2 = self.mod_enc2(x2, z)

        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        b = self.bottleneck(x4)
        b = self.mod_bottleneck(b, z)

        # Decoder
        u1 = self.up1(b)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)
        u2 = self.mod_dec2(u2, z)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv_up3(u3)

        # 3-Channel Residual Prediction and Subtraction from RGB channels
        r_hat_rgb = self.outc(u3)
        y_rgb = y[:, :3]  # Extracts RGB channels even if 4-channel ablation input is used
        x_hat = torch.clamp(y_rgb - r_hat_rgb, 0.0, 1.0)
        return x_hat


class AdaptiveDenoiser(nn.Module):
    """
    Composite Standalone Adaptive Denoiser for Training, Inference & Edge Deployment.
    
    Primary Model E Architecture:
    Noisy RGB image Y [B, 3, H, W]
           │
           ├──────────────► Adaptive U-Net (3-channel RGB input)
           │                    ▲
           ▼                    │
    Noise Estimator             │
           │                    │
           ▼                    │
    Noise-strength map M_hat    │
           │                    │
           ▼                    │
    Noise Encoder ──► z ────────┘
                             │
                             ▼
              FiLM-Style Feature Modulation
                             │
                             ▼
                        Clean image X_hat
                        
    Supports:
    - z_override: For genuine z_low Wrong-Condition sensitivity test
    - m_hat_override: For Oracle vs Predicted noise map ablation
    - use_input_concat_ablation: Optional 4-channel [R, G, B, M_hat] ablation model
    """
    def __init__(
        self,
        base_channels: int = 32,
        embedding_dim: int = 32,
        use_input_concat_ablation: bool = False
    ):
        super().__init__()
        self.use_input_concat = use_input_concat_ablation
        self.noise_estimator = SpatialNoiseEstimator(in_channels=3, hidden_channels=32)
        self.noise_encoder = NoiseEncoder(in_channels=1, embedding_dim=embedding_dim)
        
        in_ch = 4 if use_input_concat_ablation else 3
        self.adaptive_unet = AdaptiveNoiseAwareUNet(
            in_channels=in_ch,
            out_channels=3,
            base_channels=base_channels,
            embedding_dim=embedding_dim
        )

    def forward(
        self,
        y: torch.Tensor,
        z_override: torch.Tensor = None,
        m_hat_override: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            y: Noisy RGB image [B, 3, H, W]
            z_override: Optional override for z (used in Wrong-Condition Ablation)
            m_hat_override: Optional override for m_hat (used in Oracle Noise Map Ablation)
        Returns:
            x_hat: Denoised clean RGB image [B, 3, H, W]
            m_hat: Estimated noise-strength map [B, 1, H, W]
            z: Degradation embedding vector [B, embedding_dim]
        """
        # Step 1: Spatial Noise-Strength Estimation
        if m_hat_override is not None:
            m_hat = m_hat_override
        else:
            m_hat = self.noise_estimator(y)

        # Step 2: Degradation Representation Encoding
        if z_override is not None:
            z = z_override
        else:
            z = self.noise_encoder(m_hat)

        # Step 3: Adaptive Feature-Modulated Restoration
        if self.use_input_concat:
            y_in = torch.cat([y, m_hat], dim=1)
        else:
            y_in = y

        x_hat = self.adaptive_unet(y_in, z)
        return x_hat, m_hat, z

    def forward_with_conditioning_mode(
        self,
        y: torch.Tensor,
        x_clean: torch.Tensor = None,
        mode: str = "normal",
        z_wrong: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Supports systematic scientific conditioning ablations:
        - 'normal': Standard autonomous Scout CNN + Noise Encoder inference.
        - 'zero': Zero degradation embedding z = 0.
        - 'wrong': Injected wrong/mismatched z embedding.
        - 'shuffled': Batch-shuffled z embedding across samples.
        - 'oracle': Ground-truth degradation map M_target encoded to z.
        """
        mode = mode.lower()
        if mode != "oracle" and x_clean is not None:
            raise ValueError(
                f"GROUND-TRUTH LEAKAGE DETECTED! 'x_clean' was provided to forward_with_conditioning_mode "
                f"under mode='{mode}'. Clean ground-truth images are strictly prohibited during normal, "
                f"zero, shuffled, or wrong conditioning inference to prevent research integrity violations."
            )

        if mode == "oracle":
            if x_clean is None:
                raise ValueError("Oracle conditioning requires ground-truth clean reference x_clean!")
            m_oracle = compute_noise_strength_target(y, x_clean)
            return self.forward(y, m_hat_override=m_oracle)
        elif mode == "zero":
            B = y.shape[0]
            z_zero = torch.zeros(B, self.noise_encoder.fc.out_features, device=y.device)
            return self.forward(y, z_override=z_zero)
        elif mode == "shuffled":
            m_hat = self.noise_estimator(y)
            z = self.noise_encoder(m_hat)
            B = z.shape[0]
            perm = torch.randperm(B) if B > 1 else torch.zeros(1, dtype=torch.long)
            z_shuffled = z[perm]
            return self.forward(y, z_override=z_shuffled)
        elif mode == "wrong":
            if z_wrong is None:
                m_hat = self.noise_estimator(y)
                z = self.noise_encoder(m_hat)
                z_wrong = -z
            return self.forward(y, z_override=z_wrong)
        elif mode == "normal":
            return self.forward(y)
        else:
            raise ValueError(f"Unknown conditioning mode '{mode}'! Supported: 'normal', 'zero', 'wrong', 'shuffled', 'oracle'.")
