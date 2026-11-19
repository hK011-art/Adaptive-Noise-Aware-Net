import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .unet import RestorationUNet
from .adaptive_unet import AdaptiveDenoiser
from .noise_estimator import compute_noise_strength_target
from .diffusion_da import NoiseDADiffusionLoss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (Robust Smooth L1 norm variant): sqrt(||y_hat - y||^2 + eps^2)"""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps_sq = eps ** 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps_sq))


class NoiseDATrainer(nn.Module):
    """
    Unified Noise-DA & Adaptive Restoration Joint Trainer.
    Supports both Fixed Restoration U-Net (Models A, B, C, D) and
    Adaptive Degradation-Aware Restoration U-Net (Model E).
    
    Notation:
        Y_s: Synthetic noisy image tensor [B, 3, H, W]
        X_s: Clean synthetic ground truth tensor [B, 3, H, W]
        Y_r: Real noisy image tensor [B, 3, H, W] (unlabeled target domain)
    """
    def __init__(
        self,
        use_domain_adaptation: bool = True,
        is_adaptive: bool = False,
        base_channels: int = 32,
        lambda_noise: float = 0.5,
        beta_max: float = 0.2,
        gamma_rate: float = 5.0
    ):
        super().__init__()
        self.use_da = use_domain_adaptation
        self.is_adaptive = is_adaptive
        self.lambda_noise = lambda_noise
        self.beta_max = beta_max
        self.gamma_rate = gamma_rate

        # Initialize network architecture
        if is_adaptive:
            self.restoration_net = AdaptiveDenoiser(base_channels=base_channels, embedding_dim=32)
        else:
            self.restoration_net = RestorationUNet(in_channels=3, out_channels=3, base_channels=base_channels)

        self.l_res = CharbonnierLoss(eps=1e-3)

        if self.use_da:
            self.diffusion_da = NoiseDADiffusionLoss()
        else:
            self.diffusion_da = None

    def compute_restoration_loss(self, pred_clean: torch.Tensor, target_clean: torch.Tensor) -> torch.Tensor:
        """Calculates Charbonnier reconstruction loss L_rec."""
        return self.l_res(pred_clean, target_clean)

    def compute_noise_loss(self, pred_map: torch.Tensor, target_map: torch.Tensor) -> torch.Tensor:
        """Calculates degradation map estimation loss L_noise."""
        return self.l_res(pred_map, target_map)

    def compute_lambda_dif(self, current_epoch: int, total_epochs: int) -> float:
        """
        Sigmoid warmup schedule: lambda_{DA}(p) = ((2 / (1 + exp(-gamma * p))) - 1) * beta
        where p = min(current_epoch / total_epochs, 1.0)
        """
        if not self.use_da or total_epochs <= 0:
            return 0.0
        p = min(float(current_epoch) / float(total_epochs), 1.0)
        scale = (2.0 / (1.0 + np.exp(-self.gamma_rate * p))) - 1.0
        return float(scale * self.beta_max)

    def forward(
        self,
        x_s: torch.Tensor,
        y_s: torch.Tensor,
        x_r: torch.Tensor = None,
        current_epoch: int = 0,
        total_epochs: int = 100
    ) -> dict:
        """
        Forward pass during training.
        
        Args:
            x_s: Synthetic noisy image tensor Y^s [B, 3, H, W]
            y_s: Clean synthetic ground truth tensor X^s [B, 3, H, W]
            x_r: Real noisy image tensor Y^r [B, 3, H, W] (unlabeled target domain)
            current_epoch: Current epoch index
            total_epochs: Total number of training epochs
            
        Returns:
            dict containing total_loss, loss_rec, loss_noise, loss_dif, loss_con, lambda_dif, y_hat_s, y_hat_r
        """
        # Multi-domain batch-size alignment guard (ensures identical batch size across domains)
        if self.use_da and x_r is not None:
            if x_r.shape[0] != x_s.shape[0]:
                min_b = min(x_s.shape[0], x_r.shape[0])
                x_s = x_s[:min_b]
                y_s = y_s[:min_b]
                x_r = x_r[:min_b]

        if self.is_adaptive:
            # Model E: Adaptive Degradation-Aware Flow
            x_hat_s, m_hat_s, z_s = self.restoration_net(x_s)
            m_target_s = compute_noise_strength_target(x_s, y_s)

            loss_rec = self.compute_restoration_loss(x_hat_s, y_s)
            loss_noise = self.compute_noise_loss(m_hat_s, m_target_s)
            y_hat_s = x_hat_s
        else:
            # Models A, B, C, D: Fixed Restoration Flow
            y_hat_s = self.restoration_net(x_s)
            loss_rec = self.compute_restoration_loss(y_hat_s, y_s)
            loss_noise = torch.tensor(0.0, device=x_s.device)

        # Domain Adaptation Branch (Noise-DA)
        if self.use_da and x_r is not None:
            if self.is_adaptive:
                x_hat_r, _, _ = self.restoration_net(x_r)
                y_hat_r = x_hat_r
            else:
                y_hat_r = self.restoration_net(x_r)

            loss_dif, loss_mse, loss_con = self.diffusion_da(y_s, x_s, y_hat_s, x_r, y_hat_r)
            lambda_dif = self.compute_lambda_dif(current_epoch, total_epochs)
            total_loss = loss_rec + (self.lambda_noise * loss_noise if self.is_adaptive else 0.0) + lambda_dif * loss_dif
        else:
            y_hat_r = None
            loss_dif = torch.tensor(0.0, device=x_s.device)
            loss_con = torch.tensor(0.0, device=x_s.device)
            lambda_dif = 0.0
            total_loss = loss_rec + (self.lambda_noise * loss_noise if self.is_adaptive else 0.0)

        return {
            "total_loss": total_loss,
            "loss_rec": loss_rec,
            "loss_noise": loss_noise,
            "loss_dif": loss_dif,
            "loss_con": loss_con,
            "lambda_dif": lambda_dif,
            "lambda_noise": (self.lambda_noise if self.is_adaptive else 0.0),
            "y_hat_s": y_hat_s,
            "y_hat_r": y_hat_r
        }
