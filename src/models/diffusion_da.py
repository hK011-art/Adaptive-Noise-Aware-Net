import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """Timestep Embedding for Diffusion Timestep t"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionUNet(nn.Module):
    r"""
    Conditional Denoising Diffusion U-Net (\epsilon_\theta).
    Predicts added Gaussian noise \epsilon given noisy latent \tilde{y}^s and concatenated condition channels C(\hat{y}^s, \hat{y}^r).
    """
    def __init__(self, in_channels: int = 9, out_channels: int = 3, time_emb_dim: int = 128):
        super().__init__()
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU()
        )

        # Feature channels: [64, 128, 256]
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.down1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        self.mid = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU()
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)

        self.conv_out = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # Concatenate noisy latent x [B, 3, H, W] and condition tensor cond [B, 6, H, W] -> [B, 9, H, W]
        x_in = torch.cat([x, cond], dim=1)
        
        t_emb = self.time_mlp(time)[:, :, None, None]

        h1 = F.gelu(self.conv_in(x_in))
        h2 = F.gelu(self.down1(h1) + t_emb)
        h3 = F.gelu(self.down2(h2))

        m = self.mid(h3)

        u2 = F.gelu(self.up2(m))
        u2_cat = torch.cat([u2, h2], dim=1)

        u1 = F.gelu(self.up1(u2_cat))
        u1_cat = torch.cat([u1, h1], dim=1)

        return self.conv_out(u1_cat)


class NoiseDADiffusionLoss(nn.Module):
    """
    Implements Vectorized Per-Sample Channel Shuffling Layer (f_{cs})
    and Residual-Swapping Contrastive Loss (L_{Con}) from Noise-DA (ICLR 2025).
    """
    def __init__(self, timesteps: int = 1000, margin: float = 0.1):
        super().__init__()
        self.timesteps = timesteps
        self.margin = margin
        self.diffusion_model = DiffusionUNet(in_channels=9, out_channels=3)

        # Precompute linear beta schedule
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def channel_shuffle(self, y_s: torch.Tensor, y_r: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
        """
        Per-Sample Channel Shuffling Layer (f_{cs}):
        Independently and stochastically swaps condition ordering for each sample in the batch.
        """
        if not self.training:
            return torch.cat([y_s, y_r], dim=1)

        B = y_s.shape[0]
        # Generate independent per-sample coin flips
        rand_vals = torch.rand((B, 1, 1, 1), device=y_s.device, generator=generator)
        swap_mask = (rand_vals > 0.5).float()

        cond_order_1 = torch.cat([y_s, y_r], dim=1)
        cond_order_2 = torch.cat([y_r, y_s], dim=1)

        return swap_mask * cond_order_2 + (1.0 - swap_mask) * cond_order_1

    def compute_diffusion_loss(self, eps_pos: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Calculates standard diffusion MSE noise-prediction loss."""
        return F.mse_loss(eps_pos, eps)

    def compute_contrastive_loss(self, eps: torch.Tensor, eps_pos: torch.Tensor, eps_neg: torch.Tensor) -> torch.Tensor:
        """Calculates contrastive hinge loss separating positive from negative swapped residuals."""
        dist_pos = torch.mean((eps - eps_pos) ** 2, dim=[1, 2, 3])
        dist_neg = torch.mean((eps - eps_neg) ** 2, dim=[1, 2, 3])
        return torch.mean(torch.clamp(dist_pos - dist_neg + self.margin, min=0.0))

    def forward(
        self,
        y_clean_s: torch.Tensor,
        x_s: torch.Tensor,
        y_hat_s: torch.Tensor,
        x_r: torch.Tensor,
        y_hat_r: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates joint diffusion MSE and residual-swapping contrastive loss.
        Returns:
            loss_total: 0.5 * loss_mse + 0.5 * loss_con
            loss_mse: Diffusion noise prediction MSE
            loss_con: Residual-swapping contrastive hinge loss
        """
        B, C, H, W = y_clean_s.shape
        device = y_clean_s.device

        # Sample random timestep t
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        eps = torch.randn_like(y_clean_s)

        # Corrupt clean synthetic image y_clean_s to timestep t
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        y_tilde_s = sqrt_alpha_bar * y_clean_s + sqrt_one_minus_alpha_bar * eps

        # 1. Positive Condition: Correct restored outputs C(y_hat_s, y_hat_r) with per-sample shuffling
        cond_pos = self.channel_shuffle(y_hat_s, y_hat_r)
        eps_pos = self.diffusion_model(y_tilde_s, cond_pos, t)
        loss_mse = self.compute_diffusion_loss(eps_pos, eps)

        # 2. Residual-Swapping Negative Condition
        R_s = y_hat_s - x_s  # Synthetic residual shift
        R_r = y_hat_r - x_r  # Real residual shift

        y_hat_s_from_r = torch.clamp(x_s + R_r, 0.0, 1.0)
        y_hat_r_from_s = torch.clamp(x_r + R_s, 0.0, 1.0)

        cond_neg = self.channel_shuffle(y_hat_s_from_r, y_hat_r_from_s)
        eps_neg = self.diffusion_model(y_tilde_s, cond_neg, t)

        # 3. Contrastive Hinge Loss
        loss_con = self.compute_contrastive_loss(eps, eps_pos, eps_neg)

        loss_total = 0.5 * loss_mse + 0.5 * loss_con
        return loss_total, loss_mse, loss_con
