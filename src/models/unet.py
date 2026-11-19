import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BN/LN] => LeakyReLU) * 2 with Residual Connection"""
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

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class RestorationUNet(nn.Module):
    """
    Lightweight 4-Stage Restoration U-Net Architecture.
    Predicts clean restored image given noisy input image.
    
    Default channels C=64 (Unet-S configuration from Noise-DA paper).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels
        
        # Encoder
        self.inc = DoubleConv(in_channels, c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c, c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 2, c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 4, c * 8))

        # Bottleneck
        self.bottleneck = DoubleConv(c * 8, c * 8)

        # Decoder
        self.up1 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(c * 4, c * 2)

        self.up3 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(c * 2, c)

        # Final Reconstruction Head
        self.outc = nn.Conv2d(c, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        b = self.bottleneck(x4)

        u1 = self.up1(b)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv_up3(u3)

        out = self.outc(u3)
        return torch.clamp(x - out, 0.0, 1.0) # Residual learning y_restored = x - predicted_noise

