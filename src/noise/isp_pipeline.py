import torch
import torch.nn as nn
import torch.nn.functional as F

class ISPPipeline(nn.Module):
    """
    Simplified Camera-Space RGB Rendering Model.
    
    SCIENTIFIC SCOPE & LIMITATION DISCLOSURE:
    This module simulates simplified camera-space RGB transformations:
        sRGB <-> Linear RGB <-> Simplified Camera-Space RGB
    
    It does NOT implement Bayer Color Filter Array (CFA) subsampling,
    true mosaiced sensor RAW readouts, demosaicing algorithms, or full
    sensor-calibrated tone reproduction curves.
    """
    def __init__(self, gamma: float = 2.2, use_exact_srgb: bool = True):
        super().__init__()
        self.gamma = gamma
        self.use_exact_srgb = use_exact_srgb

        # Camera-space color transformation matrix (sRGB D65 to sensor spectral response proxy)
        self.register_buffer("rgb2cam", torch.tensor([
            [1.0234, -0.2969, -0.2266],
            [-0.5625, 1.6328, -0.0469],
            [-0.0703, 0.2188, 0.6406]
        ], dtype=torch.float32))

        # Exact algebraic inverse to ensure round-trip invertibility
        self.register_buffer("cam2rgb", torch.inverse(self.rgb2cam))

    def srgb_to_linear(self, srgb: torch.Tensor) -> torch.Tensor:
        """Linearizes sRGB values into linear optical intensity space."""
        srgb = torch.clamp(srgb, 0.0, 1.0)
        if self.use_exact_srgb:
            # Standard IEC 61966-2-1 piecewise sRGB transfer function
            mask = srgb <= 0.04045
            linear = torch.where(mask, srgb / 12.92, torch.pow((srgb + 0.055) / 1.055, 2.4))
        else:
            # Simplified power-law approximation
            linear = torch.pow(srgb, self.gamma)
        return linear

    def linear_to_srgb(self, linear: torch.Tensor) -> torch.Tensor:
        """Applies non-linear tone curve to convert linear RGB to sRGB display space."""
        linear_clamped = torch.clamp(linear, min=0.0)
        if self.use_exact_srgb:
            mask = linear_clamped <= 0.0031308
            srgb = torch.where(mask, 12.92 * linear_clamped, 1.055 * torch.pow(linear_clamped, 1.0 / 2.4) - 0.055)
        else:
            srgb = torch.pow(linear_clamped, 1.0 / self.gamma)
        return torch.clamp(srgb, 0.0, 1.0)

    def camera_matrix_forward(self, img: torch.Tensor) -> torch.Tensor:
        """Maps linear RGB into simplified camera RGB space: C_cam = M * C_rgb."""
        B, C, H, W = img.shape
        img_flat = img.permute(0, 2, 3, 1).reshape(-1, 3)
        cam_flat = torch.matmul(img_flat, self.rgb2cam.T)
        return cam_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    def camera_matrix_inverse(self, img: torch.Tensor) -> torch.Tensor:
        """Maps simplified camera RGB back into linear RGB space: C_rgb = M^-1 * C_cam."""
        B, C, H, W = img.shape
        img_flat = img.permute(0, 2, 3, 1).reshape(-1, 3)
        rgb_flat = torch.matmul(img_flat, self.cam2rgb.T)
        return rgb_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    def srgb_to_camera_space(self, srgb: torch.Tensor) -> torch.Tensor:
        """
        Converts sRGB to linear camera-space representation.
        Internal clamping is intentionally avoided to preserve exact algebraic invertibility.
        """
        linear = self.srgb_to_linear(srgb)
        cam_space = self.camera_matrix_forward(linear)
        return cam_space

    def camera_space_to_srgb(self, cam_space: torch.Tensor) -> torch.Tensor:
        """
        Converts linear camera-space tensor back to sRGB.
        Clamping is applied ONLY at the final sRGB display boundary [0, 1].
        """
        linear = self.camera_matrix_inverse(cam_space)
        srgb = self.linear_to_srgb(linear)
        return srgb

    # Backwards-compatible aliases with clear documentation
    def srgb_to_raw(self, srgb: torch.Tensor) -> torch.Tensor:
        return self.srgb_to_camera_space(srgb)

    def raw_to_srgb(self, raw_cam: torch.Tensor) -> torch.Tensor:
        return self.camera_space_to_srgb(raw_cam)
