import torch
import numpy as np
from .isp_pipeline import ISPPipeline

class SNICNoiseGenerator:
    """
    SNIC-Inspired Heteroscedastic Sensor-Noise Synthesis in Simplified Camera-Space RGB.
    
    SCIENTIFIC FORMULATION & MODELING PRINCIPLES:
    Given clean linear optical intensity I in camera space:
        1. Shot Noise (Quantum Photon Poisson Process):
           N ~ Poisson( I / a )
           I_shot = a * N
           where 'a' is the photon-to-digital scaling parameter: a = k * ISO^m.
        2. Read Noise (Thermal & Sensor Electronic Circuit Noise):
           epsilon_read ~ Normal(0, b)
           where 'b' is the total read-noise variance: b = b_base * (ISO / 100)^p.
        3. Observed Camera-Space Signal:
           Y_cam = I_shot + epsilon_read
           Var(Y_cam | I) = a * I + b  (Heteroscedastic Poisson-Gaussian Model).
        4. Camera Rendering:
           sRGB_noisy = Camera_ISP(Y_cam).
           
    PROVENANCE DISCLOSURE:
    Parameters are heuristic approximations inspired by camera noise profiles,
    NOT physically measured calibration curves from laboratory dark frames.
    """
    def __init__(self, sensor_type: str = "iphone11_main", isp_gamma: float = 2.2):
        self.isp = ISPPipeline(gamma=isp_gamma, use_exact_srgb=True)
        self.sensor_type = sensor_type

        # Sensor parameter configurations with explicit provenance declarations
        self.profiles = {
            "iphone11_main": {
                "name": "Apple iPhone 11 Main Camera Proxy",
                "k": 1.2e-6,
                "m": 1.15,
                "read_var_base": 4.5e-5,
                "read_power": 1.0,
                "iso_range": (100.0, 3200.0),
                "provenance": "heuristic / estimated parameterization",
                "is_measured": False
            },
            "iphone11_tele": {
                "name": "Apple iPhone 11 Telephoto Camera Proxy",
                "k": 1.5e-6,
                "m": 1.12,
                "read_var_base": 5.0e-5,
                "read_power": 1.0,
                "iso_range": (100.0, 3200.0),
                "provenance": "heuristic / estimated parameterization",
                "is_measured": False
            },
            "sony_rx100": {
                "name": "Sony RX100 Compact Sensor Proxy",
                "k": 8.0e-7,
                "m": 1.05,
                "read_var_base": 1.2e-5,
                "read_power": 1.0,
                "iso_range": (125.0, 6400.0),
                "provenance": "heuristic / estimated parameterization",
                "is_measured": False
            },
            "sony_a7r3": {
                "name": "Sony Alpha 7R III Full-Frame Sensor Proxy",
                "k": 5.0e-7,
                "m": 1.00,
                "read_var_base": 8.0e-6,
                "read_power": 1.0,
                "iso_range": (100.0, 12800.0),
                "provenance": "heuristic / estimated parameterization",
                "is_measured": False
            }
        }
        # Explicit aliases ensuring provenance clarity
        self.profiles["mobile_sensor_inspired_heuristic"] = self.profiles["iphone11_main"]
        self.profiles["iphone11_inspired_heuristic"] = self.profiles["iphone11_main"]

        if sensor_type not in self.profiles:
            sensor_type = "mobile_sensor_inspired_heuristic"
        self.profile = self.profiles[sensor_type]

    def compute_shot_scale(self, iso: float) -> float:
        """Calculates photon-to-digital scaling factor: a = k * ISO^m."""
        return float(self.profile["k"] * (iso ** self.profile["m"]))

    def compute_read_variance(self, iso: float) -> float:
        """Calculates read noise variance: b = b_base * (ISO / 100)^p."""
        return float(self.profile["read_var_base"] * ((iso / 100.0) ** self.profile.get("read_power", 1.0)))

    @torch.no_grad()
    def add_noise(self, srgb_clean: torch.Tensor, iso: float = 1600.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Injects heteroscedastic Poisson-Gaussian sensor noise.
        
        Args:
            srgb_clean: Clean sRGB image tensor [B, 3, H, W] in [0, 1]
            iso: Synthetic camera sensitivity ISO rating
        Returns:
            srgb_noisy: Degradation-corrupted sRGB image tensor [B, 3, H, W] in [0, 1]
            noise_residual: Observed RGB residual tensor (srgb_noisy - srgb_clean)
        """
        device = srgb_clean.device
        self.isp.to(device)

        # Step 1: Invert simplified ISP to camera-space intensity
        cam_clean = self.isp.srgb_to_camera_space(srgb_clean)
        
        # Ensure non-negativity for physical optical intensity
        cam_clean = torch.clamp(cam_clean, min=0.0)

        # Step 2: Poisson Shot Noise using exact torch.poisson()
        a = self.compute_shot_scale(iso)
        # Intensity scaled to photon expectation: lambda = I / a
        photons = cam_clean / a
        # torch.poisson() supports both CPU and CUDA
        shot_photons = torch.poisson(photons)
        cam_shot = shot_photons * a

        # Step 3: Additive Zero-Mean Gaussian Read Noise
        b = self.compute_read_variance(iso)
        std_read = np.sqrt(b)
        read_noise = torch.randn_like(cam_shot) * std_read
        
        cam_noisy = cam_shot + read_noise

        # Step 4: Render through forward simplified ISP to sRGB space
        srgb_noisy = self.isp.camera_space_to_srgb(cam_noisy)
        noise_residual = srgb_noisy - srgb_clean

        return srgb_noisy, noise_residual
