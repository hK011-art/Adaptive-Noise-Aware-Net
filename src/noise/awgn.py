import torch

class AWGNGenerator:
    """
    Additive White Gaussian Noise (AWGN) Generator.
    Applies homoscedastic, signal-independent Gaussian noise to image tensors [B, C, H, W] in range [0, 1].
    """
    def __init__(self, sigma_min: float = 0.0, sigma_max: float = 75.0):
        self.sigma_min = sigma_min / 255.0
        self.sigma_max = sigma_max / 255.0

    def add_noise(self, img: torch.Tensor, sigma: float = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Adds AWGN to the input image tensor.
        
        Args:
            img (torch.Tensor): Clean image tensor in range [0, 1].
            sigma (float, optional): Fixed sigma value in [0, 255]. If None, sampled uniformly in [sigma_min, sigma_max].
            
        Returns:
            noisy_img (torch.Tensor): Noisy image clamped to [0, 1].
            noise (torch.Tensor): The actual added noise map.
        """
        if sigma is None:
            sigma_val = torch.empty(img.shape[0], 1, 1, 1, device=img.device).uniform_(self.sigma_min, self.sigma_max)
        else:
            sigma_val = (sigma / 255.0)

        noise = torch.randn_like(img) * sigma_val
        noisy_img = torch.clamp(img + noise, 0.0, 1.0)
        return noisy_img, noise

