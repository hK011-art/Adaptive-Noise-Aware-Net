import os
import argparse
import torch
import numpy as np
from PIL import Image
from src.models.noise_estimator import SpatialNoiseEstimator, compute_noise_strength_target
from src.noise.snic_noise import SNICNoiseGenerator

def inspect_and_visualize_degradation_map(image_path: str = "test_street.jpg", iso: float = 3200.0, output_path: str = "degradation_map_analysis.png"):
    """
    Computes and inspects the spatial degradation magnitude map M.
    Prints raw quantitative statistics without misleading visual normalizations.
    """
    if not os.path.exists(image_path):
        print(f"[Notice] Image '{image_path}' not found. Using deterministic test patch.")
        clean_np = np.ones((128, 128, 3), dtype=np.float32) * 0.5
    else:
        clean_pil = Image.open(image_path).convert("RGB")
        clean_np = np.array(clean_pil, dtype=np.float32) / 255.0

    clean_tensor = torch.from_numpy(clean_np).permute(2, 0, 1).unsqueeze(0)
    
    # Inject noise
    noise_gen = SNICNoiseGenerator(sensor_type="iphone11_main")
    noisy_tensor, residual_tensor = noise_gen.add_noise(clean_tensor, iso=iso)

    # Compute observed degradation magnitude target
    m_target = compute_noise_strength_target(noisy_tensor, clean_tensor)

    # Scout CNN prediction
    estimator = SpatialNoiseEstimator(in_channels=3, hidden_channels=32)
    estimator.eval()
    with torch.no_grad():
        m_pred = estimator(noisy_tensor)

    # Quantitative Statistics (Raw Unnormalized Metric Values)
    m_tgt_np = m_target.squeeze().numpy()
    m_prd_np = m_pred.squeeze().numpy()
    map_abs_diff = np.abs(m_tgt_np - m_prd_np)

    print("============================================================")
    print("     SPATIAL DEGRADATION MAGNITUDE MAP STATISTICS           ")
    print("============================================================")
    print(f"Target Formula: M = sqrt( 1/3 * sum_c (Y_c - X_c)^2 + eps )")
    print(f"Target Map Shape: {m_target.shape} | Non-Negative: {bool((m_target >= 0).all())}\n")

    print(f"Target Map M_target Quantitative Statistics:")
    print(f"  Mean:        {float(np.mean(m_tgt_np)):.6f}")
    print(f"  Std Dev:     {float(np.std(m_tgt_np)):.6f}")
    print(f"  Min:         {float(np.min(m_tgt_np)):.6f}")
    print(f"  Max:         {float(np.max(m_tgt_np)):.6f}")
    print(f"  Median:      {float(np.median(m_tgt_np)):.6f}")
    print(f"  95th %ile:   {float(np.percentile(m_tgt_np, 95)):.6f}\n")

    print(f"Predicted Map M_hat Statistics (Untrained/Initial State):")
    print(f"  Mean:        {float(np.mean(m_prd_np)):.6f}")
    print(f"  Std Dev:     {float(np.std(m_prd_np)):.6f}")
    print(f"  Mean Absolute Error (|M_target - M_hat|): {float(np.mean(map_abs_diff)):.6f}")
    print("============================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Spatial Degradation Magnitude Map")
    parser.add_argument("--image", type=str, default="test_street.jpg")
    parser.add_argument("--iso", type=float, default=3200.0)
    args = parser.parse_args()
    inspect_and_visualize_degradation_map(args.image, args.iso)

