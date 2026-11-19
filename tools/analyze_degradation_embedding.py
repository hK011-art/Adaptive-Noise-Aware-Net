import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from src.models.adaptive_unet import AdaptiveDenoiser
from src.noise.snic_noise import SNICNoiseGenerator

def analyze_degradation_embeddings(checkpoint_path: str = None, device: str = "cpu"):
    """
    Analyzes whether latent degradation embedding z in R^32 encodes true degradation
    (noise intensity / sensor properties) or inadvertently encodes image scene content.
    """
    dev = torch.device(device)
    denoiser = AdaptiveDenoiser(base_channels=32, embedding_dim=32).to(dev)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=dev)
        if isinstance(state, dict) and "restoration_state_dict" in state:
            denoiser.load_state_dict(state["restoration_state_dict"])
        else:
            denoiser.load_state_dict(state)
        print(f"[Analysis] Loaded model weights from {checkpoint_path}")
    else:
        print(f"[Analysis Note] No checkpoint found. Running diagnostic on current model structure.")

    denoiser.eval()
    noise_gen = SNICNoiseGenerator(sensor_type="iphone11_main")

    iso_list = [100.0, 400.0, 800.0, 1600.0, 3200.0]

    # Test 1: Same image content across varying ISO levels
    print("\n--- Diagnostic 1: Same Content, Varying ISO Levels ---")
    torch.manual_seed(42)
    clean_sample = torch.rand(1, 3, 128, 128, device=dev)

    z_vectors = {}
    with torch.no_grad():
        for iso in iso_list:
            noisy, _ = noise_gen.add_noise(clean_sample, iso=iso)
            _, _, z = denoiser(noisy)
            z_vectors[iso] = z.squeeze(0)
            norm = torch.norm(z).item()
            mean_val = torch.mean(z).item()
            std_val = torch.std(z).item()
            print(f"  ISO {iso:4.0f} -> z L2-norm: {norm:.4f} | mean: {mean_val:.4f} | std: {std_val:.4f}")

    # Compute pairwise cosine similarity matrix
    print("\n  Pairwise Cosine Similarity Matrix between ISO levels:")
    print("        " + "".join([f"ISO {int(iso):<6}" for iso in iso_list]))
    for iso_i in iso_list:
        row = [f"ISO {int(iso_i):<4}"]
        for iso_j in iso_list:
            sim = F.cosine_similarity(z_vectors[iso_i].unsqueeze(0), z_vectors[iso_j].unsqueeze(0)).item()
            row.append(f"{sim:.4f}   ")
        print("  " + "".join(row))

    # Test 2: Different image content under IDENTICAL noise level
    print("\n--- Diagnostic 2: Different Image Content, Constant ISO (1600) ---")
    content_samples = [torch.rand(1, 3, 128, 128, device=dev) for _ in range(5)]
    z_content_vectors = []
    with torch.no_grad():
        for i, sample in enumerate(content_samples):
            noisy, _ = noise_gen.add_noise(sample, iso=1600.0)
            _, _, z = denoiser(noisy)
            z_content_vectors.append(z.squeeze(0))

    content_sims = []
    for i in range(len(content_samples)):
        for j in range(i + 1, len(content_samples)):
            sim = F.cosine_similarity(z_content_vectors[i].unsqueeze(0), z_content_vectors[j].unsqueeze(0)).item()
            content_sims.append(sim)

    avg_content_sim = float(np.mean(content_sims))
    print(f"  Average Cosine Similarity across differing content at constant ISO 1600: {avg_content_sim:.4f}")
    
    # Separation ratio
    iso_spread = 1.0 - F.cosine_similarity(z_vectors[100.0].unsqueeze(0), z_vectors[3200.0].unsqueeze(0)).item()
    print(f"\n[Diagnostic Summary]")
    print(f"  ISO Separation (1 - CosSim(ISO100, ISO3200)): {iso_spread:.6f}")
    if iso_spread < 1e-3:
        print("  WARNING: LATENT EMBEDDING COLLAPSE DETECTED!")
        print("  z does not separate ISO levels significantly (Cosine Similarity approx 1.0).")
    else:
        print("  OK: z discriminates between varying noise levels.")
    print("------------------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Degradation Embedding z Representation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/Model_E_seed42_best.pth")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    analyze_degradation_embeddings(args.checkpoint, args.device)

