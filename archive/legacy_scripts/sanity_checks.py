import torch
import torch.nn as nn
import numpy as np
from src.models.noise_estimator import SpatialNoiseEstimator, NoiseEncoder, compute_noise_strength_target
from src.models.adaptive_unet import FeatureModulation, AdaptiveNoiseAwareUNet, AdaptiveDenoiser

def run_all_sanity_checks():
    print("==========================================================================")
    print("        ADAPTIVE NOISE-AWARE U-NET: RIGOROUS SANITY CHECK SUITE           ")
    print("==========================================================================\n")
    all_passed = True

    # -------------------------------------------------------------------------
    # TEST 1: Exact RMS Noise-Strength Target Formula & Dimensionality
    # -------------------------------------------------------------------------
    print("[Check 1/6] Verifying Exact RMS Noise-Strength Target Formula & Shapes...")
    B, H, W = 2, 64, 64
    x_clean = torch.rand(B, 3, H, W)
    y_noisy = torch.clamp(x_clean + torch.randn(B, 3, H, W) * 0.1, 0.0, 1.0)
    
    m_target = compute_noise_strength_target(y_noisy, x_clean)
    assert m_target.shape == (B, 1, H, W), f"Expected shape {(B, 1, H, W)}, got {m_target.shape}"
    assert (m_target >= 0.0).all(), "M_target contains negative values!"

    # Manual numerical verification of RMS formula
    diff_r = y_noisy[:, 0:1] - x_clean[:, 0:1]
    diff_g = y_noisy[:, 1:2] - x_clean[:, 1:2]
    diff_b = y_noisy[:, 2:3] - x_clean[:, 2:3]
    manual_rms = torch.sqrt((diff_r ** 2 + diff_g ** 2 + diff_b ** 2) / 3.0)
    rms_diff = torch.max(torch.abs(m_target - manual_rms)).item()
    assert rms_diff < 1e-6, f"M_target mismatch with manual RMS! Max diff: {rms_diff}"
    print(f"  --> PASSED: M_target shape {m_target.shape}, non-negative, matches exact RMS formula (max diff: {rms_diff:.2e})\n")

    # -------------------------------------------------------------------------
    # TEST 2: Noise Estimator CNN (Resolution-Preserving, Non-Negative)
    # -------------------------------------------------------------------------
    print("[Check 2/6] Verifying Noise Estimator CNN...")
    noise_estimator = SpatialNoiseEstimator(in_channels=3, hidden_channels=32)
    m_hat = noise_estimator(y_noisy)
    assert m_hat.shape == (B, 1, H, W), f"Expected shape {(B, 1, H, W)}, got {m_hat.shape}"
    assert (m_hat >= 0.0).all(), "M_hat contains negative values! Final layer must use non-negative activation."
    print(f"  --> PASSED: M_hat shape {m_hat.shape}, strictly non-negative, preserves spatial resolution.\n")

    # -------------------------------------------------------------------------
    # TEST 3: Noise Encoder (Latent Degradation Embedding z in R^32)
    # -------------------------------------------------------------------------
    print("[Check 3/6] Verifying Degradation Representation Encoder...")
    noise_encoder = NoiseEncoder(in_channels=1, embedding_dim=32)
    z = noise_encoder(m_hat)
    assert z.shape == (B, 32), f"Expected shape {(B, 32)}, got {z.shape}"
    assert torch.isfinite(z).all(), "z contains NaN or Inf values!"
    print(f"  --> PASSED: z shape {z.shape}, compact latent embedding vector in R^32.\n")

    # -------------------------------------------------------------------------
    # TEST 4: FiLM Zero-Initialization & Identity Modulation
    # -------------------------------------------------------------------------
    print("[Check 4/6] Verifying FiLM Feature Modulation Zero-Initialization...")
    feat_dim = 64
    feat_mod = FeatureModulation(feature_channels=feat_dim, embedding_dim=32)
    dummy_feat = torch.randn(B, feat_dim, 32, 32)
    dummy_z = torch.randn(B, 32)
    
    # Check weights are zero-initialized
    assert torch.all(feat_mod.fc_gamma.weight == 0.0), "fc_gamma.weight not zero-initialized!"
    assert torch.all(feat_mod.fc_beta.weight == 0.0), "fc_beta.weight not zero-initialized!"
    
    mod_feat = feat_mod(dummy_feat, dummy_z)
    identity_diff = torch.max(torch.abs(mod_feat - dummy_feat)).item()
    assert identity_diff < 1e-6, f"FiLM did not produce identity transform at initialization! Max diff: {identity_diff}"
    print(f"  --> PASSED: Zero-initialization verified. Initial modulation is exact identity (diff: {identity_diff:.2e})\n")

    # -------------------------------------------------------------------------
    # TEST 5: Adaptive U-Net 3-Channel Residual & Clamped Clean Output
    # -------------------------------------------------------------------------
    print("[Check 5/6] Verifying Adaptive U-Net Architecture & Decoupled Output...")
    adaptive_unet = AdaptiveNoiseAwareUNet(in_channels=3, out_channels=3, base_channels=32, embedding_dim=32)
    x_hat = adaptive_unet(y_noisy, z)
    assert x_hat.shape == (B, 3, H, W), f"Expected shape {(B, 3, H, W)}, got {x_hat.shape}"
    assert (x_hat >= 0.0).all() and (x_hat <= 1.0).all(), "X_hat values outside [0, 1] range!"
    print(f"  --> PASSED: X_hat shape {x_hat.shape}, strictly bounded in [0.0, 1.0].\n")

    # -------------------------------------------------------------------------
    # TEST 6: End-to-End Gradient Flow (No Detach in Conditioning Path)
    # -------------------------------------------------------------------------
    print("[Check 6/6] Verifying End-to-End Gradient Flow Across All Modules...")
    denoiser = AdaptiveDenoiser(base_channels=32, embedding_dim=32)
    # FiLM layers are zero-initialized by default (verified in Check 4).
    # To test that the computational graph connects end-to-end through z without any .detach(),
    # assign small non-zero weights to the FiLM projections for this connectivity test:
    for mod in [denoiser.adaptive_unet.mod_enc2, denoiser.adaptive_unet.mod_bottleneck, denoiser.adaptive_unet.mod_dec2]:
        nn.init.normal_(mod.fc_gamma.weight, std=0.01)
        nn.init.normal_(mod.fc_beta.weight, std=0.01)
    denoiser.train()
    
    x_pred, m_pred, z_emb = denoiser(y_noisy)
    loss_rec = torch.mean(torch.abs(x_pred - x_clean))
    loss_noise = torch.mean(torch.abs(m_pred - m_target))
    loss_total = loss_rec + 0.5 * loss_noise
    
    loss_total.backward()
    
    # Verify gradients reach each sub-module
    estimator_grad = False
    for p in denoiser.noise_estimator.parameters():
        if p.grad is not None and torch.norm(p.grad) > 0:
            estimator_grad = True
            break
            
    encoder_grad = False
    for p in denoiser.noise_encoder.parameters():
        if p.grad is not None and torch.norm(p.grad) > 0:
            encoder_grad = True
            break

    unet_grad = False
    for p in denoiser.adaptive_unet.parameters():
        if p.grad is not None and torch.norm(p.grad) > 0:
            unet_grad = True
            break

    assert estimator_grad, "NoiseEstimator received NO gradients!"
    assert encoder_grad, "NoiseEncoder received NO gradients!"
    assert unet_grad, "AdaptiveUNet received NO gradients!"
    print(f"  --> PASSED: End-to-end gradients verified across NoiseEstimator, NoiseEncoder, FiLM layers, and U-Net!\n")

    print("==========================================================================")
    print(" >>> ALL 6 RIGOROUS MATHEMATICAL & DIMENSIONAL SANITY CHECKS PASSED! <<<")
    print("==========================================================================\n")

if __name__ == "__main__":
    run_all_sanity_checks()

