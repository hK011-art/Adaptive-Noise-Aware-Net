# Neural Network Architecture Specifications
**Project:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Audience:** Research Reviewers, Thesis Examiners, and ML Engineers

---

## 1. Experimental Model Grid

To isolate the contributions of synthetic sensor-noise modeling, diffusion domain adaptation, and degradation-aware FiLM conditioning, the project specifies five capacity-controlled models:

| Model | Noise Synthesis Model | Domain Adaptation | Restoration Network | Base Channels | Param Count |
|---|---|---|---|---|---|
| **Model A** | Additive White Gaussian (AWGN) | None | Fixed Residual U-Net | 64 | ~31.0M |
| **Model B** | Additive White Gaussian (AWGN) | Noise-DA (Diffusion) | Fixed Residual U-Net | 64 | ~31.0M |
| **Model C** | Heteroscedastic Poisson-Gaussian | None | Fixed Residual U-Net | 64 | ~31.0M |
| **Model D** | Heteroscedastic Poisson-Gaussian | Noise-DA (Diffusion) | Fixed Residual U-Net | 32 | ~7.76M |
| **Model E** | Heteroscedastic Poisson-Gaussian | Noise-DA (Diffusion) | Adaptive Noise-Aware U-Net | 32 | ~7.80M |

> **Capacity Matching:**  
> The core scientific hypothesis contrasts **Model D** against **Model E**. Both utilize `base_channels = 32`. Model E's auxiliary Scout CNN (~14.8K params) and Noise Encoder (~25.8K params) add fewer than 0.5% extra parameters, ensuring rigorous capacity parity.

---

## 2. Model E Sub-Modules

### A. Scout CNN (`SpatialNoiseEstimator`)
- **Input:** Noisy RGB image $Y \in \mathbb{R}^{B \times 3 \times H \times W}$
- **Architecture:** 4-layer resolution-preserving CNN
  - Conv2d(3, 16, k=3, p=1) + ReLU
  - Conv2d(16, 32, k=3, p=1) + ReLU
  - Conv2d(32, 32, k=3, p=1) + ReLU
  - Conv2d(32, 1, k=3, p=1) + ReLU
- **Output:** Estimated spatial degradation magnitude map $\hat{M} \in \mathbb{R}^{B \times 1 \times H \times W}$
- **Property:** Strict non-negativity enforced by final ReLU activation.

### B. Noise Encoder (`NoiseEncoder`)
- **Input:** Degradation map $\hat{M} \in \mathbb{R}^{B \times 1 \times H \times W}$
- **Architecture:**
  - Conv2d(1, 16, k=3, s=2, p=1) + LeakyReLU(0.2)
  - Conv2d(16, 32, k=3, s=2, p=1) + LeakyReLU(0.2)
  - AdaptiveAvgPool2d((1, 1)) + Flatten
  - Linear(32, 32)
- **Output:** Compact degradation vector $z \in \mathbb{R}^{B \times 32}$.

### C. FiLM Feature Modulation (`FeatureModulation`)
- **Formula:**
  $$F_l' = (1 + \gamma_l(z)) \odot F_l + \beta_l(z)$$
- **Projections:**
  - $\gamma_l(z) = \mathbf{W}_{\gamma} z + b_{\gamma}$
  - $\beta_l(z) = \mathbf{W}_{\beta} z + b_{\beta}$
- **Initialization:** Zero-initialized ($\mathbf{W}=0, b=0$). At epoch 0:
  $$\gamma_l(z) = 0, \quad \beta_l(z) = 0 \implies F_l' = F_l \quad \text{(Exact identity baseline)}$$
- **Modulation Locations:**
  1. Encoder Stage 2 ($C = 64$)
  2. Bottleneck ($C = 256$)
  3. Decoder Stage 2 ($C = 64$)

### D. Decoupled Residual Head
The U-Net predicts residual noise $\hat{R}_{\text{rgb}} \in \mathbb{R}^{B \times 3 \times H \times W}$. Restored output is:
$$\hat{X} = \text{clamp}(Y - \hat{R}_{\text{rgb}}, 0.0, 1.0)$$

