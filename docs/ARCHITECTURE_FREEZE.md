# AUTHORITATIVE ARCHITECTURE FREEZE SPECIFICATION

**Project Title:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Status:** **FROZEN FOR FINAL EXPERIMENTS**  
**Verification Method:** Programmatic parameter counting and layer inspection  
**Date:** October 2026  

---

## 1. Summary of Authoritative Parameter Counts

All numbers listed below are programmatically calculated from the active codebase. No approximate or manually typed values are used.

| Model / Sub-module | Base Channels | Total Parameters | Trainable Parameters | Capacity vs Model D | Active at Inference |
|---|---|---|---|---|---|
| **Model A** (AWGN + No DA + Fixed U-Net) | 64 | **12,768,131** | **12,768,131** | +299.59% | Yes (Restoration) |
| **Model B** (AWGN + Noise-DA + Fixed U-Net) | 64 | **12,768,131** | **12,768,131** | +299.59% | Yes (Restoration) |
| **Model C** (SNIC + No DA + Fixed U-Net) | 64 | **12,768,131** | **12,768,131** | +299.59% | Yes (Restoration) |
| **Model D** (SNIC + Noise-DA + Fixed U-Net) | 32 | **3,195,331** | **3,195,331** | **Baseline (0.00%)** | Yes (Restoration) |
| **Model E** (SNIC + Noise-DA + Adaptive Denoiser) | 32 | **3,241,156** | **3,241,156** | **+1.43%** | Yes (Autonomous) |
| ↳ *Scout CNN (`SpatialNoiseEstimator`)* | 32 | 14,625 | 14,625 | — | Yes (Inside Model E) |
| ↳ *Noise Encoder (`NoiseEncoder`)* | 32 | 5,856 | 5,856 | — | Yes (Inside Model E) |
| ↳ *Adaptive U-Net (`AdaptiveNoiseAwareUNet`)* | 32 | 3,220,675 | 3,220,675 | — | Yes (Inside Model E) |
| ↳ *FiLM Feature Modulation (sub-component of Adaptive U-Net)* | — | *(25,344)* | *(25,344)* | — | Yes (Already included in U-Net total) |
| **Diffusion Network** (`DiffusionUNet`) | 64/128/256 | **2,058,179** | **2,058,179** | Training Only | **NO (Stripped)** |

> [!IMPORTANT]
> **Near-Capacity-Matched Comparison (Model D vs Model E):**  
> Model D has **3,195,331** parameters and Model E has **3,241,156** parameters.  
> The delta is exactly **+45,825 parameters (+1.43%)**. This comparison is framed as **near-capacity-matched**. This small difference substantially reduces the likelihood that observed performance is explained by parameter count, but does not mathematically eliminate that possibility.

---

## 2. Detailed Architectural Breakdown

### 2.1 Fixed U-Net Architecture (Models A, B, C, D)
- **Input:** $Y \in [0, 1]^{B \times 3 \times H \times W}$ (Noisy RGB image)
- **Output:** Denoised RGB image $\hat{X} = \text{clamp}(Y - \hat{R}, 0.0, 1.0)$
- **Residual Head:** Predicts 3-channel noise residual $\hat{R} \in \mathbb{R}^{B \times 3 \times H \times W}$

```
Layer Sequence:
- Inc: DoubleConv(3, C)
    - Conv2d(3, C, kernel_size=3, padding=1, bias=False) -> BatchNorm2d(C) -> ReLU(inplace=True)
    - Conv2d(C, C, kernel_size=3, padding=1, bias=False) -> BatchNorm2d(C) -> ReLU(inplace=True)
- Down 1: MaxPool2d(kernel_size=2, stride=2) -> DoubleConv(C, 2C)
- Down 2: MaxPool2d(kernel_size=2, stride=2) -> DoubleConv(2C, 4C)
- Down 3: MaxPool2d(kernel_size=2, stride=2) -> DoubleConv(4C, 8C)
- Bottleneck: DoubleConv(8C, 8C)
- Up 1: ConvTranspose2d(8C, 4C, kernel_size=2, stride=2) -> Concat(Skip Down 2) -> DoubleConv(8C, 4C)
- Up 2: ConvTranspose2d(4C, 2C, kernel_size=2, stride=2) -> Concat(Skip Down 1) -> DoubleConv(4C, 2C)
- Up 3: ConvTranspose2d(2C, C, kernel_size=2, stride=2) -> Concat(Skip Inc) -> DoubleConv(2C, C)
- Out Head: Conv2d(C, 3, kernel_size=3, padding=1)
```
- **Models A, B, C:** $C = 64 \implies 12,768,131$ parameters.
- **Model D:** $C = 32 \implies 3,195,331$ parameters.

---

### 2.2 Model E: Composite Adaptive Denoiser (`AdaptiveDenoiser`)

```
Noisy RGB image Y [B, 3, H, W]
       │
       ├──────────────► Adaptive U-Net (3-channel input)
       │                    ▲
       ▼                    │
Spatial Noise Estimator     │
       │                    │
Noise-strength map M_hat    │
       │                    │
       ▼                    │
 Degradation Encoder ───────┴─► Latent z [B, 32]
                                    │
                                    ▼
                         FiLM Feature Modulation
                                    │
                                    ▼
                             Clean image X_hat
```

#### A. Scout CNN (`SpatialNoiseEstimator` — 14,625 parameters)
- **Input:** $Y \in [0, 1]^{B \times 3 \times H \times W}$
- **Architecture:**
  - Conv2d(3, 16, kernel_size=3, padding=1) + ReLU
  - Conv2d(16, 32, kernel_size=3, padding=1) + ReLU
  - Conv2d(32, 32, kernel_size=3, padding=1) + ReLU
  - Conv2d(32, 1, kernel_size=3, padding=1) + ReLU
- **Output:** $\hat{M} \in [0, \infty)^{B \times 1 \times H \times W}$ (Non-negative noise-strength map)
- **Receptive Field:** $7 \times 7$ pixels.

#### B. Noise Encoder (`NoiseEncoder` — 5,856 parameters)
- **Input:** $\hat{M} \in [0, \infty)^{B \times 1 \times H \times W}$
- **Architecture:**
  - Conv2d(1, 16, kernel_size=3, stride=2, padding=1) + LeakyReLU(0.2)
  - Conv2d(16, 32, kernel_size=3, stride=2, padding=1) + LeakyReLU(0.2)
  - AdaptiveAvgPool2d((1, 1))
  - Flatten(1) -> Linear(32, 32)
- **Output:** Latent degradation vector $z \in \mathbb{R}^{B \times 32}$.

#### C. FiLM Modulation Blocks (`FeatureModulation` — 25,344 parameters)
Formulation:
$$F' = (1 + \gamma(z)) \odot F + \beta(z)$$
where $[\gamma(z), \beta(z)] = W_{\text{film}} z + b_{\text{film}}$.
Zero-initialization ($W_{\text{film}} = 0, b_{\text{film}} = 0$) guarantees identity modulation at training initialization.
- **Encoder Stage 2 (`mod_enc2`):** Linear(32, 64*2) -> 4,224 parameters
- **Bottleneck (`mod_bottleneck`):** Linear(32, 256*2) -> 16,896 parameters
- **Decoder Stage 2 (`mod_dec2`):** Linear(32, 64*2) -> 4,224 parameters

---

### 2.3 Noise-DA Diffusion Network (`DiffusionUNet` — 2,058,179 parameters)
- **Role:** Domain adaptation noise regularizer (**Training ONLY**; never instantiated during inference).
- **Inputs:**
  - Noisy latent $x \in \mathbb{R}^{B \times 3 \times H \times W}$
  - Concatenated condition $C(\hat{y}^s, \hat{y}^r) \in \mathbb{R}^{B \times 6 \times H \times W}$
  - Timestep $t \in \{1, \dots, T\}$
- **Architecture:**
  - Time embedding: Sinusoidal(128) -> Linear(128, 128) + GELU
  - Conv In: Conv2d(9, 64, kernel_size=3, padding=1)
  - Down 1: Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
  - Down 2: Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
  - Middle Block: Conv2d(256, 256, kernel_size=3, padding=1) + GroupNorm(8, 256) + GELU
  - Up 2: ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
  - Up 1: ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
  - Conv Out: Conv2d(128, 3, kernel_size=3, padding=1)
- **Target:** Predicts injected Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$.

