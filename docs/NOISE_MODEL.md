# Heteroscedastic Sensor-Noise Synthesis Formulation
**Project:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Date:** October 2026

---

## 1. Physical Principles

Digital camera image sensors suffer from two fundamental physical noise components:
1. **Photon Shot Noise:** Arises from the quantum arrival of discrete photons obeying Poisson statistics.
2. **Electronic Read Noise:** Arises from thermal fluctuations, dark current leakage, and analog-to-digital converter readout circuits, modeled as zero-mean Gaussian noise.

---

## 2. Mathematical Formulation

Let $I \in [0, 1]$ denote true optical intensity in linear camera space.

### Step 1: Poisson Shot Noise
The expected photon count is scaled by the sensor conversion gain $a$:
$$\lambda = \frac{I}{a}$$
$$N \sim \text{Poisson}(\lambda)$$
$$I_{\text{shot}} = a \cdot N$$

where $a = k \cdot \text{ISO}^m$ characterizes the ISO-dependent electronic gain factor.

### Step 2: Gaussian Read Noise
Electronic circuitry introduces additive disturbance:
$$\epsilon_{\text{read}} \sim \mathcal{N}(0, b)$$
where $b = b_{\text{base}} \cdot \left(\frac{\text{ISO}}{100}\right)^p$ represents total read noise variance.

### Step 3: Heteroscedastic Variance Relation
Combining shot and read noise yields the linear Poisson-Gaussian variance relationship:
$$\mathbb{E}[Y_{\text{cam}} \mid I] = I$$
$$\operatorname{Var}(Y_{\text{cam}} \mid I) = a \cdot I + b$$
This heteroscedasticity models the physical reality that noisy variance is higher in bright highlights than in shadows.

---

## 3. Sensor Profile Specifications & Provenance

| Sensor Profile Name | Proxy Device | $k$ | $m$ | $b_{\text{base}}$ | Provenance Declaration |
|---|---|---|---|---|---|
| `iphone11_main` | iPhone 11 Main Camera | $1.2 \times 10^{-6}$ | 1.15 | $4.5 \times 10^{-5}$ | Heuristic / estimated proxy |
| `iphone11_tele` | iPhone 11 Telephoto | $1.5 \times 10^{-6}$ | 1.12 | $5.0 \times 10^{-5}$ | Heuristic / estimated proxy |
| `sony_rx100` | Sony RX100 1" Sensor | $8.0 \times 10^{-7}$ | 1.05 | $1.2 \times 10^{-5}$ | Heuristic / estimated proxy |
| `sony_a7r3` | Sony A7R III Full-Frame | $5.0 \times 10^{-7}$ | 1.00 | $8.0 \times 10^{-6}$ | Heuristic / estimated proxy |

> **Transparency Note:** These parameters are designated as empirical parameterizations. They are **not** represented as direct laboratory radiometric calibration measurements.

