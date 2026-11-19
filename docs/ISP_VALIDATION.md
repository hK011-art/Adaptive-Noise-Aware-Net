# Simplified Camera ISP Round-Trip Numerical Validation
**Project:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Validation Date:** October 2026  
**Test Suite:** `tests/test_isp_roundtrip.py`

---

## 1. Scope & Pipeline Definition

The camera rendering module simulates an idealized transformation pipeline between sRGB display space and simplified linear camera RGB sensor space:
1. **sRGB Linearization:** IEC 61966-2-1 piecewise sRGB transfer function.
2. **Camera-Space Projection:** Forward matrix multiplication with calibrated sensor spectral sensitivity matrix $\mathbf{M}_{\text{rgb}\to\text{cam}}$.
3. **Inverse Camera Projection:** Algebraic inverse transformation $\mathbf{M}_{\text{cam}\to\text{rgb}} = \mathbf{M}_{\text{rgb}\to\text{cam}}^{-1}$.
4. **Display Reconstruction:** Piecewise non-linear sRGB gamma tone mapping.

> **Physical Scope Disclosure:**  
> This pipeline models camera-space RGB coordinates. It does **not** simulate Bayer color filter array (CFA) subsampling, raw analog-to-digital converter (ADC) integer quantization, or spatial demosaicing interpolation.

---

## 2. Zero-Noise Invertibility Benchmark

Under zero added noise, passing a continuous tensor $X \in [0, 1]^{4 \times 3 \times 64 \times 64}$ through the complete round trip:
$$X \xrightarrow{\text{srgb\_to\_camera\_space}} Y_{\text{cam}} \xrightarrow{\text{camera\_space\_to\_srgb}} \hat{X}$$
yields the following empirical numerical error metrics:

| Metric | Measured Value | Scientific Tolerance | Status |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | $2.98 \times 10^{-8}$ | $< 1.0 \times 10^{-4}$ | **PASS** |
| **Root Mean Squared Error (RMSE)** | $7.35 \times 10^{-8}$ | $< 1.0 \times 10^{-4}$ | **PASS** |
| **Maximum Absolute Error** | $1.32 \times 10^{-6}$ | $< 1.0 \times 10^{-3}$ | **PASS** |
| **Reconstruction PSNR** | **142.67 dB** | $> 70.0\text{ dB}$ | **PASS** |

---

## 3. Boundary Condition Fidelity

- **Zero Black Level ($X = 0.0$):** Exactly preserved ($0.0 \to 0.0$), absolute difference $= 0.0$.
- **Full White Level ($X = 1.0$):** Exactly preserved ($1.0 \to 1.0$), absolute difference $\le 1.0 \times 10^{-7}$.
- **Intermediate Clipping:** Clamping is restricted solely to the final physical display boundary $[0, 1]$, eliminating artificial internal color truncation artifacts.

