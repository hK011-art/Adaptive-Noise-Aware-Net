# Simplified Camera-Space Rendering Model Specification
**Project:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Date:** October 2026

---

## 1. Architectural Scope

This module models the color space mapping between display-referred sRGB images and linear camera-referred RGB signals:
$$\text{sRGB} \iff \text{Linear RGB} \iff \text{Camera-Space RGB}$$

---

## 2. Transformation Pipeline Stages

### Stage 1: Piecewise sRGB Linearization
Display sRGB values $C_{\text{srgb}} \in [0, 1]$ are mapped to linear optical radiant energy:
$$C_{\text{linear}} = \begin{cases} 
\frac{C_{\text{srgb}}}{12.92}, & C_{\text{srgb}} \le 0.04045 \\
\left(\frac{C_{\text{srgb}} + 0.055}{1.055}\right)^{2.4}, & C_{\text{srgb}} > 0.04045 
\end{cases}$$

### Stage 2: Camera Spectral Sensitivity Matrix
Linear RGB colors are transformed into camera sensor RGB space via forward matrix $\mathbf{M}$:
$$C_{\text{cam}} = \mathbf{M}_{\text{rgb}\to\text{cam}} \cdot C_{\text{linear}}$$
$$\mathbf{M}_{\text{rgb}\to\text{cam}} = \begin{bmatrix}
1.0234 & -0.2969 & -0.2266 \\
-0.5625 & 1.6328 & -0.0469 \\
-0.0703 & 0.2188 & 0.6406
\end{bmatrix}$$

### Stage 3: Inverse Camera Matrix
Noisy camera-space features are returned to linear RGB via exact algebraic matrix inverse:
$$C_{\text{linear}} = \mathbf{M}_{\text{rgb}\to\text{cam}}^{-1} \cdot C_{\text{cam}}$$

### Stage 4: Non-Linear sRGB Tone Response Curve
Linear features are tone-mapped to display sRGB:
$$C_{\text{srgb}} = \begin{cases}
12.92 \cdot C_{\text{linear}}, & C_{\text{linear}} \le 0.0031308 \\
1.055 \cdot C_{\text{linear}}^{1/2.4} - 0.055, & C_{\text{linear}} > 0.0031308
\end{cases}$$
followed by display boundary clamping: $\text{clamp}(C_{\text{srgb}}, 0.0, 1.0)$.

---

## 3. Explicit Model Delimitation

- **No Bayer Sampling:** Features are processed across all 3 RGB channels simultaneously; no color filter array mosaicing is simulated.
- **No Spatial Demosaicing:** No bilinear, edge-directed, or neural demosaicing is performed.
- **Internal Non-Clamping:** Intermediate camera-space tensors are never clipped, guaranteeing that zero-noise round-trips achieve $>140\text{ dB}$ PSNR.

