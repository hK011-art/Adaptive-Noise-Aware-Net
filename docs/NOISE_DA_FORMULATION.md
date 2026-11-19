# Mathematical Formulation of Noise-DA & Residual Swapping
**Project:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Date:** October 2026

---

## 1. Domain Adaptation Objective

In cross-domain image restoration, the training setup consists of:
- **Source Domain (Synthetic, Supervised):**  
  Pairs $(Y^s, X^s)$ where $X^s$ is the clean image and $Y^s = X^s + N_{\text{syn}}$ is corrupted by the synthetic heteroscedastic camera noise simulator.
- **Target Domain (Real, Unlabeled):**  
  Unpaired real camera captures $Y^r$ drawn from real-world sensors (e.g. smartphone sensors) without ground truth clean reference.

The denoiser $f_\theta$ maps a noisy image to its restored clean estimate:
$$\hat{X}^s = f_\theta(Y^s), \quad \hat{X}^r = f_\theta(Y^r)$$

---

## 2. Residual Definition & Algebraic Sign Conventions

### Standard Definition
The restoration network in our framework is residual:
$$f_\theta(Y) = \text{clamp}(Y - \hat{R}_{\text{noise}}, 0, 1)$$
where $\hat{R}_{\text{noise}} \approx Y - X$ represents the predicted degradation component.

### Domain-Adaptation Shift Residual
In the Noise-DA formulation (ICLR 2025), the residual shift vector from noisy to clean is defined as:
$$R^s = \hat{X}^s - Y^s \quad (\text{Synthetic domain restoration delta})$$
$$R^r = \hat{X}^r - Y^r \quad (\text{Real domain restoration delta})$$

Notice that:
$$Y^s + R^s = Y^s + (\hat{X}^s - Y^s) = \hat{X}^s$$
$$Y^r + R^r = Y^r + (\hat{X}^r - Y^r) = \hat{X}^r$$

### Residual-Swapping Mechanism
To prevent the denoiser from learning trivial domain shortcuts or overfitting to synthetic noise artifacts, Noise-DA synthesizes counterfactual "swapped" conditions:
1. **Synthetic image shifted by real restoration delta:**
   $$\hat{X}^{s \leftarrow r} = \text{clamp}(Y^s + R^r, 0, 1) = \text{clamp}(Y^s + (\hat{X}^r - Y^r), 0, 1)$$
2. **Real image shifted by synthetic restoration delta:**
   $$\hat{X}^{r \leftarrow s} = \text{clamp}(Y^r + R^s, 0, 1) = \text{clamp}(Y^r + (\hat{X}^s - Y^s), 0, 1)$$

---

## 3. Diffusion Conditioning & Contrastive Loss

### Positive Condition ($C_{\text{pos}}$)
The true restored outputs are combined under stochastic per-sample channel shuffling $f_{cs}$:
$$C_{\text{pos}} = f_{cs}(\hat{X}^s, \hat{X}^r)$$

### Negative Swapped Condition ($C_{\text{neg}}$)
The cross-domain corrupted outputs are combined under $f_{cs}$:
$$C_{\text{neg}} = f_{cs}(\hat{X}^{s \leftarrow r}, \hat{X}^{r \leftarrow s})$$

### Contrastive Hinge Loss
Given diffusion model $\epsilon_\theta(\tilde{Y}_t^s, C, t)$ predicting noise $\epsilon \sim \mathcal{N}(0, I)$ added to clean synthetic image $X^s$ at timestep $t$:
$$L_{\text{MSE}} = \|\epsilon - \epsilon_\theta(\tilde{Y}_t^s, C_{\text{pos}}, t)\|^2$$
$$L_{\text{Con}} = \max\left( \|\epsilon - \epsilon_\theta(\tilde{Y}_t^s, C_{\text{pos}}, t)\|^2 - \|\epsilon - \epsilon_\theta(\tilde{Y}_t^s, C_{\text{neg}}, t)\|^2 + \text{margin}, 0 \right)$$
$$L_{\text{Dif}} = 0.5 \cdot L_{\text{MSE}} + 0.5 \cdot L_{\text{Con}}$$

---

## 4. Total Compound Objective

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}}(\hat{X}^s, X^s) + \lambda_{\text{noise}}\mathcal{L}_{\text{noise}}(\hat{M}^s, M_{\text{target}}^s) + \lambda_{\text{DA}}(p)\mathcal{L}_{\text{Dif}}$$

with sigmoid warmup:
$$\lambda_{\text{DA}}(p) = \beta_{\text{max}} \cdot \left( \frac{2}{1 + \exp(-\gamma p)} - 1 \right), \quad p = \frac{\text{epoch}}{\text{total\_epochs}}$$

