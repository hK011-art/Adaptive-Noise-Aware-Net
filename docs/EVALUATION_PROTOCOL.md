# Quantitative Evaluation Protocol & Statistical Standards
**Project:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Date:** October 2026

---

## 1. Per-Image Metric Evaluation

To ensure accurate statistical variance estimation and avoid skewing from whole-batch averages, all metrics are computed strictly on individual samples ($B=1$ or per-image across batch):
- **PSNR (dB):**
  $$\text{MSE}_i = \frac{1}{3HW} \sum_{c,h,w} (\hat{X}_{i,c,h,w} - X_{i,c,h,w})^2$$
  $$\text{PSNR}_i = 10 \log_{10} \left( \frac{1.0}{\text{MSE}_i} \right)$$
- **SSIM:** Multi-channel 2D Gaussian kernel ($11 \times 11, \sigma=1.5$) structural similarity computed per sample.
- **LPIPS:** AlexNet perceptual patch similarity. If the `lpips` library is unavailable, the evaluation logs explicitly mark the metric as `UNAVAILABLE` rather than returning a deceptive 0.0 value.

---

## 2. Statistical Aggregations & Reporting

For every benchmark test set:
- Mean and standard deviation: $\mu \pm \sigma$.
- Sample median: $Q_2$.
- 95% Confidence Interval: $\pm 1.96 \cdot \frac{\sigma}{\sqrt{N}}$.
- Raw per-image metric table exported to `results/results_per_image_{benchmark}_{mode}.csv`.
- Summary row exported to `results/summary_{benchmark}_{mode}.csv`.

---

## 3. Paired Statistical Hypothesis Testing

For comparing capacity-matched pairs (Model D vs Model E):
- **Test:** Non-parametric Wilcoxon signed-rank test on paired per-image differences $\Delta_i = \text{PSNR}_{E,i} - \text{PSNR}_{D,i}$.
- **Significance Level:** $\alpha = 0.05$.
- **Effect Size:** Cohen's $d = \frac{\bar{\Delta}}{s_{\Delta}}$.
- **Win Rate:** $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\Delta_i > 0)$.
- Claims of superiority are permitted **only** when $p < 0.05$ and win rate $> 50\%$.

---

## 4. Degradation Conditioning Ablation Suite

The evaluation CLI supports five systematic conditioning modes:
1. `--conditioning normal`: Standard inference with Scout CNN $\hat{M}$ and Noise Encoder $z$.
2. `--conditioning zero`: $z = \mathbf{0}$. Tests network performance when FiLM modulation is completely deactivated.
3. `--conditioning wrong`: $z = -z_{\text{pred}}$ or inverted noise-level embedding. Tests sensitivity to incorrect noise estimates.
4. `--conditioning shuffled`: $z_{\pi(i)}$ permutation across samples in batch. Tests semantic degradation alignment.
5. `--conditioning oracle`: $z$ encoded from ground-truth degradation map $M_{\text{target}}$. Quantifies theoretical ceiling of the conditioning mechanism.

