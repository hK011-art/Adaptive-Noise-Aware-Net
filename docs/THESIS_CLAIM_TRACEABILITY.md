# THESIS CLAIM TRACEABILITY MATRIX

**Project Title:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Status:** Pre-Experiment Verification Protocol  
**Date:** October 2026  

---

## 1. Overview & Research Integrity Protocol

This document maps every intended scientific claim of the Master's thesis directly to its required experimental comparison, evaluation metric, dataset, and statistical hypothesis test. 

**Strict Protocol Rule:**  
No thesis claim may be stated as confirmed in the dissertation without the corresponding experimental evidence meeting the pre-registered statistical criteria defined below.

---

## 2. Claim-to-Evidence Traceability Matrix

#### Claim 1: "Degradation-Aware FiLM Conditioning Improves Denoising Performance"
- **Research Question:** Does dynamic feature modulation conditioned on the Scout CNN's noise estimate outperform a near-capacity-matched fixed U-Net?
- **Required Comparison:** **Model E vs Model D** (Near-capacity-matched at 32 base channels, +1.43% parameter delta).
  - *Capacity Framing Note:* The delta is +45,825 parameters (+1.43%). This near-capacity match substantially reduces the likelihood that observed performance is explained by parameter count, but does not mathematically eliminate that possibility.
- **Datasets:**
  - Real: Genuine SIDD Smartphone Validation Subset.
  - Held-out Real: DND Benchmark.
  - Synthetic: DIV2K Validation (Multi-ISO).
- **Evaluation Metrics:** Per-image PSNR (dB), SSIM, LPIPS.
- **Statistical Test:** Two-sided paired Wilcoxon signed-rank test across identical sample manifests.
- **Effect Size:** Paired Cohen's $d_z = \bar{D} / s_D$.
- **Required Evidence:**
  - Statistically significant improvement ($p < 0.05$ after Holm-Bonferroni correction within Family 1).
  - Positive win rate ($> 60\%$ of test samples showing $PSNR(E) > PSNR(D)$).
  - Positive lower bound on paired Student-$t$ 95% Confidence Interval for mean PSNR difference.

---

### Claim 2: "Noise-DA Domain Adaptation Enhances Generalization to Real Sensor Noise"
- **Research Question:** Does diffusion-based domain adaptation in noise space close the synthetic-to-real domain gap without paired real training data?
- **Required Comparisons:**
  - **Model B vs Model A** (Clean DA Isolation: Both Fixed 64ch U-Net under AWGN; strictly isolates Noise-DA effect).
  - **Model D vs Model C** (Exploratory / Combined Comparison: Evaluates transition from 64ch SNIC baseline to 32ch SNIC + Noise-DA; confounds DA with capacity reduction, so must NOT be claimed as isolating Noise-DA alone).
- **Datasets:** Genuine SIDD Smartphone Benchmark & Held-out DND.
- **Evaluation Metrics:** PSNR, SSIM, LPIPS.
- **Statistical Test:** Paired Wilcoxon signed-rank test.
- **Required Evidence:**
  - Statistically significant PSNR/SSIM gain on real camera test images for models trained with Noise-DA compared to their domain-adaptation-free counterparts ($p < 0.05$ in Family 2).

---

### Claim 3: "Physics-Inspired Heteroscedastic Noise Modeling Outperforms Pure AWGN"
- **Research Question:** Does synthetic training with heteroscedastic Poisson-Gaussian sensor noise provide a more realistic prior than stationary additive white Gaussian noise?
- **Required Comparison:** **Model C vs Model A** (Both 64 base channels, No DA).
- **Datasets:** Real SIDD Smartphone Validation Crops.
- **Evaluation Metrics:** PSNR, SSIM.
- **Statistical Test:** Paired Wilcoxon signed-rank test.
- **Required Evidence:**
  - Model C demonstrates superior restoration on real photographic noise compared to Model A ($p < 0.05$).

---

### Claim 4: "The Adaptive Restorer Actively Responds to Degradation Conditioning (Non-Trivial FiLM Utilization)"
- **Research Question:** Are the FiLM layers genuinely utilizing the degradation embedding $z$, or has the model learned to ignore the conditioning input?
- **Required Ablations (Within Model E):**
  - **E (Normal)** vs **E (Zero Conditioning)** ($z = 0$)
  - **E (Normal)** vs **E (Wrong Conditioning)** ($z_{\text{wrong}} = -z$)
  - **E (Normal)** vs **E (Shuffled Conditioning)** ($z_{\text{shuffled}} \sim \text{Permute}(z)$)
  - **E (Oracle)** vs **E (Normal)** ($z_{\text{oracle}}$ computed from ground-truth $M_{\text{target}}$)
- **Datasets:** DIV2K Validation (Synthetic Multi-ISO) and SIDD Validation.
- **Evaluation Metrics:** PSNR delta ($\Delta\text{PSNR}$), SSIM delta.
- **Statistical Test:** Paired Wilcoxon signed-rank tests across conditioning modes (Family 3).
- **Required Evidence:**
  - Significant performance degradation under Zero, Wrong, and Shuffled conditions ($\text{PSNR}_{\text{Normal}} > \text{PSNR}_{\text{Wrong}}$, $p < 0.05$).
  - Oracle conditioning provides a ground-truth-derived diagnostic reference for assessing the value of the predicted degradation representation (note: Oracle is not mathematically guaranteed to be strictly $\ge$ normal, but serves as a diagnostic benchmark).

---

### Claim 5: "The Architecture is Deployable on Edge CPU Runtimes via OpenVINO"
- **Engineering Question:** Can the autonomous restoration pipeline ($Y \to \hat{M} \to z \to \hat{X}$) execute with low latency on commodity CPU hardware without requiring diffusion network execution?
- **Required Evaluation:**
  - Export `AdaptiveDenoiser` to ONNX and OpenVINO IR (`FP16` and `FP32`).
  - Benchmark inference latency (ms/patch) and throughput (FPS) across patch size $128 \times 128$ on standard x86 CPU.
- **Required Evidence:**
  - Verified static graph export without unsupported dynamic operations.
  - Exact numerical parity: $\| \hat{X}_{\text{PyTorch}} - \hat{X}_{\text{OpenVINO}} \|_{\infty} < 10^{-4}$.
  - Measured latency profile documented in thesis deployment chapter.

---

## 3. Disallowed Claims & Boundaries

The following claims are **strictly disallowed** unless specific, formal physical calibration experiments are independently executed:
1. ❌ "The model implements full Bayer-RAW sensor physics." (The pipeline operates on camera-space RGB, not raw CFA Bayer patterns).
2. ❌ "The degradation encoder extracts calibrated physical sensor read/shot noise parameters." (The encoder produces a learned latent representation $z \in \mathbb{R}^{32}$, not physical gain coefficients $a, b$).
3. ❌ "The model achieves State-of-the-Art (SOTA) on DND." (Comparative claims must strictly refer to the pre-registered baseline grid A–E).
4. ❌ "Model C vs Model D cleanly isolates the effect of Noise-DA alone." (This comparison confounds domain adaptation with a 64-channel to 32-channel capacity reduction; Model B vs Model A is the clean DA ablation).
5. ❌ "The noise synthesis parameters are physically calibrated to hardware sensors." (Sensor profiles are heuristic mobile-sensor-inspired approximations, not laboratory dark-frame calibrations).

