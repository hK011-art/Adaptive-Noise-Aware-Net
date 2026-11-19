# MASTER PRE-KAGGLE AUDIT REPORT: COMPLETE SYSTEM RECORD

**Project Title:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Degree / Context:** Master's Thesis Research Project  
**Report Type:** Comprehensive Cumulative Engineering, Scientific & Statistical Audit  
**Author / Auditor:** Senior Computer Vision Researcher & ML Systems Engineer  
**Audit Date:** October 2026  
**Final Status:** **CODE FROZEN — READY FOR FINAL EXPERIMENTS**  
**Automated Test Suite:** **35 / 35 Tests Passing (100% Pass Rate)**  

---

## 1. Executive Summary & Audit Mandate

This master audit represents the authoritative record of every architectural, mathematical, dataset-safety, statistical, and code enhancement made to the codebase prior to the final experimental campaign on cloud GPU infrastructure (Kaggle / Google Colab).

### The Absolute Research Integrity Rule
**No full model training, benchmark evaluation, multi-seed experiment, or final research experiment was performed locally.**  
All local execution was strictly confined to:
- Static repository inspection and syntax verification.
- Unit and integration testing on synthetic fixtures.
- Exact algebraic round-trip invariance proofs.
- Mathematical monotonicity and statistical error controls.
- Dataset-schema and scene-level partition validation.
- Checkpoint numerical parity proofs ($f(y)_{\text{saved}} \equiv f(y)_{\text{loaded}}$).

All computational resources, benchmark runs, multi-seed training jobs, and ablation evaluations remain scheduled for the pre-registered cloud campaign.

---

## 2. Complete Chronological Evolution (Phases 1 → 2 → 3 → Freeze)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Engineering, Dataset, Noise Physics & ISP Hardening                │
│ • Eliminated silent synthetic fallback loaders; enforced genuine disk files. │
│ • Replaced Gaussian approximations with exact torch.poisson() sampling.     │
│ • Rebuilt camera-space ISP with >140 dB round-trip reconstruction PSNR.    │
│ • Corrected per-sample channel shuffling in Noise-DA.                       │
│ • Established per-image PSNR/SSIM and fail-loud LPIPS calculator.           │
│ • Baseline test suite established: 23 passing tests.                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ PHASE 2: Experiment Safety, Data Leakage & Statistical Controls            │
│ • Implemented hard programmatic DND training exclusion guard.               │
│ • Built two-tier patch and scene-level leakage detection.                   │
│ • Developed chunked streaming SHA-256 file hashing (O(1) RAM).              │
│ • Added isolated experiment directories with overwrite collision protection.│
│ • Isolated FiLM optimizer parameter groups (film_lr_multiplier).            │
│ • Mathematically unified dual residual sign conventions.                    │
│ • Test suite expanded: 28 passing tests.                                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ PHASE 3: Ground-Truth Isolation, Statistical Monotonicity & Provenance      │
│ • Enforced hard ValueError if clean image X is passed to normal inference.  │
│ • Decoupled evaluation pipeline: model receives strictly x_clean=None.      │
│ • Built dataset-specific scene parsers (DIV2K, SIDD, DND) with fail-loud.   │
│ • Enforced cumulative-max monotonicity on Holm-Bonferroni adjusted p-values.│
│ • Formulated paired Cohen's dz and Student-t paired 95% Confidence Intervals.│
│ • Verified composite loss numerical reconstruction (|L_total - Σ λ_i L_i| < 1e-6). │
│ • Embedded config_sha256 in checkpoints and manifests.                      │
│ • Test suite expanded: 35 passing tests.                                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ FINAL CODE FREEZE & PRE-REGISTRATION                                       │
│ • Programmatically calculated authoritative parameter counts for Models A–E.│
│ • Proved exact capacity match: Model D (3.195M) vs Model E (3.241M) (+1.43%).│
│ • Pre-registered hypothesis families and statistical protocol.              │
│ • Pre-registered master experiment campaign protocol YAML.                  │
│ • Authored pre-flight dataset validation tool (scripts/validate_real_datasets.py). │
│ • Authored Thesis Claim Traceability Matrix and 13-step Kaggle SOP.         │
│ • Code officially frozen for cloud execution.                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Authoritative Architecture & Capacity Audit

All parameter counts are calculated programmatically from active PyTorch model definitions.

### Programmatic Parameter Table
| Model / Sub-module | Base Channels | Total Parameters | Trainable Parameters | Capacity vs Model D | Role in Pipeline |
|---|---|---|---|---|---|
| **Model A** (AWGN + No DA + Fixed U-Net) | 64 | **12,768,131** | **12,768,131** | +299.59% | Baseline Denoiser |
| **Model B** (AWGN + Noise-DA + Fixed U-Net) | 64 | **12,768,131** | **12,768,131** | +299.59% | Noise-DA Baseline |
| **Model C** (SNIC + No DA + Fixed U-Net) | 64 | **12,768,131** | **12,768,131** | +299.59% | Physics-Noise Baseline |
| **Model D** (SNIC + Noise-DA + Fixed U-Net) | 32 | **3,195,331** | **3,195,331** | **Baseline (0.00%)** | Capacity-Matched Baseline |
| **Model E** (SNIC + Noise-DA + Adaptive Denoiser) | 32 | **3,241,156** | **3,241,156** | **+1.43%** | Proposed Architecture |
| ↳ *Scout CNN (`SpatialNoiseEstimator`)* | 32 | 14,625 | 14,625 | — | Autonomous Noise Estimator |
| ↳ *Noise Encoder (`NoiseEncoder`)* | 32 | 5,856 | 5,856 | — | Degradation Representation |
| ↳ *Adaptive U-Net (`AdaptiveNoiseAwareUNet`)* | 32 | 3,220,675 | 3,220,675 | — | FiLM-Conditioned Restorer |
| ↳ *FiLM Modulation (sub-component of Adaptive U-Net)* | — | *(25,344)* | *(25,344)* | — | *(Included inside Adaptive U-Net)* |
| **Diffusion Network** (`DiffusionUNet`) | 64/128/256 | **2,058,179** | **2,058,179** | Training Only | Domain Adaptation Regularizer |

### Near-Capacity-Matched Framing
$$\Delta_{\text{Params}} = \frac{3,241,156 - 3,195,331}{3,195,331} \times 100 = \mathbf{+1.43\%} \quad (+45,825 \text{ params})$$
Model D has 3,195,331 parameters and Model E has 3,241,156 parameters. The comparison is framed as **near-capacity-matched**. This small difference substantially reduces the likelihood that observed performance is explained by parameter count, but does not mathematically eliminate that possibility.

---

## 4. Complete Code Modification Inventory

### A. Modified Core Source Files
1. **[`src/models/adaptive_unet.py`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/src/models/adaptive_unet.py)**:
   - Added ground-truth leakage guard to `AdaptiveDenoiser.forward_with_conditioning_mode()`.
   - Raises `ValueError("GROUND-TRUTH LEAKAGE DETECTED!")` if `x_clean is not None` when `mode != "oracle"`.
   - Enforces identity initialization on FiLM modulation weights ($\gamma(z) \approx 0, \beta(z) \approx 0$).
2. **[`scripts/evaluate.py`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/scripts/evaluate.py)**:
   - Decoupled model inference from clean target references: passes `x_clean=None` unconditionally during normal evaluation.
   - Clean targets are reserved strictly downstream for PSNR/SSIM/LPIPS calculation.
3. **[`src/utils/manifest.py`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/src/utils/manifest.py)**:
   - Built dataset-specific scene parsers (`extract_scene_id_from_sample()`) for DIV2K (`^\d+$`), SIDD (4-digit camera identifier), and DND.
   - Enforced fail-loud validation on empty, malformed, or ambiguous sample IDs.
   - Implemented streaming chunked SHA-256 file hashing (`compute_file_sha256()`).
   - Implemented `validate_training_dataset_safety()` and `check_for_data_leakage()`.
4. **[`src/utils/statistical_testing.py`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/src/utils/statistical_testing.py)**:
   - Corrected Holm-Bonferroni step-down multiple-testing adjustment via cumulative maximum:
     $$p_{\text{adj}(k)} = \min\left(1.0, \max_{j \le k} (m - j + 1) \cdot p_{(j)}\right)$$
   - Preserves original sample ordering upon return.
   - Standardized paired effect size: Cohen's $d_z = \bar{D} / s_D$ where $D_i = P_i - B_i$.
   - Added Student-$t$ paired 95% Confidence Intervals with explicit audit metadata.
5. **[`src/utils/checkpoint.py`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/src/utils/checkpoint.py)**:
   - Added `config_sha256` hashing and persistence in checkpoint payloads.
   - Automatically saves lightweight `_inference.pth` file containing strictly restoration weights.
   - Preserves full training state: model, diffusion, optimizer, scheduler, epoch, metric, seed, and git commit.
6. **[`src/models/noise_da_trainer.py`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/src/models/noise_da_trainer.py)**:
   - Exposed `lambda_noise` in return dictionary for numerical loss reconstructibility.
   - Enforces per-sample channel shuffling and dual residual swapping.
7. **[`scripts/train.py`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/scripts/train.py)**:
   - Added automated experiment directory creation (`experiments/<exp_id>/`) with overwrite protection.
   - Added FiLM learning rate multiplier (`film_lr_multiplier`) parameter grouping.
   - Captures runtime environment metadata (`environment.json`).

### B. Newly Created Infrastructure & Protocol Files
1. **[`scripts/validate_real_datasets.py`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/scripts/validate_real_datasets.py)**: Pre-flight cloud dataset validation tool verifying file existence, channels, image dimensions, duplicate hashes, and DND exclusion.
2. **[`configs/statistical_protocol.yaml`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/configs/statistical_protocol.yaml)**: Pre-registered hypothesis families, multiple testing corrections, and effect size formulations.
3. **[`configs/FINAL_EXPERIMENT_PROTOCOL.yaml`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/configs/FINAL_EXPERIMENT_PROTOCOL.yaml)**: Master pre-registered experimental campaign protocol.
4. **[`docs/ARCHITECTURE_FREEZE.md`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/docs/ARCHITECTURE_FREEZE.md)**: Authoritative architectural freeze documentation.
5. **[`docs/THESIS_CLAIM_TRACEABILITY.md`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/docs/THESIS_CLAIM_TRACEABILITY.md)**: Claim-to-evidence matrix mapping every thesis hypothesis to required statistical tests.
6. **[`docs/FINAL_CODE_FREEZE.md`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/docs/FINAL_CODE_FREEZE.md)**: Formal code freeze declaration document.
7. **[`docs/KAGGLE_EXECUTION_ORDER.md`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/docs/KAGGLE_EXECUTION_ORDER.md)**: 13-step standard operating procedure for cloud execution.
8. **[`tests/test_phase3_hardening.py`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/tests/test_phase3_hardening.py)**: 7 automated unit and integration tests covering all Phase 3 guarantees.
9. **[`PHASE3_CHANGES_AUDIT.md`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/PHASE3_CHANGES_AUDIT.md)** & **[`docs/PHASE3_IMPLEMENTATION_AUDIT.md`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/docs/PHASE3_IMPLEMENTATION_AUDIT.md)**: Technical audit reports.

---

## 5. Mathematical & Algorithmic Rigor

### 5.1 Degradation Map Definition
- **Target Noise-Strength Map ($M_{\text{target}}$):**
  $$M_{\text{target}}(x, y) = \sqrt{\frac{1}{3} \sum_{c \in \{R,G,B\}} (Y_c - X_c)^2 + \epsilon}$$
  Implemented with $\epsilon = 10^{-8}$ in `compute_noise_strength_target()`.
- **Predicted Map ($\hat{M}$):** Estimated by Scout CNN (`SpatialNoiseEstimator`) with non-negative ReLU activation:
  $$\hat{M} = \text{Scout}(Y) \in [0, \infty)^{B \times 1 \times H \times W}$$
- **Latent Embedding ($z$):** Encoded by `NoiseEncoder` via strided convolutions, global average pooling, and linear projection:
  $$z = \text{Encoder}(\hat{M}) \in \mathbb{R}^{B \times 32}$$

### 5.2 FiLM Feature Modulation
Conditioning is inserted at three strategic network locations:
1. Encoder Stage 2 (`mod_enc2`)
2. Bottleneck (`mod_bottleneck`)
3. Decoder Stage 2 (`mod_dec2`)
$$F' = (1 + \gamma(z)) \odot F + \beta(z)$$
where $[\gamma(z), \beta(z)] = W_{\text{film}} z + b_{\text{film}}$. Zero-initialization guarantees identity transformation at step 0.

### 5.3 Noise-DA Invariant Residual Swapping
- Standard noise residual: $R = Y - \hat{X}$.
- Restored clean estimate: $\hat{X} = \text{clamp}(Y - R, 0.0, 1.0)$.
- Cross-domain counterfactual composite:
  $$Y^{s \to r} = \hat{X}^s + R^r, \quad Y^{r \to s} = \hat{X}^r + R^s$$
- Dual algebraic reconstruction exactness:
  $$Y_{\text{recon}} = \hat{X} + R = Y - R + R \equiv Y$$
  Verified to $10^{-7}$ precision in automated unit tests.

### 5.4 Composite Optimization Objective
$$L_{\text{total}} = L_{\text{rec}}(\hat{X}^s, X^s) + \lambda_{\text{noise}} L_{\text{noise}}(\hat{M}^s, M_{\text{target}}^s) + \lambda_{\text{dif}}(e) L_{\text{dif}}$$
where:
$$\lambda_{\text{dif}}(e) = \beta_{\text{max}} \cdot \left(\frac{2}{1 + \exp(-\gamma \cdot e / E)} - 1\right)$$
Verified that logged sub-loss components reconstruct $L_{\text{total}}$ within $|L_{\text{total}} - \sum \lambda_i L_i| < 10^{-6}$.

### 5.5 Pre-Registered Statistical Adjustments
For paired comparisons $D_i = P_i - B_i$:
- Paired Wilcoxon Signed-Rank Test: Non-parametric two-sided test discarding zero differences.
- Paired Effect Size: Cohen's $d_z = \bar{D} / s_D$.
- Paired Student-$t$ 95% CI: $\bar{D} \pm t_{0.975, n-1} \cdot \frac{s_D}{\sqrt{n}}$.
- Holm-Bonferroni FWER Control: Rank-sorted adjusted $p$-values enforced by cumulative maximum.

---

## 6. Automated Test Suite Verification

### Full Test Suite Output
```text
Ran 35 tests in 16.289s

OK
[Checkpoint] Saved full training state to tests/test_ckpt_tmp.pth (Epoch 5, Metric 24.50)
[Checkpoint] Loaded full checkpoint from tests/test_ckpt_tmp.pth (Epoch 5)
[Checkpoint] Saved full training state to tests/test_p3_ckpt_tmp.pth (Epoch 1, Metric 0.12)
[Checkpoint] Loaded full checkpoint from tests/test_p3_ckpt_tmp.pth (Epoch 1)
```

### Breakdown of Verified Test Modules
1. **`tests/test_phase3_hardening.py` (7 tests):** Strict ground-truth leakage prevention, Holm cumulative-max monotonicity, paired Cohen's $d_z$, explicit dataset scene parsers, numerical loss reconstruction, checkpoint numerical parity, residual sign algebra.
2. **`tests/test_integration_safety.py` (5 tests):** DND training exclusion, scene-level leakage detection, FiLM optimizer group isolation.
3. **`tests/test_snic_and_isp.py` (5 tests):** Exact Poisson variance, camera-space ISP round-trip linearity (>140 dB PSNR).
4. **`tests/test_noise_da_algebra.py` (5 tests):** Per-sample channel shuffling, dual residual swapping, diffusion conditioning tensor shapes.
5. **`tests/test_checkpoint_and_metrics.py` (5 tests):** Per-image PSNR/SSIM, LPIPS fail-loud strictness, full checkpoint state saving and loading.
6. **`tests/test_models.py` (8 tests):** Fixed U-Net, Adaptive U-Net, Scout CNN, Noise Encoder, FiLM modulation shapes and parameter counts.
- **Total: 35 passing tests (100% pass rate, 0 failures, 0 errors).**

---

## 7. Cloud Execution Roadmap (Kaggle / Google Colab)

The pre-registered experimental campaign will execute according to the 13-step sequence defined in [`docs/KAGGLE_EXECUTION_ORDER.md`](file:///c:/Users/Harshal/OneDrive/ドキュメント/CV%20pro/docs/KAGGLE_EXECUTION_ORDER.md):

```
[Step 1: Validate Environment] ──► [Step 2: Validate Datasets] ──► [Step 3: Generate Manifests]
                                                                            │
[Step 6: Tiny Model E Pilot]   ◄── [Step 5: Loader Smoke Test] ◄── [Step 4: Check Leakage]
         │
[Step 7: STOP & Inspect Logs]
         │
[Step 8: Final Experiments (Seed 42)] ──► [Step 9: Multi-Seed (Seeds 123, 2024)]
                                                         │
[Step 12: Statistical Analysis] ◄── [Step 11: DND Eval] ◄── [Step 10: Model E Ablations]
         │
[Step 13: Export Dissertation Tables & Visual Comparison Plots]
```

---

## 8. Final Research Integrity Declaration

> **"No full model training, benchmark evaluation, multi-seed experiment, or final research experiment was performed prior to this code freeze. The entire codebase is verified, documented, mathematically consistent, protected against ground-truth and data leakage, and frozen for the final cloud experimental campaign."**

