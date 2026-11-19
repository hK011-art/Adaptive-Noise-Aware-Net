# Dataset Management & Anti-Leakage Protocol
**Project:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Date:** October 2026

---

## 1. Principles of Empirical Integrity

1. **Zero Silent Fallback:** If any required dataset is missing from its expected path, loaders must raise a `FileNotFoundError` explaining the missing directory structure. Generating synthetic toy gradients as a substitute for real benchmark images is strictly prohibited.
2. **Quarantined Test Fixtures:** Toy synthetic data exists exclusively inside `tests/fixtures/toy_data.py` and is restricted to unit test suites and rapid smoke testing.
3. **Held-Out Benchmark Isolation:** The Darmstadt Noise Dataset (DND) is strictly held out. Loading DND with `is_train=True` immediately raises a `ValueError`.

---

## 2. Dataset Specifications

### A. DIV2K (Clean Source Domain)
- **Role:** High-resolution clean photographic images for supervised synthetic noise training and validation.
- **Path:** `data/div2k/`
- **Partitioning:** Scene-level split (80% train scenes, 20% validation scenes) to ensure no patch from a training scene ever appears in validation.

### B. SIDD (Real Unlabeled Target Domain & Benchmark)
- **Role:** Real smartphone camera photographs.
- **Path:** `data/sidd/`
- **Training Usage:** In Noise-DA, unpaired real noisy crops $Y^r$ guide diffusion adaptation.
- **Validation Usage:** Paired noisy/clean validation crops evaluate real smartphone denoising performance.

### C. DND (Cross-Camera Held-Out Benchmark)
- **Role:** High-end DSLR noisy captures for evaluating zero-shot cross-sensor domain generalization.
- **Path:** `data/dnd/`
- **Policy:** Never used for model training, hyperparameter search, or checkpoint selection.

---

## 3. Data Leakage Verification & Manifest Tracking
Before initiating training, an `experiment_manifest.json` file is exported documenting:
- Dataset name, split, sample IDs, and seed.
- SHA-256 hash of sample identifiers.
- Verification via `check_for_data_leakage(train_ids, val_ids)` confirming zero intersection between sets.

