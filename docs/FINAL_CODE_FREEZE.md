# FINAL CODE FREEZE DECLARATION

**Project Title:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Freeze Status:** **CODE IS FROZEN FOR THE FINAL EXPERIMENTAL CAMPAIGN**  
**Freeze Date:** October 2026  
**Repository Version:** 3.0-RC-FROZEN  

---

## 1. Specification & Version Metadata

- **Architecture Specification Version:** `v1.0-frozen` ([docs/ARCHITECTURE_FREEZE.md](docs/ARCHITECTURE_FREEZE.md))
- **Experiment Protocol Version:** `v1.0-frozen` ([configs/FINAL_EXPERIMENT_PROTOCOL.yaml](configs/FINAL_EXPERIMENT_PROTOCOL.yaml))
- **Statistical Protocol Version:** `v1.0-frozen` ([configs/statistical_protocol.yaml](configs/statistical_protocol.yaml))
- **Claim Traceability Version:** `v1.0-frozen` ([docs/THESIS_CLAIM_TRACEABILITY.md](docs/THESIS_CLAIM_TRACEABILITY.md))
- **Target Execution Environments:** Kaggle Notebooks (NVIDIA T4 / P100 GPU) & Google Colab (T4 / A100 GPU)
- **Baseline Test Pass Rate:** **35 / 35 Unit & Integration Tests Passing (100%)**

---

## 2. Environment & Dependency Specifications

### Core Runtime:
- **Python:** >= 3.9 (Tested on 3.10 / 3.11 / 3.13)
- **PyTorch:** >= 2.0.0 (CUDA enabled for training)
- **Torchvision:** >= 0.15.0
- **NumPy:** >= 1.24.0
- **SciPy:** >= 1.10.0 (Required for Wilcoxon & Student-t CI)
- **PyYAML:** >= 6.0
- **Pandas:** >= 2.0.0
- **Pillow:** >= 9.5.0

### Optional Evaluation Packages:
- **LPIPS:** >= 0.1.4 (Perceptual metric)
- **OpenVINO:** >= 2023.0 (Edge deployment verification)
- **ONNX / ONNX Runtime:** >= 1.15.0

---

## 3. Authoritative Pre-Flight & Experiment Commands

### Step 1: Pre-Flight Environment Verification
```bash
python scripts/validate_environment.py
```

### Step 2: Pre-Flight Real Dataset Verification
```bash
python scripts/validate_real_datasets.py \
  --div2k_dir data/div2k \
  --sidd_dir data/sidd \
  --dnd_dir data/dnd \
  --output_report dataset_validation_report.json
```

### Step 3: Local / Cloud Smoke Test (1 Step CPU/GPU Pipeline Verification)
```bash
python scripts/train.py --config configs/model_e.yaml --smoke_test
```

### Step 4: Final Experiment Training Execution
Train Models A through E across pre-registered seeds (`42`, `123`, `2024`):

```bash
# Model A: AWGN + No DA (64ch)
python scripts/train.py --config configs/model_a.yaml --experiment_id Model_A_seed42

# Model B: AWGN + Noise-DA (64ch)
python scripts/train.py --config configs/model_b.yaml --experiment_id Model_B_seed42

# Model C: SNIC + No DA (64ch)
python scripts/train.py --config configs/model_c.yaml --experiment_id Model_C_seed42

# Model D: SNIC + Noise-DA (32ch - Capacity Matched Baseline)
python scripts/train.py --config configs/model_d.yaml --experiment_id Model_D_seed42

# Model E: SNIC + Noise-DA + Adaptive Denoiser (32ch - Proposed)
python scripts/train.py --config configs/model_e.yaml --experiment_id Model_E_seed42
```

### Step 5: Benchmark Evaluation Execution
```bash
# Evaluate on SIDD Validation
python scripts/evaluate.py \
  --checkpoint checkpoints/Model_E_seed42_best.pth \
  --model_config configs/model_e.yaml \
  --benchmark sidd \
  --output_dir results/sidd

# Evaluate on Held-Out DND (Final Evaluation Only)
python scripts/evaluate.py \
  --checkpoint checkpoints/Model_E_seed42_best.pth \
  --model_config configs/model_e.yaml \
  --benchmark dnd \
  --output_dir results/dnd

# Conditioning Ablations on Model E
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --model_config configs/model_e.yaml --benchmark sidd --conditioning zero --output_dir results/ablations
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --model_config configs/model_e.yaml --benchmark sidd --conditioning wrong --output_dir results/ablations
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --model_config configs/model_e.yaml --benchmark sidd --conditioning shuffled --output_dir results/ablations
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --model_config configs/model_e.yaml --benchmark sidd --conditioning oracle --output_dir results/ablations
```

---

## 4. Known Limitations & Research Boundaries

1. **Camera-Space Representation:** Noise synthesis and restoration operate in camera-space sRGB. The framework does not process raw Bayer mosaic patterns.
2. **Heuristic Calibration Parameters:** Poisson-Gaussian noise parameters are inspired by mobile sensor profiles (e.g. iPhone 11) rather than laboratory dark-frame calibration runs.
3. **Training Hardware Requirement:** Training with Noise-DA and batch size 16 requires a dedicated GPU with >= 15 GB VRAM (Tesla T4 / P100 / A100).

---

## 5. Absolute Freeze Commitment

> **"Code is frozen for the final experimental campaign. No architecture, loss function, or hyperparameter modifications may be made based on preliminary benchmark results."**

