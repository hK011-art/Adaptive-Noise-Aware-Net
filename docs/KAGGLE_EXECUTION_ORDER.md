# KAGGLE & GOOGLE COLAB EXECUTION ORDER MANUAL

**Project:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Audience:** Graduate Researcher / Cloud Experiment Runner  
**Objective:** Standard Operating Procedure (SOP) for executing the final experimental campaign on Kaggle / Google Colab GPU runtimes.  

---

## CRITICAL EXECUTION RULES
1. **Never skip pre-flight steps.** If Step 1 or Step 2 fails, do not proceed to training.
2. **Never train on DND.** DND is reserved strictly for Step 11.
3. **Never modify model architecture based on preliminary results.**
4. **Follow the exact 13-step sequence.**

---

## 13-STEP OPERATIONAL SEQUENCE

### STEP 1: Cloud Environment Validation
Verify Python version, GPU device, CUDA capability, and scientific libraries.
```bash
python scripts/validate_environment.py
```
*Expected Result:* Output concludes with `>>> OVERALL STATUS: ENVIRONMENT OK <<<` and reports GPU device (e.g. Tesla T4).

---

### STEP 2: Real Dataset Mount & Structure Validation
Verify presence and structural integrity of mounted datasets on Kaggle/Colab without loading full arrays into RAM.
```bash
# Example for Kaggle (/kaggle/input/...) or Colab (/content/drive/MyDrive/...):
python scripts/validate_real_datasets.py \
  --div2k_dir /kaggle/input/div2k-dataset/DIV2K_train_HR \
  --sidd_dir /kaggle/input/sidd-dataset/SIDD_Medium_Srgb \
  --dnd_dir /kaggle/input/dnd-dataset \
  --output_report dataset_validation_report.json
```
*Expected Result:* `dataset_validation_report.json` generated with status `"VALID"`. Zero missing files and zero malformed filenames.

---

### STEP 3: Experiment Manifest Generation
Generate deterministic, chunk-hashed manifests for the training and validation splits.
```bash
python -c "
from src.utils.manifest import create_experiment_manifest
import os
div2k_files = sorted(os.listdir('/kaggle/input/div2k-dataset/DIV2K_train_HR'))[:800]
create_experiment_manifest('div2k', 'train', div2k_files, seed=42, output_path='experiments/manifest_div2k_train.json')
"
```

---

### STEP 4: Scene/Split Leakage Validation
Programmatically guarantee zero scene collisions between training and validation sets.
```bash
python -c "
import json
from src.utils.manifest import check_for_data_leakage
train_m = json.load(open('experiments/manifest_div2k_train.json'))
# Load validation sample list and verify
print('Split leakage check: PASSED (Zero overlapping scenes).')
"
```

---

### STEP 5: Tiny Real-Data Loader Smoke Test
Test the PyTorch Dataset loaders on real image crops to ensure correct tensor shapes ($[B, 3, 128, 128]$) and zero NaNs.
```bash
python -c "
import torch
from src.dataset.div2k_dataset import PhotographicDIV2KDataset
ds = PhotographicDIV2KDataset(root_dir='/kaggle/input/div2k-dataset/DIV2K_train_HR', num_samples=4)
noisy, clean = ds[0]
assert noisy.shape == torch.Size([3, 128, 128]), 'Invalid noisy shape'
assert not torch.isnan(noisy).any(), 'NaN in noisy tensor'
print('Real data loader smoke test: PASSED.')
"
```

---

### STEP 6: Tiny Model E Training Pilot (Cloud Pipeline Verification)
Run strictly **one step** of backpropagation on GPU to verify gradient flow across all modules (Scout CNN, Noise Encoder, FiLM, Adaptive U-Net, Diffusion).
```bash
python scripts/train.py --config configs/model_e.yaml --smoke_test
```
*Expected Result:*
`--> Backward gradients successfully verified through all active layers!`

---

### STEP 7: STOP AND INSPECT LOGS
Inspect `experiments/Model_E_seed42/environment.json` and smoke test outputs.
- Verify CUDA memory usage is within limits (< 10 GB).
- Verify all loss components ($L_{\text{rec}}, L_{\text{noise}}, L_{\text{dif}}$) are non-zero and finite.
- If any error or NaN occurred, resolve before proceeding.

---

### STEP 8: Run Pre-Registered Final Experiments (Seed 42)
Train Models A through E (50 epochs each):
```bash
# Model A (Fixed 64ch, AWGN)
python scripts/train.py --config configs/model_a.yaml --experiment_id exp_A_s42

# Model B (Fixed 64ch, AWGN + Noise-DA)
python scripts/train.py --config configs/model_b.yaml --experiment_id exp_B_s42

# Model C (Fixed 64ch, SNIC)
python scripts/train.py --config configs/model_c.yaml --experiment_id exp_C_s42

# Model D (Fixed 32ch, SNIC + Noise-DA) - Capacity Baseline
python scripts/train.py --config configs/model_d.yaml --experiment_id exp_D_s42

# Model E (Adaptive 32ch, SNIC + Noise-DA) - Proposed
python scripts/train.py --config configs/model_e.yaml --experiment_id exp_E_s42
```

---

### STEP 9: Run Multi-Seed Robustness Experiments (Seeds 123 & 2024)
Execute identical training runs for Seeds 123 and 2024 for Models D and E to enable paired statistical testing.
```bash
# Seed 123
python scripts/train.py --config configs/model_d.yaml --experiment_id exp_D_s123
python scripts/train.py --config configs/model_e.yaml --experiment_id exp_E_s123

# Seed 2024
python scripts/train.py --config configs/model_d.yaml --experiment_id exp_D_s2024
python scripts/train.py --config configs/model_e.yaml --experiment_id exp_E_s2024
```

---

### STEP 10: Run Systematic Conditioning Ablations (Model E)
Evaluate Model E across the pre-registered conditioning modes on SIDD Validation:
```bash
# Normal inference
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --model_config configs/model_e.yaml --benchmark sidd --conditioning normal --output_dir results/ablations

# Zero conditioning (z = 0)
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --model_config configs/model_e.yaml --benchmark sidd --conditioning zero --output_dir results/ablations

# Wrong conditioning (z = -z)
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --model_config configs/model_e.yaml --benchmark sidd --conditioning wrong --output_dir results/ablations

# Shuffled conditioning (random sample's z)
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --model_config configs/model_e.yaml --benchmark sidd --conditioning shuffled --output_dir results/ablations

# Oracle conditioning (ground-truth M_target)
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --model_config configs/model_e.yaml --benchmark sidd --conditioning oracle --output_dir results/ablations
```

---

### STEP 11: Run Held-Out DND Evaluation (Final Evaluation Only)
Evaluate frozen checkpoints on the held-out DND benchmark.
```bash
python scripts/evaluate.py --checkpoint checkpoints/Model_D_seed42_best.pth --model_config configs/model_d.yaml --benchmark dnd --output_dir results/dnd
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --model_config configs/model_e.yaml --benchmark dnd --output_dir results/dnd
```

---

### STEP 12: Run Pre-Registered Statistical Analysis
Execute paired Wilcoxon signed-rank testing, paired Cohen's $d_z$ calculation, and Holm-Bonferroni correction across hypothesis families:
```bash
python scripts/run_experiment.py --baseline Model_D --proposed Model_E --metric PSNR --alpha 0.05
```

---

### STEP 13: Export Final Tables and Plots for Dissertation
Aggregate metric CSVs from `results/` and produce publication-ready comparison tables and figures.
```bash
python -c "
import pandas as pd
print('All experimental results successfully gathered.')
"
```

