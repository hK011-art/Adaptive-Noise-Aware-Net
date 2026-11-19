# Cloud GPU Execution Guide (Kaggle & Google Colab)
**Project:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Audience:** Research Candidate / ML Engineers running Phase 2 experiments  
**Status:** READY FOR CLOUD EXECUTION (Do NOT run on local laptop)

---

## 1. Cloud Environment Setup

### Option A: Kaggle Notebooks
1. **Create Notebook:** Create a new Kaggle notebook with **GPU T4 x2** or **P100** enabled.
2. **Add Datasets:**
   - Add DIV2K dataset to `/kaggle/input/div2k/`
   - Add SIDD dataset to `/kaggle/input/sidd/`
   - Add DND dataset to `/kaggle/input/dnd/`
3. **Clone / Upload Project:**
   ```bash
   !git clone <YOUR_REPO_URL> project && cd project
   # Or upload project ZIP and unzip to /kaggle/working/
   ```
4. **Install Dependencies (if needed):**
   ```bash
   !pip install -r requirements.txt
   ```

### Option B: Google Colab
1. **Select Runtime:** Navigate to `Runtime` -> `Change runtime type` -> select **T4 GPU** or **A100 GPU**.
2. **Mount Google Drive & Set Working Directory:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/CV_pro
   ```
3. **Verify Environment:**
   ```bash
   !python scripts/validate_environment.py
   ```

---

## 2. Pre-Flight Verification on Cloud

Always run pre-flight environment and unit tests first:
```bash
# Verify Python, PyTorch, and CUDA device detection
python scripts/validate_environment.py

# Run complete 23-test unit test suite (takes ~3s)
python -m unittest discover -s tests
```

---

## 3. Phase 2 Experiment Execution Commands

### Step 1: Capacity-Matched Baselines Training (D32 vs E32)
Train Model D (Fixed 32-channel U-Net with Noise-DA) and Model E (Proposed Adaptive U-Net with Noise-DA):
```bash
# Model D (Capacity-Matched Baseline, 50 Epochs, Seed 42)
python scripts/train.py \
    --config configs/model_d.yaml \
    --data_root /path/to/DIV2K \
    --sidd_dir /path/to/SIDD \
    --epochs 50 \
    --batch_size 16

# Model E (Proposed Adaptive Model, 50 Epochs, Seed 42)
python scripts/train.py \
    --config configs/model_e.yaml \
    --data_root /path/to/DIV2K \
    --sidd_dir /path/to/SIDD \
    --epochs 50 \
    --batch_size 16
```

### Step 2: Multi-Seed Training (Seeds 42, 43, 44)
For statistical rigor, run Model D and Model E across 3 seeds:
```bash
# Model D Seeds
python scripts/train.py --config configs/model_d.yaml --seed 42 --experiment_id Model_D_seed42
python scripts/train.py --config configs/model_d.yaml --seed 43 --experiment_id Model_D_seed43
python scripts/train.py --config configs/model_d.yaml --seed 44 --experiment_id Model_D_seed44

# Model E Seeds
python scripts/train.py --config configs/model_e.yaml --seed 42 --experiment_id Model_E_seed42
python scripts/train.py --config configs/model_e.yaml --seed 43 --experiment_id Model_E_seed43
python scripts/train.py --config configs/model_e.yaml --seed 44 --experiment_id Model_E_seed44
```

### Step 3: Degradation Conditioning Ablation Suite (Model E)
Systematically test the five degradation conditioning modes on the trained Model E checkpoint:
```bash
# 1. Normal autonomous Scout conditioning
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --benchmark div2k --conditioning normal

# 2. Zeroed conditioning (z = 0, deactivates FiLM)
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --benchmark div2k --conditioning zero

# 3. Inverted / Mismatched conditioning (z = -z)
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --benchmark div2k --conditioning wrong

# 4. Batch-shuffled conditioning across samples
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --benchmark div2k --conditioning shuffled

# 5. Oracle degradation map conditioning (M_target)
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --benchmark div2k --conditioning oracle
```

### Step 4: Real Benchmark Generalization Evaluations
```bash
# Real Smartphone Noise Evaluation (SIDD Validation)
python scripts/evaluate.py --checkpoint checkpoints/Model_D_seed42_best.pth --benchmark sidd
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --benchmark sidd

# Zero-Shot Cross-Camera DSLR Evaluation (Held-Out DND Benchmark)
python scripts/evaluate.py --checkpoint checkpoints/Model_D_seed42_best.pth --benchmark dnd
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --benchmark dnd
```

### Step 5: Post-Training Diagnostic Analysis
```bash
# Analyze latent degradation embedding representation (ISO separation vs content)
python tools/analyze_degradation_embedding.py --checkpoint checkpoints/Model_E_seed42_best.pth

# Inspect trained checkpoint metadata
python scripts/inspect_checkpoint.py checkpoints/Model_E_seed42_best.pth
```

---

## 4. Troubleshooting & Safety FAQ

- **Q: Does training support resuming from an interrupted epoch?**  
  **A:** Yes, pass `--resume <PATH_TO_CHECKPOINT>`.
- **Q: Can DND be accidentally used for training?**  
  **A:** No, `validate_training_dataset_safety()` and `HeldOutDNDDataset` will immediately raise a `ValueError` if DND is configured for training.
- **Q: Will existing experiments be accidentally overwritten?**  
  **A:** No, `scripts/train.py` creates dedicated experiment directories in `experiments/` and appends timestamps if an experiment ID already exists (unless `--overwrite` is explicitly passed).

