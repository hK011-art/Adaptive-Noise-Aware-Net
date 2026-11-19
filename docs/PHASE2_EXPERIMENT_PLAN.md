# Phase 2 Cloud GPU Experiment Plan & Execution Protocol
**Project:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Target Environment:** Kaggle Notebooks / Google Colab (Cloud GPU: NVIDIA T4 / P100 / A100)  
**Execution Phase:** Phase 2 (Full Training & Benchmark Execution)

---

## 1. Cloud Pre-requisites & Dataset Setup

Before launching experiments, download official datasets into the repository `data/` directory:
```bash
# 1. Download DIV2K (Train & Validation HR)
# Place in data/div2k/*.png

# 2. Download SIDD Benchmark / Validation
# Place in data/sidd/noisy/*.png and data/sidd/gt/*.png

# 3. Download DND Benchmark Crops
# Place in data/dnd/*.png
```

Verify environment readiness:
```bash
python scripts/validate_environment.py
```

---

## 2. Planned Experiment Matrix

### Experiment 1: Main Model Comparison (Models A, B, C, D, E)
Train all five configurations under identical hyperparameter budgets (50 epochs, Adam lr=1e-4, batch_size=16):
```bash
python scripts/train.py --config configs/model_a.yaml
python scripts/train.py --config configs/model_b.yaml
python scripts/train.py --config configs/model_c.yaml
python scripts/train.py --config configs/model_d.yaml
python scripts/train.py --config configs/model_e.yaml
```

### Experiment 2: Capacity-Matched Comparison (D32 vs E32)
Evaluate Model D (Fixed 32-channel U-Net) vs Model E (Adaptive 32-channel U-Net) across 3 seeds (42, 43, 44):
- Seen ISO evaluation (DIV2K validation with ISO 100, 200, 400, 800)
- Unseen ISO evaluation (DIV2K validation with ISO 1600, 3200)
- Real smartphone evaluation (SIDD validation)
- Cross-camera held-out evaluation (DND benchmark)

### Experiment 3: Degradation Conditioning Ablation Suite (Model E)
Evaluate the impact of degradation vector $z$ on Model E:
```bash
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --benchmark div2k --conditioning normal
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --benchmark div2k --conditioning zero
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --benchmark div2k --conditioning wrong
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --benchmark div2k --conditioning shuffled
python scripts/evaluate.py --checkpoint checkpoints/Model_E_seed42_best.pth --benchmark div2k --conditioning oracle
```

### Experiment 4: Noise-DA Loss Component Ablation
Train Model E variants with:
1. No DA ($\lambda_{\text{DA}} = 0$)
2. Diffusion MSE only (no contrastive hinge loss)
3. Full Noise-DA with per-sample channel shuffling and residual swap

### Experiment 5: Latent Degradation Diagnostic Analysis
Run the latent diagnostic tool on trained checkpoints:
```bash
python tools/analyze_degradation_embedding.py --checkpoint checkpoints/Model_E_seed42_best.pth
```
Verify that:
- $z$ exhibits significant separation across ISO levels ($1 - \text{CosSim} > 0.1$).
- $z$ remains invariant to varying image content under constant noise levels.

### Experiment 6: Paired Statistical Hypothesis Testing
Run Wilcoxon signed-rank test and Cohen's $d$ effect size analysis comparing per-image outputs:
```python
from src.utils.statistical_testing import perform_paired_comparison, export_statistical_report
# Run on paired Model D and Model E results
```

### Experiment 7: Edge Deployment & OpenVINO Benchmark
Export Model E to ONNX and OpenVINO and benchmark latency:
```bash
python export_onnx.py --model_id Model_E --ckpt_path checkpoints/Model_E_seed42_best.pth
python openvino_inference.py --xml adaptive_denoiser.xml --device CPU
```
Report:
- Parameter count and disk size (MB)
- CPU latency (ms per $128 \times 128$ crop and per $512 \times 512$ image)
- OpenVINO FP32 vs FP16 speedup

