# Research Reproducibility Framework
**Project:** Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and Diffusion-Based Domain Adaptation  
**Date:** October 2026

---

## 1. Deterministic Seeding Protocol

All experiments invoke `src/utils/seed.py:set_seed(seed)` before initializing model weights or data loaders:
- Python `random.seed(seed)`
- `PYTHONHASHSEED = str(seed)`
- NumPy `np.random.seed(seed)`
- PyTorch CPU `torch.manual_seed(seed)`
- PyTorch CUDA `torch.cuda.manual_seed_all(seed)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

---

## 2. Environment & Manifest Traceability

Every experiment automatically produces two persistent JSON records:
1. **`environment.json`:**  
   Records operating system, Python version, PyTorch version, CUDA runtime status, compute capability, GPU device name, and active Git commit hash.
2. **`experiment_manifest.json`:**  
   Records dataset name, split, sample IDs, sample count, random seed, and SHA-256 fingerprint of the evaluated sample list.

---

## 3. Comprehensive Checkpoint Schema

Checkpoints saved via `src/utils/checkpoint.py:save_checkpoint()` encapsulate the complete training state:
- `epoch`: Current training epoch index.
- `model_name`: String identifier (e.g. `Model_E`).
- `restoration_state_dict`: Model weights for the denoiser network.
- `diffusion_state_dict`: Noise-DA diffusion network weights (if applicable).
- `optimizer_state_dict`: Adam optimizer momentum and variance buffers.
- `scheduler_state_dict`: Learning rate schedule state.
- `best_metric`: Best recorded validation metric.
- `config`: Complete copy of the YAML configuration dictionary used for training.
- `seed`: Random seed.
- `git_commit`: Active git revision hash at time of saving.
- `dataset_manifest_path`: Link to the active dataset manifest.

A paired lightweight file (`{checkpoint_name}_inference.pth`) is saved alongside for inference, containing only the restoration network weights.

