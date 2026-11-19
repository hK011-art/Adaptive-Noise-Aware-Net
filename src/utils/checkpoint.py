import os
import subprocess
import random
import json
import hashlib
import numpy as np
import torch
from typing import Dict, Any, Optional

def get_git_commit_hash() -> str:
    """Retrieves current git commit hash if available, else returns 'unknown'."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


def save_checkpoint(
    path: str,
    epoch: int,
    model_name: str,
    restoration_net: torch.nn.Module,
    diffusion_da: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    best_metric: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    dataset_manifest_path: Optional[str] = None,
    experiment_id: Optional[str] = None
) -> str:
    """
    Saves a fully comprehensive, reproducible training checkpoint containing:
    - Model weights (restoration and training-only diffusion)
    - Optimizer and scheduler states
    - Epoch, best metric, and experiment ID
    - Configuration payload and config SHA-256
    - Complete multi-library RNG states (Python, NumPy, PyTorch CPU & CUDA)
    - Git commit hash and dataset manifest reference
    
    Also generates a stripped `_inference.pth` file containing strictly the restoration weights.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    config_sha256 = "none"
    if config:
        try:
            config_sha256 = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
        except Exception:
            config_sha256 = hashlib.sha256(str(config).encode("utf-8")).hexdigest()

    ckpt_data = {
        "epoch": epoch,
        "model_name": model_name,
        "experiment_id": experiment_id or f"{model_name}_seed{seed}",
        "restoration_state_dict": restoration_net.state_dict(),
        "diffusion_state_dict": diffusion_da.state_dict() if diffusion_da is not None else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "config": config or {},
        "config_sha256": config_sha256,
        "seed": seed,
        "git_commit": get_git_commit_hash(),
        "dataset_manifest_path": dataset_manifest_path,
        "rng_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
    }

    torch.save(ckpt_data, path)
    print(f"[Checkpoint] Saved full training state to {path} (Epoch {epoch}, Metric {best_metric:.2f})")

    # Also save a lightweight inference-only checkpoint
    inference_path = path.replace(".pth", "_inference.pth")
    if inference_path == path:
        inference_path = path + "_inference.pth"
    torch.save(restoration_net.state_dict(), inference_path)
    return path


def load_checkpoint(
    path: str,
    restoration_net: torch.nn.Module,
    diffusion_da: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    restore_rng: bool = False,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Loads checkpoint. Distinguishes between full training state dicts and inference-only model weights.
    Optionally restores random number generator states when resuming training.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    try:
        data = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        data = torch.load(path, map_location=device)

    # Check if this is a full checkpoint dictionary
    if isinstance(data, dict) and "restoration_state_dict" in data:
        restoration_net.load_state_dict(data["restoration_state_dict"])
        if diffusion_da is not None and data.get("diffusion_state_dict") is not None:
            diffusion_da.load_state_dict(data["diffusion_state_dict"])
        if optimizer is not None and data.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(data["optimizer_state_dict"])
        if scheduler is not None and data.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(data["scheduler_state_dict"])

        if restore_rng and "rng_states" in data and data["rng_states"] is not None:
            try:
                rng = data["rng_states"]
                if "python" in rng and rng["python"]:
                    random.setstate(rng["python"])
                if "numpy" in rng and rng["numpy"]:
                    np.random.set_state(rng["numpy"])
                if "torch_cpu" in rng and rng["torch_cpu"] is not None:
                    torch.set_rng_state(rng["torch_cpu"])
                if "torch_cuda" in rng and rng["torch_cuda"] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["torch_cuda"])
                print("[Checkpoint] Restored Python, NumPy, and PyTorch RNG states.")
            except Exception as e:
                print(f"[Checkpoint Warning] Could not restore all RNG states: {e}")

        print(f"[Checkpoint] Loaded full checkpoint from {path} (Epoch {data.get('epoch')})")
        return data
    else:
        # Legacy or inference-only state dict
        restoration_net.load_state_dict(data)
        print(f"[Checkpoint] Loaded inference-only restoration weights from {path}")
        return {"epoch": 0, "restoration_state_dict": data}
