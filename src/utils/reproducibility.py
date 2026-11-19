import os
import sys
import json
import torch
from .checkpoint import get_git_commit_hash

def dump_environment_info(output_path: str = "environment.json", extra_info: dict = None) -> dict:
    """
    Records exact hardware, runtime, and framework environment specifications
    to ensure full scientific reproducibility.
    """
    env_info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "git_commit": get_git_commit_hash(),
        "os_platform": sys.platform,
        "extra_info": extra_info or {}
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(env_info, f, indent=4)
        print(f"[Reproducibility] Captured runtime environment details in {output_path}")

    return env_info

