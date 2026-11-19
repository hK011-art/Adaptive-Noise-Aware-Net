import os
import sys
import argparse
import torch

def inspect_checkpoint(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        print(f"[Error] Checkpoint file does not exist: {ckpt_path}")
        sys.exit(1)

    print(f"============================================================")
    print(f"       CHECKPOINT METADATA INSPECTOR                        ")
    print(f"============================================================")
    print(f"Path: {ckpt_path}")
    file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"Size: {file_size_mb:.2f} MB\n")

    data = torch.load(ckpt_path, map_location="cpu")

    if isinstance(data, dict) and "restoration_state_dict" in data:
        print("[Status] Type: Full Research Training Checkpoint")
        print(f"Model Name:       {data.get('model_name', 'N/A')}")
        print(f"Trained Epochs:   {data.get('epoch', 'N/A')}")
        print(f"Best Metric:      {data.get('best_metric', 'N/A')}")
        print(f"Random Seed:      {data.get('seed', 'N/A')}")
        print(f"Git Commit:       {data.get('git_commit', 'N/A')}")
        print(f"Manifest Path:    {data.get('dataset_manifest_path', 'N/A')}")
        
        has_opt = data.get('optimizer_state_dict') is not None
        has_diff = data.get('diffusion_state_dict') is not None
        has_sched = data.get('scheduler_state_dict') is not None
        print(f"Contains Optimizer:   {has_opt}")
        print(f"Contains Diffusion DA: {has_diff}")
        print(f"Contains Scheduler:    {has_sched}")
        
        cfg = data.get('config', {})
        if cfg:
            print(f"\nEmbedded Configuration: {cfg}")
    else:
        print("[Status] Type: Legacy / Inference-Only Weight State Dict")
        if isinstance(data, dict):
            print(f"Layer Keys Count: {len(data.keys())}")
            print(f"Sample Layer Names: {list(data.keys())[:5]}")

    print(f"============================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect PyTorch Checkpoint State and Metadata")
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint file")
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint_path)

