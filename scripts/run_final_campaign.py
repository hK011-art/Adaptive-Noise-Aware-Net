"""
=============================================================================
AUTHORITATIVE MASTER EXPERIMENT CAMPAIGN RUNNER
Project: Adaptive Noise-Aware U-Net with Heteroscedastic Noise Modeling and
         Diffusion-Based Domain Adaptation
Status: Pre-Flight Verified & Production Ready (5 Models x 3 Seeds = 15 Runs)
=============================================================================
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from typing import Dict, Any, List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.manifest import validate_training_dataset_safety

PRE_REGISTERED_MODELS = [
    {
        "id": "Model_A",
        "name": "Model A (Baseline Fixed U-Net, AWGN, No DA)",
        "config": "configs/model_a.yaml",
        "base_channels": 64,
        "params": 12768131,
        "est_vram_gb": 4.8
    },
    {
        "id": "Model_B",
        "name": "Model B (Fixed U-Net, AWGN, Noise-DA)",
        "config": "configs/model_b.yaml",
        "base_channels": 64,
        "params": 12768131,
        "est_vram_gb": 6.5
    },
    {
        "id": "Model_C",
        "name": "Model C (Fixed U-Net, SNIC Heteroscedastic, No DA)",
        "config": "configs/model_c.yaml",
        "base_channels": 64,
        "params": 12768131,
        "est_vram_gb": 4.8
    },
    {
        "id": "Model_D",
        "name": "Model D (Near-Capacity Fixed U-Net, SNIC, Noise-DA)",
        "config": "configs/model_d.yaml",
        "base_channels": 32,
        "params": 3195331,
        "est_vram_gb": 4.2
    },
    {
        "id": "Model_E",
        "name": "Model E (Adaptive Noise-Aware U-Net, SNIC, Noise-DA)",
        "config": "configs/model_e.yaml",
        "base_channels": 32,
        "params": 3241156,
        "est_vram_gb": 4.5
    }
]

PRE_REGISTERED_SEEDS = [42, 123, 2024]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Master Pre-Registered Experiment Campaign Orchestrator (15 Runs)"
    )
    parser.add_argument(
        "--mode",
        choices=["dry_run", "verify_only", "execute"],
        default="dry_run",
        help="Campaign execution mode (default: dry_run)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["Model_A", "Model_B", "Model_C", "Model_D", "Model_E", "all"],
        help="Filter specific model or run all (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        choices=[42, 123, 2024],
        help="Filter specific seed"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="Target CUDA GPU device index (default: 0)"
    )
    parser.add_argument(
        "--manifest_out",
        type=str,
        default="experiment_campaign_matrix.json",
        help="Output JSON file for campaign matrix"
    )
    return parser.parse_args()

def build_full_matrix() -> List[Dict[str, Any]]:
    runs = []
    run_idx = 1
    for m in PRE_REGISTERED_MODELS:
        for seed in PRE_REGISTERED_SEEDS:
            exp_id = f"{m['id']}_seed{seed}"
            ckpt_path = os.path.join("checkpoints", f"{exp_id}_best.pth")
            log_dir = os.path.join("experiments", exp_id)
            log_file = os.path.join(log_dir, "train.log")
            
            runs.append({
                "run_index": run_idx,
                "model_id": m["id"],
                "model_name": m["name"],
                "config_file": m["config"],
                "seed": seed,
                "params": m["params"],
                "est_vram_gb": m["est_vram_gb"],
                "experiment_id": exp_id,
                "checkpoint_path": ckpt_path,
                "log_dir": log_dir,
                "log_file": log_file,
                "command": f"python scripts/train.py --config {m['config']} --seed {seed} --experiment_id {exp_id}"
            })
            run_idx += 1
    return runs

def verify_campaign_integrity(runs: List[Dict[str, Any]]) -> bool:
    print("[Verification] Checking campaign pre-flight integrity...")
    all_ok = True

    # 1. Config file existence
    for r in runs:
        if not os.path.exists(r["config_file"]):
            print(f"  [ERROR] Config file missing: {r['config_file']}")
            all_ok = False
        else:
            try:
                with open(r["config_file"], "r", encoding="utf-8") as f:
                    _ = yaml.safe_load(f)
            except Exception as e:
                print(f"  [ERROR] Invalid YAML in {r['config_file']}: {e}")
                all_ok = False

    # 2. Unique checkpoint and log paths (anti-collision)
    ckpt_paths = [r["checkpoint_path"] for r in runs]
    if len(ckpt_paths) != len(set(ckpt_paths)):
        print("  [ERROR] Duplicate checkpoint paths detected!")
        all_ok = False

    log_dirs = [r["log_dir"] for r in runs]
    if len(log_dirs) != len(set(log_dirs)):
        print("  [ERROR] Duplicate log directories detected!")
        all_ok = False

    # 3. DND training isolation check
    try:
        validate_training_dataset_safety("data/dnd", split="train")
        print("  [ERROR] DND safety validator failed to block train split!")
        all_ok = False
    except ValueError:
        pass  # Expected behavior: DND training strictly forbidden

    if all_ok:
        print("  [PASS] All configuration files, paths, and dataset isolation rules verified.")
    return all_ok

def main():
    args = parse_args()
    
    print("==============================================================================")
    print("        AUTHORITATIVE 15-RUN EXPERIMENT CAMPAIGN ORCHESTRATOR                 ")
    print("        5 Pre-Registered Models x 3 Pre-Registered Seeds                      ")
    print("==============================================================================")

    matrix = build_full_matrix()

    # Filter if requested
    selected_runs = matrix
    if args.model and args.model != "all":
        selected_runs = [r for r in selected_runs if r["model_id"] == args.model]
    if args.seed:
        selected_runs = [r for r in selected_runs if r["seed"] == args.seed]

    # Save manifest
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump({
            "campaign": "Master Thesis Adaptive Denoising & Noise-DA",
            "version": "1.0-frozen",
            "total_runs": len(matrix),
            "selected_runs": len(selected_runs),
            "runs": matrix
        }, f, indent=4)
    print(f"[Manifest] Wrote authoritative 15-run matrix to '{args.manifest_out}'.\n")

    # Run verification
    integrity_ok = verify_campaign_integrity(matrix)
    if not integrity_ok:
        print("[FAIL] Campaign integrity checks failed. Aborting.")
        sys.exit(1)

    print(f"\nTarget Selected Runs: {len(selected_runs)} / {len(matrix)}")
    print("------------------------------------------------------------------------------")
    for r in selected_runs:
        print(f"[{r['run_index']:02d}/15] {r['experiment_id']:<22} | Params: {r['params']:,} | Est VRAM: ~{r['est_vram_gb']}GB")
        print(f"       Checkpoint: {r['checkpoint_path']}")
        print(f"       Command:    {r['command']}")
    print("------------------------------------------------------------------------------\n")

    if args.mode == "verify_only":
        print("[Status] Verification only requested. All checks passed. Exiting.")
        return

    if args.mode == "dry_run":
        print("[Dry Run Complete] To launch training on Kaggle/Colab, run:")
        print("  python scripts/run_final_campaign.py --mode execute")
        print("or run individual commands sequentially as displayed above.")
        return

    if args.mode == "execute":
        print("[EXECUTE MODE] Starting sequential campaign execution on GPU...")
        for idx, r in enumerate(selected_runs, 1):
            print(f"\n>>> [{idx}/{len(selected_runs)}] Launching {r['experiment_id']}...")
            os.makedirs(r["log_dir"], exist_ok=True)
            cmd = [
                sys.executable, "scripts/train.py",
                "--config", r["config_file"],
                "--seed", str(r["seed"]),
                "--experiment_id", r["experiment_id"]
            ]
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"[CRITICAL ERROR] Run {r['experiment_id']} failed with code {res.returncode}. Aborting remaining runs.")
                sys.exit(res.returncode)
            print(f">>> [{idx}/{len(selected_runs)}] Completed {r['experiment_id']} successfully.")

        print("\n==============================================================================")
        print("        ALL CAMPAIGN RUNS COMPLETED SUCCESSFULLY                              ")
        print("==============================================================================")

if __name__ == "__main__":
    main()

