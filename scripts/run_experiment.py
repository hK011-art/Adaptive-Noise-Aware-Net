import os
import sys
import json
import argparse
import subprocess
from typing import Dict, Any, List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PRE_REGISTERED_MODELS = [
    ("Model_A", "configs/model_a.yaml"),
    ("Model_B", "configs/model_b.yaml"),
    ("Model_C", "configs/model_c.yaml"),
    ("Model_D", "configs/model_d.yaml"),
    ("Model_E", "configs/model_e.yaml"),
]

PRE_REGISTERED_SEEDS = [42, 123, 2024]

def parse_args():
    parser = argparse.ArgumentParser(description="Authoritative Experiment Campaign Orchestrator (5 Models x 3 Seeds)")
    parser.add_argument("--model", type=str, default=None, choices=["Model_A", "Model_B", "Model_C", "Model_D", "Model_E", "all"], help="Model to train or 'all' for complete grid")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed or None for pre-registered seed list")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1 step smoke test on tiny synthetic fixture")
    parser.add_argument("--dry_run", action="store_true", help="Print and export campaign plan without executing commands")
    parser.add_argument("--generate_matrix", action="store_true", help="Generate experiment_campaign_matrix.json and exit")
    parser.add_argument("--output_matrix", type=str, default="experiment_campaign_matrix.json", help="Path to export experiment matrix JSON")
    return parser.parse_args()

def build_experiment_matrix() -> List[Dict[str, Any]]:
    matrix = []
    for model_name, cfg_path in PRE_REGISTERED_MODELS:
        for seed in PRE_REGISTERED_SEEDS:
            exp_id = f"{model_name}_seed{seed}"
            matrix.append({
                "model_name": model_name,
                "config_file": cfg_path,
                "seed": seed,
                "experiment_id": exp_id,
                "experiment_dir": os.path.join("experiments", exp_id),
                "checkpoint_path": os.path.join("checkpoints", f"{model_name}_seed{seed}_best.pth"),
                "status": "PENDING"
            })
    return matrix

def main():
    args = parse_args()

    print("============================================================")
    print("      RESEARCH EXPERIMENT CAMPAIGN ORCHESTRATOR             ")
    print("      5 Pre-Registered Models x 3 Pre-Registered Seeds      ")
    print("============================================================")

    matrix = build_experiment_matrix()

    # Always generate/export matrix manifest
    with open(args.output_matrix, "w", encoding="utf-8") as f:
        json.dump({
            "total_runs": len(matrix),
            "models": [m[0] for m in PRE_REGISTERED_MODELS],
            "seeds": PRE_REGISTERED_SEEDS,
            "runs": matrix
        }, f, indent=4)
    print(f"[Matrix] Exported authoritative 15-run campaign manifest to '{args.output_matrix}'\n")

    if args.generate_matrix:
        print("[Done] Experiment matrix generated. Exiting.")
        return

    # Filter matrix if specific model/seed requested
    active_runs = matrix
    if args.model and args.model != "all":
        active_runs = [r for r in active_runs if r["model_name"] == args.model]
    if args.seed:
        active_runs = [r for r in active_runs if r["seed"] == args.seed]

    print(f"Target Runs Selected: {len(active_runs)} / {len(matrix)}")
    for r in active_runs:
        print(f"  - [{r['model_name']}] Seed {r['seed']:<4} -> ID: {r['experiment_id']:<20} (Config: {r['config_file']})")
    print("============================================================\n")

    if args.dry_run:
        print("[Dry Run Mode] Commands that would be executed:")
        for r in active_runs:
            cmd = f"python scripts/train.py --config {r['config_file']} --seed {r['seed']} --experiment_id {r['experiment_id']}"
            if args.smoke_test:
                cmd += " --smoke_test"
            print(f"  $ {cmd}")
        print("\n[Done] Dry run complete. No processes executed.")
        return

    # Pre-flight environment check
    print("[Pre-flight] Validating environment...")
    res = subprocess.run([sys.executable, "scripts/validate_environment.py"])
    if res.returncode != 0:
        print("[Error] Environment validation failed. Aborting campaign.")
        sys.exit(1)

    # Execute selected runs
    for idx, r in enumerate(active_runs):
        print(f"\n[{idx+1}/{len(active_runs)}] Launching: {r['experiment_id']}...")
        train_cmd = [
            sys.executable, "scripts/train.py",
            "--config", r["config_file"],
            "--seed", str(r["seed"]),
            "--experiment_id", r["experiment_id"]
        ]
        if args.smoke_test:
            train_cmd.append("--smoke_test")

        res = subprocess.run(train_cmd)
        if res.returncode != 0:
            print(f"[Error] Run {r['experiment_id']} failed with code {res.returncode}. Aborting remaining runs.")
            sys.exit(1)

    print("\n============================================================")
    print("      CAMPAIGN EXECUTION BATCH COMPLETE                     ")
    print("============================================================\n")

if __name__ == "__main__":
    main()
