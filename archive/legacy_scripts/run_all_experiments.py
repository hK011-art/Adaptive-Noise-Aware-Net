import subprocess
import sys
import os

def run_cmd(cmd):
    print(f"\n==========================================================================")
    print(f" Executing: {' '.join(cmd)}")
    print(f"==========================================================================\n", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    subprocess.run(cmd, check=True, env=env)

def main():
    epochs = 2
    seed = 42

    models = ["Model_A", "Model_B", "Model_C", "Model_D", "Model_E"]

    print("\nStarting Full 4-Model Experimental Training & Evaluation Pipeline...", flush=True)

    # Step 1: Train all 4 models
    for m in models:
        cmd = [
            sys.executable, "-u", "train.py",
            "--model_id", m,
            "--epochs", str(epochs),
            "--seed", str(seed)
        ]
        run_cmd(cmd)

    # Step 2: Run automated benchmark evaluation on SIDD and DND
    print("\n==========================================================================", flush=True)
    print(" Running Automated Benchmark Evaluation (SIDD vs DND)...", flush=True)
    print("==========================================================================\n", flush=True)
    eval_cmd = [sys.executable, "-u", "eval.py", "--num_samples", "50"]
    run_cmd(eval_cmd)

if __name__ == "__main__":
    main()

