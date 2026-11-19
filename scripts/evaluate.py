import os
import sys
# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.metrics import compute_per_image_psnr, compute_per_image_ssim, LPIPSCalculator, aggregate_metric_distribution
from src.models.unet import RestorationUNet
from src.models.adaptive_unet import AdaptiveDenoiser
from src.dataset.div2k_dataset import PhotographicDIV2KDataset
from src.dataset.sidd_dataset import GenuineSIDDDataset
from src.dataset.dnd_dataset import HeldOutDNDDataset
class ToySyntheticDataset(torch.utils.data.Dataset):
    """Quarantined toy fixture strictly for --benchmark fixture mode."""
    def __init__(self, num_samples=16, patch_size=64, seed=42):
        super().__init__()
        self.samples = [
            (torch.rand(3, patch_size, patch_size), torch.rand(3, patch_size, patch_size))
            for _ in range(num_samples)
        ]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Cloud-Ready Denoising Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/evaluation.yaml", help="Path to evaluation config YAML")
    parser.add_argument("--model_config", type=str, default="configs/model_e.yaml", help="Path to model config YAML")
    parser.add_argument("--dataset_config", type=str, default="configs/datasets.yaml", help="Path to dataset config YAML")
    parser.add_argument("--benchmark", type=str, default="div2k", choices=["div2k", "sidd", "dnd", "fixture"], help="Benchmark to evaluate")
    parser.add_argument("--conditioning", type=str, default="normal", choices=["normal", "zero", "wrong", "shuffled", "oracle"], help="FiLM conditioning mode")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save metric reports")
    parser.add_argument("--skip_lpips", action="store_true", help="Skip LPIPS computation")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)

    with open(args.config, "r") as f:
        eval_cfg = yaml.safe_load(f)["evaluation"]

    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)["model"]

    with open(args.dataset_config, "r") as f:
        d_cfg = yaml.safe_load(f)["datasets"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n============================================================")
    print(f"  EVALUATION RUN: {model_cfg['name']} | Benchmark: {args.benchmark.upper()}")
    print(f"  Conditioning Mode: {args.conditioning.upper()} | Device: {device}")
    print(f"============================================================")

    # 1. Instantiate Model Architecture
    is_adaptive = model_cfg.get("is_adaptive", False)
    if is_adaptive:
        model = AdaptiveDenoiser(
            base_channels=model_cfg.get("base_channels", 32),
            embedding_dim=model_cfg.get("embedding_dim", 32)
        ).to(device)
    else:
        model = RestorationUNet(
            in_channels=3,
            out_channels=3,
            base_channels=model_cfg.get("base_channels", 32)
        ).to(device)

    # 2. Load Weights
    if os.path.exists(args.checkpoint):
        try:
            ckpt_data = torch.load(args.checkpoint, map_location=device, weights_only=False)
        except TypeError:
            ckpt_data = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt_data, dict) and "restoration_state_dict" in ckpt_data:
            model.load_state_dict(ckpt_data["restoration_state_dict"])
            print(f"[Eval] Loaded full checkpoint from {args.checkpoint} (Epoch {ckpt_data.get('epoch')})")
        else:
            model.load_state_dict(ckpt_data)
            print(f"[Eval] Loaded model state from {args.checkpoint}")
    else:
        print(f"[Eval Notice] Checkpoint '{args.checkpoint}' not found on disk. Evaluating initialized weights.")

    model.eval()

    # 3. Setup Dataset
    if args.benchmark == "fixture":
        print("[Eval] Evaluating on test fixture dataset (toy verification).")
        dataset = ToySyntheticDataset(num_samples=10, patch_size=64)
    elif args.benchmark == "div2k":
        dataset = PhotographicDIV2KDataset(
            root_dir=d_cfg["div2k"]["root_dir"],
            split="val",
            patch_size=d_cfg["div2k"]["patch_size"],
            num_samples=d_cfg["div2k"]["num_val_samples"]
        )
    elif args.benchmark == "sidd":
        dataset = GenuineSIDDDataset(
            root_dir=d_cfg["sidd"]["root_dir"],
            is_train=False,
            patch_size=d_cfg["sidd"]["patch_size"],
            num_samples=d_cfg["sidd"]["num_samples"]
        )
    elif args.benchmark == "dnd":
        dataset = HeldOutDNDDataset(
            root_dir=d_cfg["dnd"]["root_dir"],
            patch_size=d_cfg["dnd"]["patch_size"],
            num_samples=d_cfg["dnd"]["num_samples"]
        )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Setup LPIPS
    skip_lpips = args.skip_lpips or eval_cfg.get("skip_lpips_if_unavailable", False)
    lpips_calc = LPIPSCalculator(device=device, strict=not skip_lpips)

    # 4. Evaluation Loop (Per-Image Metrics)
    records = []
    all_psnr = []
    all_ssim = []
    all_lpips = []

    with torch.no_grad():
        for idx, (noisy, clean) in enumerate(loader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            if is_adaptive:
                # Ground-truth leakage prevention: clean image provided ONLY for oracle ablation
                x_clean_arg = clean if args.conditioning == "oracle" else None
                restored, _, _ = model.forward_with_conditioning_mode(
                    noisy, x_clean=x_clean_arg, mode=args.conditioning
                )
            else:
                restored = model(noisy)

            # Per-image metric calculation
            psnr_vals = compute_per_image_psnr(restored, clean)
            ssim_vals = compute_per_image_ssim(restored, clean)
            lpips_vals = lpips_calc.calculate_per_image(restored, clean) if lpips_calc.enabled else None

            p_val = psnr_vals[0]
            s_val = ssim_vals[0]
            l_val = lpips_vals[0] if lpips_vals is not None else float('nan')

            all_psnr.append(p_val)
            all_ssim.append(s_val)
            if lpips_vals is not None:
                all_lpips.append(l_val)

            records.append({
                "sample_id": f"{args.benchmark}_{idx}",
                "model": model_cfg["name"],
                "benchmark": args.benchmark,
                "conditioning": args.conditioning,
                "psnr": p_val,
                "ssim": s_val,
                "lpips": l_val if lpips_vals is not None else None
            })

    # 5. Statistical Aggregation
    psnr_stats = aggregate_metric_distribution(all_psnr)
    ssim_stats = aggregate_metric_distribution(all_ssim)
    lpips_stats = aggregate_metric_distribution(all_lpips) if all_lpips else None

    print(f"\n============================================================")
    print(f"               BENCHMARK EVALUATION SUMMARY                 ")
    print(f"============================================================")
    print(f"Samples Evaluated: {len(records)}")
    print(f"PSNR (dB):  Mean = {psnr_stats['mean']:.2f} ± {psnr_stats['std']:.3f} | Median = {psnr_stats['median']:.2f} | 95% CI = ±{psnr_stats['ci_95']:.3f}")
    print(f"SSIM:       Mean = {ssim_stats['mean']:.4f} ± {ssim_stats['std']:.4f} | Median = {ssim_stats['median']:.4f} | 95% CI = ±{ssim_stats['ci_95']:.4f}")
    if lpips_stats:
        print(f"LPIPS:      Mean = {lpips_stats['mean']:.4f} ± {lpips_stats['std']:.4f} | Median = {lpips_stats['median']:.4f}")
    else:
        print("LPIPS:      [SKIPPED / UNAVAILABLE]")
    print(f"============================================================\n")

    # 6. Save Reports
    os.makedirs(args.output_dir, exist_ok=True)
    per_image_df = pd.DataFrame(records)
    per_image_path = os.path.join(args.output_dir, f"results_per_image_{args.benchmark}_{args.conditioning}.csv")
    per_image_df.to_csv(per_image_path, index=False)
    print(f"[Results] Exported per-image metrics to {per_image_path}")

    summary_record = [{
        "model": model_cfg["name"],
        "benchmark": args.benchmark,
        "conditioning": args.conditioning,
        "n_samples": len(records),
        "psnr_mean": psnr_stats["mean"],
        "psnr_std": psnr_stats["std"],
        "psnr_median": psnr_stats["median"],
        "ssim_mean": ssim_stats["mean"],
        "ssim_std": ssim_stats["std"],
        "ssim_median": ssim_stats["median"],
        "lpips_mean": lpips_stats["mean"] if lpips_stats else None
    }]
    summary_df = pd.DataFrame(summary_record)
    summary_path = os.path.join(args.output_dir, f"summary_{args.benchmark}_{args.conditioning}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[Results] Exported summary metrics to {summary_path}")

if __name__ == "__main__":
    main()

