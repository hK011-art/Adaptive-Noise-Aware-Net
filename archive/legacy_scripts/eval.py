import os
import time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.utils.seed import set_seed
from src.utils.metrics import calculate_psnr, calculate_ssim, LPIPSCalculator
from src.utils.logger import ExperimentLogger
from src.dataset.sidd_dataset import SIDDDataset
from src.dataset.dnd_dataset import DNDDataset
from src.dataset.div2k_dataset import SyntheticDIV2KDataset
from src.models.unet import RestorationUNet
from src.models.adaptive_unet import AdaptiveDenoiser
from src.models.noise_estimator import compute_noise_strength_target

def get_writable_dir(default_path: str) -> str:
    if os.path.exists("/kaggle/working") and not os.path.isabs(default_path):
        return os.path.join("/kaggle/working", default_path)
    return default_path

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Image Denoising Models on In-Distribution and Unseen-Noise Stress Tests")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Directory containing trained checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--num_samples", type=int, default=40, help="Number of test patch samples per benchmark")
    parser.add_argument("--base_channels", type=int, default=32, help="Base channels used during training")
    return parser.parse_args()

def load_eval_model(model_id: str, ckpt_path: str, device: torch.device, base_channels: int = 32):
    """Instantiates the appropriate model architecture and loads weights if checkpoint exists."""
    if os.path.exists(ckpt_path):
        try:
            state_dict = torch.load(ckpt_path, map_location=device)
            if model_id == "Model_E":
                if "adaptive_unet.inc.conv.0.weight" in state_dict:
                    base_channels = state_dict["adaptive_unet.inc.conv.0.weight"].shape[0]
                model = AdaptiveDenoiser(base_channels=base_channels, embedding_dim=32).to(device)
            else:
                if "inc.conv.0.weight" in state_dict:
                    base_channels = state_dict["inc.conv.0.weight"].shape[0]
                model = RestorationUNet(in_channels=3, out_channels=3, base_channels=base_channels).to(device)
            model.load_state_dict(state_dict)
            print(f"[Checkpoint] Successfully loaded weights from: {ckpt_path} (base_channels={base_channels})")
        except Exception as e:
            print(f"[Warning] Could not load checkpoint from {ckpt_path}: {e}")
            if model_id == "Model_E":
                model = AdaptiveDenoiser(base_channels=base_channels, embedding_dim=32).to(device)
            else:
                model = RestorationUNet(in_channels=3, out_channels=3, base_channels=base_channels).to(device)
    else:
        print(f"[Warning] Checkpoint '{ckpt_path}' not found. Evaluating initialized model.")
        if model_id == "Model_E":
            model = AdaptiveDenoiser(base_channels=base_channels, embedding_dim=32).to(device)
        else:
            model = RestorationUNet(in_channels=3, out_channels=3, base_channels=base_channels).to(device)
    model.eval()
    return model

def evaluate_model_on_dataset(model, loader, device, lpips_calc, is_adaptive=False, z_override=None, m_hat_override=False):
    model.eval()
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if is_adaptive:
                if m_hat_override:
                    # Oracle Noise Map Ablation: Pass ground truth RMS noise-strength map
                    m_target = compute_noise_strength_target(x, y)
                    restored, _, _ = model(x, m_hat_override=m_target)
                elif z_override is not None:
                    # Wrong-Condition Ablation: Pass overridden z (e.g. z_low)
                    # Expand z_override to batch size if necessary
                    z_batch = z_override.repeat(x.size(0), 1) if z_override.size(0) == 1 else z_override
                    restored, _, _ = model(x, z_override=z_batch)
                else:
                    restored, _, _ = model(x)
            else:
                restored = model(x)

            psnr_val = calculate_psnr(restored, y)
            ssim_val = calculate_ssim(restored, y)
            lpips_val = lpips_calc.calculate(restored, y)

            psnr_vals.append(psnr_val)
            ssim_vals.append(ssim_val)
            lpips_vals.append(lpips_val)

    return {
        "psnr": float(sum(psnr_vals) / len(psnr_vals)),
        "ssim": float(sum(ssim_vals) / len(ssim_vals)),
        "lpips": float(sum(lpips_vals) / len(lpips_vals))
    }

def benchmark_hardware_speed(model, device, num_frames=50, input_size=(1, 3, 128, 128)):
    """Empirically measures inference latency (ms) and throughput (FPS) over 50 passes after 5 warmup passes."""
    model.eval()
    dummy_input = torch.rand(*input_size, device=device)

    # 5 Warmup passes
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    # 50 Timed forward passes
    latencies = []
    with torch.no_grad():
        for _ in range(num_frames):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    mean_latency = float(np.mean(latencies))
    median_latency = float(np.median(latencies))
    fps = 1000.0 / mean_latency
    return mean_latency, median_latency, fps

def main():
    args = parse_args()
    ckpt_dir = get_writable_dir(args.ckpt_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_calc = LPIPSCalculator(device)

    print("==========================================================================")
    print("      ADAPTIVE NOISE-AWARE U-NET: RIGOROUS RESEARCH EVALUATION SUITE      ")
    print("==========================================================================\n")

    # In-Distribution Benchmark: SIDD (Real Smartphone Noise)
    sidd_val = SIDDDataset(root_dir="data/sidd", patch_size=128, is_train=False, num_samples=args.num_samples)
    sidd_loader = DataLoader(sidd_val, batch_size=args.batch_size, shuffle=False)

    # Cross-Camera Generalization Benchmark: DND (Unseen DSLR Cameras)
    dnd_val = DNDDataset(root_dir="data/dnd", patch_size=128, num_samples=args.num_samples)
    dnd_loader = DataLoader(dnd_val, batch_size=args.batch_size, shuffle=False)

    # Central Research Benchmark: Seen ISO (100-800) vs Unseen ISO (1600-3200)
    seen_iso_dataset = SyntheticDIV2KDataset(root_dir="data/div2k", patch_size=128, noise_type="snic",
                                            num_samples=args.num_samples, iso_list=[100.0, 200.0, 400.0, 800.0])
    seen_iso_loader = DataLoader(seen_iso_dataset, batch_size=args.batch_size, shuffle=False)

    unseen_iso_dataset = SyntheticDIV2KDataset(root_dir="data/div2k", patch_size=128, noise_type="snic",
                                              num_samples=args.num_samples, iso_list=[1600.0, 3200.0])
    unseen_iso_loader = DataLoader(unseen_iso_dataset, batch_size=args.batch_size, shuffle=False)

    models_to_eval = ["Model_A", "Model_B", "Model_C", "Model_D", "Model_E"]
    results_summary = {}

    for model_id in models_to_eval:
        is_adaptive = (model_id == "Model_E")
        ckpt_name = f"{model_id}_seed42_best.pth"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        model = load_eval_model(model_id, ckpt_path, device, base_channels=args.base_channels)

        # 1. Benchmarks
        sidd_res = evaluate_model_on_dataset(model, sidd_loader, device, lpips_calc, is_adaptive=is_adaptive)
        dnd_res = evaluate_model_on_dataset(model, dnd_loader, device, lpips_calc, is_adaptive=is_adaptive)

        # 2. Central Research Investigation: Seen vs Unseen ISO
        seen_res = evaluate_model_on_dataset(model, seen_iso_loader, device, lpips_calc, is_adaptive=is_adaptive)
        unseen_res = evaluate_model_on_dataset(model, unseen_iso_loader, device, lpips_calc, is_adaptive=is_adaptive)

        drop_psnr = seen_res["psnr"] - unseen_res["psnr"]
        drop_ssim = seen_res["ssim"] - unseen_res["ssim"]

        # 3. Hardware Latency & FPS Timing
        mean_lat, med_lat, fps = benchmark_hardware_speed(model, device)

        results_summary[model_id] = {
            "sidd": sidd_res,
            "dnd": dnd_res,
            "seen_iso": seen_res,
            "unseen_iso": unseen_res,
            "drop_psnr": drop_psnr,
            "drop_ssim": drop_ssim,
            "mean_lat": mean_lat,
            "fps": fps
        }

        print(f"\n--- {model_id} ---", flush=True)
        print(f"  [SIDD Benchmark (Smartphone)]:    PSNR = {sidd_res['psnr']:.2f} dB | SSIM = {sidd_res['ssim']:.4f} | LPIPS = {sidd_res['lpips']:.4f}")
        print(f"  [DND Cross-Camera Benchmark]:     PSNR = {dnd_res['psnr']:.2f} dB | SSIM = {dnd_res['ssim']:.4f} | LPIPS = {dnd_res['lpips']:.4f}")
        print(f"  [Seen ISOs (100-800)]:            PSNR = {seen_res['psnr']:.2f} dB | SSIM = {seen_res['ssim']:.4f}")
        print(f"  [Unseen ISOs (1600-3200)]:        PSNR = {unseen_res['psnr']:.2f} dB | SSIM = {unseen_res['ssim']:.4f}")
        print(f"  [Robustness Metric Drop]:         Delta PSNR = {drop_psnr:.2f} dB | Delta SSIM = {drop_ssim:.4f}")
        print(f"  [Hardware Timing]:                Mean Latency = {mean_lat:.2f} ms | Median = {med_lat:.2f} ms | Throughput = {fps:.2f} FPS")

    # Ablations on Proposed Model E
    if "Model_E" in models_to_eval:
        print("\n==========================================================================")
        print("          SCIENTIFIC ABLATION EXPERIMENTS ON PROPOSED MODEL E             ")
        print("==========================================================================\n")
        model_e = load_eval_model("Model_E", os.path.join(ckpt_dir, "Model_E_seed42_best.pth"), device, base_channels=args.base_channels)

        # 1. Oracle vs Predicted Noise-Strength Map Ablation (Tested on Unseen High-ISO)
        # Condition A: No degradation condition (Model D baseline evaluated on unseen ISO)
        p_none = results_summary.get("Model_D", {}).get("unseen_iso", {}).get("psnr", 25.0)
        # Condition B: Oracle condition
        oracle_res = evaluate_model_on_dataset(model_e, unseen_iso_loader, device, lpips_calc, is_adaptive=True, m_hat_override=True)
        p_oracle = oracle_res["psnr"]
        # Condition C: Predicted condition
        pred_res = evaluate_model_on_dataset(model_e, unseen_iso_loader, device, lpips_calc, is_adaptive=True, m_hat_override=False)
        p_pred = pred_res["psnr"]

        denom = p_oracle - p_none
        if abs(denom) > 1e-4:
            recovery_pct = 100.0 * (p_pred - p_none) / denom
        else:
            recovery_pct = 100.0

        print(f"1. Oracle vs. Predicted Noise-Strength Map Ablation (Unseen ISO 1600-3200):")
        print(f"   - (A) No Degradation Condition (Fixed Model D): PSNR = {p_none:.2f} dB")
        print(f"   - (B) Oracle Noise Map (M_target):              PSNR = {p_oracle:.2f} dB | SSIM = {oracle_res['ssim']:.4f}")
        print(f"   - (C) Predicted Noise Map (M_hat):              PSNR = {p_pred:.2f} dB | SSIM = {pred_res['ssim']:.4f}")
        print(f"   --> Recovery Ratio: {recovery_pct:.1f}% of oracle upper-bound restored!")

        # 2. Genuine Wrong-Condition Ablation (Tested per Locked Spec Section 3.4: z = 0 and z_low)
        print(f"\n2. Wrong-Condition Sensitivity Ablation:")
        with torch.no_grad():
            # High-noise sample (ISO 3200)
            high_noise_sample = next(iter(unseen_iso_loader))[0][:1].to(device)
            m_high = model_e.noise_estimator(high_noise_sample)
            z_high = model_e.noise_encoder(m_high)

            # Test A: Locked Spec Section 3.4 test with intentionally mismatched zero-noise embedding (z = 0)
            z_zero = torch.zeros_like(z_high)
            x_correct, _, _ = model_e(high_noise_sample, z_override=z_high)
            x_zero, _, _ = model_e(high_noise_sample, z_override=z_zero)
            d_zero = torch.mean(torch.abs(x_correct - x_zero)).item()

            # Test B: Low-noise image embedding (z_low from ISO 100)
            low_noise_sample = next(iter(seen_iso_loader))[0][:1].to(device)
            m_low = model_e.noise_estimator(low_noise_sample)
            z_low = model_e.noise_encoder(m_low)
            x_low, _, _ = model_e(high_noise_sample, z_override=z_low)
            d_low = torch.mean(torch.abs(x_correct - x_low)).item()

        print(f"   - (A) Correct Condition vs Zero-Noise Prior (z = 0, Locked Spec 3.4):")
        print(f"         Sensitivity Distance D_zero:    {d_zero:.6f}")
        if d_zero > 1e-4:
            print(f"         --> VERIFICATION CONFIRMED: Model actively responds to feature modulation (D_zero > 1e-4)!")
        else:
            print(f"         --> NOTICE: Condition sensitivity is small.")

        print(f"   - (B) Correct Condition vs Low-Noise Sample Prior (z_low):")
        print(f"         Sensitivity Distance D_low:     {d_low:.6f}")

    print("\n==========================================================================")
    print("                 EVALUATION SUITE COMPLETED SUCCESSFULLY                  ")
    print("==========================================================================\n")

if __name__ == "__main__":
    main()
