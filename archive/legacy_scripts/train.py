import os
import argparse
import torch
from torch.utils.data import DataLoader
from src.utils.seed import set_seed
from src.utils.metrics import calculate_psnr, calculate_ssim, LPIPSCalculator
from src.utils.logger import ExperimentLogger
from src.dataset.div2k_dataset import SyntheticDIV2KDataset
from src.dataset.sidd_dataset import SIDDDataset
from src.models.noise_da_trainer import NoiseDATrainer

def get_writable_dir(default_path: str) -> str:
    """Redirects relative paths to /kaggle/working/ if running inside Kaggle."""
    if os.path.exists("/kaggle/working") and not os.path.isabs(default_path):
        return os.path.join("/kaggle/working", default_path)
    return default_path

def parse_args():
    parser = argparse.ArgumentParser(description="Train Image Denoising Models (Models A, B, C, D, E)")
    parser.add_argument("--model_id", type=str, default="Model_E",
                        choices=["Model_A", "Model_B", "Model_C", "Model_D", "Model_E"],
                        help="Model configuration to train")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--iso_mode", type=str, default="seen", choices=["seen", "full"],
                        help="'seen' trains strictly on ISO 100, 200, 400, 800 for unseen-noise stress testing")
    parser.add_argument("--base_channels", type=int, default=32, help="Base U-Net channels (32 for laptop feasibility)")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for model checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for experiment logs")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = get_writable_dir(args.output_dir)
    log_dir = get_writable_dir(args.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Running on device: {device} | Configuration: {args.model_id} | Seed: {args.seed} | ISO Mode: {args.iso_mode}", flush=True)

    # 5-Model Grid Configuration
    if args.model_id == "Model_A":
        noise_type = "awgn"
        use_da = False
        is_adaptive = False
    elif args.model_id == "Model_B":
        noise_type = "awgn"
        use_da = True
        is_adaptive = False
    elif args.model_id == "Model_C":
        noise_type = "snic"
        use_da = False
        is_adaptive = False
    elif args.model_id == "Model_D":
        noise_type = "snic"
        use_da = True
        is_adaptive = False
    elif args.model_id == "Model_E":
        noise_type = "snic"
        use_da = True
        is_adaptive = True

    # Seen ISOs for controlled unseen-noise generalization experiment
    if args.iso_mode == "seen":
        iso_list = [100.0, 200.0, 400.0, 800.0]
    else:
        iso_list = [100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0]

    synthetic_dataset = SyntheticDIV2KDataset(
        root_dir="data/div2k",
        patch_size=128,
        noise_type=noise_type,
        num_samples=300,
        iso_list=iso_list
    )
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=args.batch_size, shuffle=True)

    if use_da:
        real_target_dataset = SIDDDataset(root_dir="data/sidd", patch_size=128, is_train=True, num_samples=300)
        real_target_loader = DataLoader(real_target_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        real_target_loader = None

    val_sidd_dataset = SIDDDataset(root_dir="data/sidd", patch_size=128, is_train=False, num_samples=50)
    val_loader = DataLoader(val_sidd_dataset, batch_size=args.batch_size, shuffle=False)

    trainer = NoiseDATrainer(
        use_domain_adaptation=use_da,
        is_adaptive=is_adaptive,
        base_channels=args.base_channels,
        lambda_noise=0.5,
        beta_max=0.2,
        gamma_rate=5.0
    ).to(device)

    optimizer = torch.optim.Adam(trainer.parameters(), lr=args.lr, betas=(0.9, 0.999))
    lpips_calc = LPIPSCalculator(device)
    logger = ExperimentLogger(log_dir, f"{args.model_id}_seed{args.seed}")

    os.makedirs(output_dir, exist_ok=True)
    best_psnr = -1.0

    for epoch in range(1, args.epochs + 1):
        trainer.train()
        total_epoch_loss = 0.0
        total_rec_loss = 0.0
        total_noise_loss = 0.0

        if use_da and real_target_loader is not None:
            real_iter = iter(real_target_loader)

        for x_s, y_s in synthetic_loader:
            x_s, y_s = x_s.to(device), y_s.to(device)

            if use_da and real_target_loader is not None:
                try:
                    x_r = next(real_iter).to(device)
                except StopIteration:
                    real_iter = iter(real_target_loader)
                    x_r = next(real_iter).to(device)
            else:
                x_r = None

            optimizer.zero_grad()
            outputs = trainer(x_s, y_s, x_r=x_r, current_epoch=epoch, total_epochs=args.epochs)
            loss = outputs["total_loss"]
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            total_rec_loss += outputs["loss_rec"].item()
            total_noise_loss += outputs["loss_noise"].item()

        avg_train_loss = total_epoch_loss / len(synthetic_loader)
        avg_rec_loss = total_rec_loss / len(synthetic_loader)
        avg_noise_loss = total_noise_loss / len(synthetic_loader)

        # Validation on SIDD Benchmark
        trainer.eval()
        val_psnr_list, val_ssim_list, val_lpips_list = [], [], []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                out = trainer.restoration_net(x_val)
                restored = out[0] if isinstance(out, tuple) else out

                val_psnr_list.append(calculate_psnr(restored, y_val))
                val_ssim_list.append(calculate_ssim(restored, y_val))
                val_lpips_list.append(lpips_calc.calculate(restored, y_val))

        mean_psnr = sum(val_psnr_list) / len(val_psnr_list)
        mean_ssim = sum(val_ssim_list) / len(val_ssim_list)
        mean_lpips = sum(val_lpips_list) / len(val_lpips_list)

        logger.log_epoch(epoch, avg_train_loss, mean_psnr, mean_ssim, mean_lpips)
        log_str = f"Epoch [{epoch}/{args.epochs}] | Loss: {avg_train_loss:.4f} (Rec: {avg_rec_loss:.4f}"
        if is_adaptive:
            log_str += f", Noise: {avg_noise_loss:.4f}"
        log_str += f") | Val SIDD -> PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f} | LPIPS: {mean_lpips:.4f}"
        print(log_str, flush=True)

        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            ckpt_path = os.path.join(output_dir, f"{args.model_id}_seed{args.seed}_best.pth")
            torch.save(trainer.restoration_net.state_dict(), ckpt_path)
            print(f"  --> Saved Best Checkpoint to {ckpt_path}", flush=True)

    print(f"\n[Train Complete] {args.model_id} Best PSNR: {best_psnr:.2f} dB\n", flush=True)

if __name__ == "__main__":
    main()
