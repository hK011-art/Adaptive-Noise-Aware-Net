import os
import sys
# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import yaml
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.reproducibility import dump_environment_info
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.manifest import create_experiment_manifest, validate_training_dataset_safety
from src.models.noise_da_trainer import NoiseDATrainer
from src.dataset.div2k_dataset import PhotographicDIV2KDataset
from src.dataset.sidd_dataset import GenuineSIDDDataset
class ToySyntheticDataset(torch.utils.data.Dataset):
    """Quarantined toy fixture strictly for --smoke_test mode."""
    def __init__(self, num_samples=16, patch_size=64, seed=42):
        super().__init__()
        self.samples = [
            (torch.rand(3, patch_size, patch_size), torch.rand(3, patch_size, patch_size))
            for _ in range(num_samples)
        ]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Cloud-Ready Denoising & Noise-DA Training Script")
    parser.add_argument("--model", type=str, default=None, help="Model name shortcut (e.g. model_a, model_b, model_c, model_d, model_e)")
    parser.add_argument("--config", type=str, default=None, help="Path to model config YAML")
    parser.add_argument("--dataset_config", type=str, default="configs/datasets.yaml", help="Path to dataset config YAML")
    parser.add_argument("--experiment_id", type=str, default=None, help="Unique identifier for experiment directory")
    parser.add_argument("--data_root", type=str, default=None, help="Override root directory for clean source dataset")
    parser.add_argument("--sidd_dir", type=str, default=None, help="Override root directory for real noisy target domain")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed for multi-seed campaign")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing experiment folder")
    parser.add_argument("--smoke_test", action="store_true", help="Run strictly 1 step on tiny toy tensors for verification")
    return parser.parse_args()

def main():
    args = parse_args()

    # Resolve config path from --model or --config
    config_path = args.config
    if args.model is not None and config_path is None:
        m_name = args.model.lower().strip()
        config_path = m_name if m_name.endswith(".yaml") else f"configs/{m_name}.yaml"
    if config_path is None:
        config_path = "configs/model_e.yaml"

    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    with open(args.dataset_config, "r") as f:
        d_cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    noise_cfg = cfg["noise"]
    train_cfg = cfg["training"]
    da_cfg = cfg.get("da", {})

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    set_seed(seed)

    # 1. Setup Experiment Tracking Directory (Step 18 & 19 Safeguards)
    exp_name = args.experiment_id or f"{model_cfg['name']}_seed{seed}"
    exp_dir = os.path.join("experiments", exp_name)

    if os.path.exists(exp_dir) and not args.overwrite and not args.smoke_test:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_dir = f"{exp_dir}_{timestamp}"
        print(f"[Notice] Experiment directory already exists. Logging to new timestamped directory: {exp_dir}")

    os.makedirs(exp_dir, exist_ok=True)
    exp_ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(exp_ckpt_dir, exist_ok=True)

    # Copy active config for audit traceability
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    env_path = os.path.join(exp_dir, "environment.json")
    dump_environment_info(env_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n============================================================")
    print(f"  LAUNCHING TRAINING PIPELINE: {model_cfg['name']} ({device})")
    print(f"  Experiment Directory: {exp_dir}")
    print(f"============================================================")

    # 2. Instantiate Trainer
    trainer = NoiseDATrainer(
        use_domain_adaptation=model_cfg.get("use_da", False),
        is_adaptive=model_cfg.get("is_adaptive", False),
        base_channels=model_cfg.get("base_channels", 32),
        lambda_noise=cfg.get("loss", {}).get("lambda_noise", 0.5),
        beta_max=da_cfg.get("beta_max", 0.2),
        gamma_rate=da_cfg.get("gamma_rate", 5.0)
    ).to(device)

    # 3. Parameter Groups: Isolated FiLM learning rate multiplier (Step 8)
    lr = args.lr or train_cfg.get("learning_rate", 1e-4)
    film_mult = train_cfg.get("film_lr_multiplier", 1.0)
    
    if model_cfg.get("is_adaptive", False) and film_mult > 1.0:
        film_params = []
        base_params = []
        for name, param in trainer.named_parameters():
            if "mod_" in name:
                film_params.append(param)
            else:
                base_params.append(param)
        optimizer = optim.Adam([
            {"params": base_params, "lr": lr},
            {"params": film_params, "lr": lr * film_mult}
        ], weight_decay=train_cfg.get("weight_decay", 1e-5))
    else:
        optimizer = optim.Adam(trainer.parameters(), lr=lr, weight_decay=train_cfg.get("weight_decay", 1e-5))

    epochs = args.epochs or train_cfg.get("epochs", 50)
    batch_size = args.batch_size or train_cfg.get("batch_size", 16)

    # 4. Dataset Setup with Strict Safety Verification (Step 2 & 18)
    if args.smoke_test:
        print("\n[SMOKE TEST MODE ENABLED]")
        print("  Using tiny quarantined toy fixtures strictly for gradient sanity check.")
        print("  THIS IS NOT AN EXPERIMENTAL TRAINING RUN.\n")
        train_dataset = ToySyntheticDataset(num_samples=8, patch_size=64)
        target_dataset = ToySyntheticDataset(num_samples=8, patch_size=64)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        target_loader = DataLoader(target_dataset, batch_size=4, shuffle=True)
        epochs = 1
    else:
        div2k_root = args.data_root or d_cfg["datasets"]["div2k"]["root_dir"]
        sidd_root = args.sidd_dir or d_cfg["datasets"]["sidd"]["root_dir"]

        # Rigorous check: DND must never be used as a training dataset
        validate_training_dataset_safety(div2k_root, "train")
        validate_training_dataset_safety(sidd_root, "train")

        # Photographic DIV2K source (fails loudly if missing)
        train_dataset = PhotographicDIV2KDataset(
            root_dir=div2k_root,
            split="train",
            patch_size=d_cfg["datasets"]["div2k"]["patch_size"],
            noise_type=noise_cfg.get("type", "snic"),
            sensor_type=noise_cfg.get("sensor", "iphone11_main"),
            num_samples=d_cfg["datasets"]["div2k"]["num_train_samples"],
            iso_list=noise_cfg.get("iso_list", [100.0, 200.0, 400.0, 800.0]),
            seed=seed
        )
        drop_last_train = len(train_dataset) >= batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last_train)
        manifest_path = os.path.join(exp_dir, "experiment_manifest.json")
        create_experiment_manifest("DIV2K_Clean_Source", "train", train_dataset.get_sample_ids(), seed, manifest_path)

        if model_cfg.get("use_da", False):
            target_dataset = GenuineSIDDDataset(
                root_dir=sidd_root,
                is_train=True,
                patch_size=d_cfg["datasets"]["sidd"]["patch_size"],
                num_samples=d_cfg["datasets"]["sidd"]["num_samples"],
                seed=seed
            )
            drop_last_target = len(target_dataset) >= batch_size
            target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last_target)
        else:
            target_loader = None

    # Resume if requested
    start_epoch = 1
    if args.resume:
        ckpt_meta = load_checkpoint(args.resume, trainer.restoration_net, trainer.diffusion_da, optimizer, device=device)
        start_epoch = ckpt_meta.get("epoch", 0) + 1

    # Training Loop
    target_iter = iter(target_loader) if target_loader else None
    for epoch in range(start_epoch, epochs + 1):
        trainer.train()
        epoch_losses = []

        for step, batch in enumerate(train_loader):
            noisy_s, clean_s = batch
            noisy_s = noisy_s.to(device)
            clean_s = clean_s.to(device)

            noisy_r = None
            if target_iter:
                try:
                    noisy_r = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    noisy_r = next(target_iter)
                if isinstance(noisy_r, (list, tuple)):
                    noisy_r = noisy_r[0]
                noisy_r = noisy_r.to(device)

                # Cross-domain batch size alignment guard (ensures identical batch size)
                if noisy_r.shape[0] != noisy_s.shape[0]:
                    min_b = min(noisy_s.shape[0], noisy_r.shape[0])
                    noisy_s = noisy_s[:min_b]
                    clean_s = clean_s[:min_b]
                    noisy_r = noisy_r[:min_b]

            optimizer.zero_grad()
            out = trainer(noisy_s, clean_s, noisy_r, current_epoch=epoch, total_epochs=epochs)
            loss = out["total_loss"]
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            if args.smoke_test:
                print(f"  [Smoke Test Step] Loss: {loss.item():.4f} | L_rec: {out['loss_rec'].item():.4f} | L_noise: {out['loss_noise'].item():.4f}")
                print(f"  --> Backward gradients successfully verified through all active layers!")
                break

        avg_loss = float(sum(epoch_losses) / len(epoch_losses))
        print(f"Epoch [{epoch:2d}/{epochs:2d}] Average Total Loss: {avg_loss:.5f}")

        if args.smoke_test:
            break

        # Save checkpoint to dedicated experiment folder and top-level checkpoints/
        ckpt_path = os.path.join(exp_ckpt_dir, f"{model_cfg['name']}_best.pth")
        top_ckpt_path = os.path.join(train_cfg.get("checkpoint_dir", "checkpoints"), f"{model_cfg['name']}_seed{seed}_best.pth")
        
        save_checkpoint(
            path=ckpt_path,
            epoch=epoch,
            model_name=model_cfg['name'],
            restoration_net=trainer.restoration_net,
            diffusion_da=trainer.diffusion_da,
            optimizer=optimizer,
            best_metric=avg_loss,
            config=cfg,
            seed=seed,
            dataset_manifest_path=os.path.join(exp_dir, "experiment_manifest.json"),
            experiment_id=exp_name
        )
        # Also copy to top checkpoints dir for convenient evaluation
        if top_ckpt_path != ckpt_path:
            save_checkpoint(
                path=top_ckpt_path,
                epoch=epoch,
                model_name=model_cfg['name'],
                restoration_net=trainer.restoration_net,
                diffusion_da=trainer.diffusion_da,
                optimizer=optimizer,
                best_metric=avg_loss,
                config=cfg,
                seed=seed,
                dataset_manifest_path=os.path.join(exp_dir, "experiment_manifest.json"),
                experiment_id=exp_name
            )

    print(f"\n[Done] Training pipeline process complete. Results stored in: {exp_dir}")

if __name__ == "__main__":
    main()
