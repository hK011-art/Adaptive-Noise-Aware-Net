import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.noise.snic_noise import SNICNoiseGenerator
from src.models.unet import RestorationUNet
from src.models.adaptive_unet import AdaptiveDenoiser
from src.utils.metrics import compute_per_image_psnr, compute_per_image_ssim

def main():
    print("=" * 70)
    print("   GENERATING REALISTIC BENCHMARK COMPARISONS (BUILDING & PEOPLE)   ")
    print("=" * 70)

    device = torch.device("cpu")
    os.makedirs("results/figures", exist_ok=True)

    # 1. Load Trained Models
    print("Loading Model D (Baseline)...")
    net_d = RestorationUNet(3, 3, base_channels=32).to(device)
    ckpt_d = torch.load("checkpoints/Model_D_seed42_best.pth", map_location=device, weights_only=False)
    net_d.load_state_dict(ckpt_d["restoration_state_dict"])
    net_d.eval()

    print("Loading Model E (Proposed Adaptive)...")
    net_e = AdaptiveDenoiser(base_channels=32, embedding_dim=32).to(device)
    ckpt_e = torch.load("checkpoints/Model_E_seed42_best.pth", map_location=device, weights_only=False)
    net_e.load_state_dict(ckpt_e["restoration_state_dict"])
    net_e.eval()

    # 2. Load Realistic Image
    img_path = "test_street.jpg"
    full_pil = Image.open(img_path).convert("RGB")
    print(f"Loaded '{img_path}' ({full_pil.size[0]}x{full_pil.size[1]})")

    # 3. Setup SNIC Smartphone Noise (iPhone 11 sensor at high ISO 3200)
    snic = SNICNoiseGenerator(sensor_type="iphone11_main")
    torch.manual_seed(42)
    np.random.seed(42)

    # Targets to process
    targets = [
        {
            "name": "Building & Architectural Masonry",
            "box": (400, 400, 750, 750),  # x1, y1, x2, y2
            "zoom_box": (140, 160, 260, 280), # inside crop: roses and window
            "filename": "results/figures/realistic_building_comparison.png"
        },
        {
            "name": "Pedestrians & Cobblestone Street",
            "box": (10, 680, 260, 930),    # x1, y1, x2, y2
            "zoom_box": (70, 70, 180, 180),   # inside crop: walking people
            "filename": "results/figures/realistic_people_comparison.png"
        }
    ]

    for item in targets:
        print(f"\nProcessing: {item['name']}...")
        crop_pil = full_pil.crop(item["box"])
        w, h = crop_pil.size
        # Ensure dimensions divisible by 16 for U-Net pooling
        w_pad = (w // 16) * 16
        h_pad = (h // 16) * 16
        crop_pil = crop_pil.crop((0, 0, w_pad, h_pad))

        clean_np = np.array(crop_pil, dtype=np.float32) / 255.0
        clean_t = torch.from_numpy(clean_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # Inject realistic smartphone sensor noise
        noisy_t, _ = snic.add_noise(clean_t, iso=3200.0)

        with torch.no_grad():
            out_d = net_d(noisy_t)
            out_e, _, _ = net_e(noisy_t)

        out_d = torch.clamp(out_d, 0.0, 1.0)
        out_e = torch.clamp(out_e, 0.0, 1.0)

        # Calculate exact metrics
        psnr_noisy = compute_per_image_psnr(noisy_t, clean_t)[0]
        ssim_noisy = compute_per_image_ssim(noisy_t, clean_t)[0]

        psnr_d = compute_per_image_psnr(out_d, clean_t)[0]
        ssim_d = compute_per_image_ssim(out_d, clean_t)[0]

        psnr_e = compute_per_image_psnr(out_e, clean_t)[0]
        ssim_e = compute_per_image_ssim(out_e, clean_t)[0]

        gain = psnr_e - psnr_d
        print(f"  Noisy Input:  {psnr_noisy:.2f} dB | SSIM {ssim_noisy:.4f}")
        print(f"  Model D:      {psnr_d:.2f} dB | SSIM {ssim_d:.4f}")
        print(f"  Model E:      {psnr_e:.2f} dB | SSIM {ssim_e:.4f}  (Gain: +{gain:.2f} dB)")

        # Convert to numpy for plotting
        clean_disp = clean_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        noisy_disp = noisy_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        d_disp = out_d.squeeze(0).permute(1, 2, 0).cpu().numpy()
        e_disp = out_e.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Render 4-column figure with zoomed insets
        fig, axes = plt.subplots(1, 4, figsize=(20, 6), facecolor="#0f172a")

        panels = [
            {"img": clean_disp, "title": "Clean Reference", "sub": "Ground Truth Target", "color": "#10b981"},
            {"img": noisy_disp, "title": "Noisy Smartphone Input", "sub": f"iPhone 11 Sensor | ISO 3200\nPSNR: {psnr_noisy:.2f} dB | SSIM: {ssim_noisy:.4f}", "color": "#ef4444"},
            {"img": d_disp, "title": "Model D (Capacity Baseline)", "sub": f"Static U-Net (3.19M params)\nPSNR: {psnr_d:.2f} dB | SSIM: {ssim_d:.4f}", "color": "#f59e0b"},
            {"img": e_disp, "title": "Model E (Proposed Adaptive)", "sub": f"Adaptive FiLM (+1.43% params)\nPSNR: {psnr_e:.2f} dB (+{gain:.2f} dB) | SSIM: {ssim_e:.4f}", "color": "#38bdf8"}
        ]

        zx1, zy1, zx2, zy2 = item["zoom_box"]

        for ax, p in zip(axes, panels):
            ax.imshow(p["img"])
            ax.set_title(p["title"], fontsize=13, fontweight="bold", color=p["color"], pad=10)
            ax.set_xlabel(p["sub"], fontsize=11, color="#cbd5e1", labelpad=8)
            ax.set_xticks([])
            ax.set_yticks([])

            # Draw zoom box
            rect = patches.Rectangle((zx1, zy1), zx2 - zx1, zy2 - zy1, linewidth=2, edgecolor="#f43f5e", facecolor="none", linestyle="--")
            ax.add_patch(rect)

            for spine in ax.spines.values():
                spine.set_color(p["color"])
                spine.set_linewidth(2.5)

        plt.suptitle(f"Master's Thesis Denoising Evaluation: {item['name']}", fontsize=16, fontweight="bold", color="white", y=0.98)
        plt.tight_layout()
        plt.savefig(item["filename"], dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  --> Saved: {item['filename']}")

    print("\nAll realistic comparisons generated successfully!")

if __name__ == "__main__":
    main()
