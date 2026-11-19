import os
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from src.noise.snic_noise import SNICNoiseGenerator
from src.models.adaptive_unet import AdaptiveDenoiser
from src.utils.metrics import calculate_psnr, calculate_ssim

def get_font(size: int, bold: bool = False):
    """Loads system TTF font with fallback to default."""
    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

def create_labeled_panel(img_pil: Image.Image, panel_num: int, title: str, subtitle: str, color_rgb: tuple, target_size=(850, 850)) -> Image.Image:
    """Creates a high-visibility, labeled panel with colored header badge and borders."""
    resized_img = img_pil.resize(target_size, Image.Resampling.LANCZOS)
    pw, ph = target_size
    banner_height = 95
    border_w = 4

    panel = Image.new("RGB", (pw + 2 * border_w, ph + banner_height + 2 * border_w), color=(20, 24, 33))
    draw = ImageDraw.Draw(panel)

    # Draw colored header bar
    draw.rectangle([(0, 0), (panel.width, banner_height)], fill=(28, 35, 51))
    draw.rectangle([(0, 0), (14, banner_height)], fill=color_rgb)

    # Badge circle with number
    badge_radius = 26
    bx, by = 50, banner_height // 2
    draw.ellipse([(bx - badge_radius, by - badge_radius), (bx + badge_radius, by + badge_radius)], fill=color_rgb)
    font_badge = get_font(28, bold=True)
    draw.text((bx - 8, by - 16), str(panel_num), fill=(255, 255, 255), font=font_badge)

    # Title & Subtitle text
    font_title = get_font(28, bold=True)
    font_sub = get_font(19, bold=False)
    draw.text((95, 14), title, fill=(255, 255, 255), font=font_title)
    draw.text((95, 52), subtitle, fill=(200, 210, 225), font=font_sub)

    # Paste image inside border
    panel.paste(resized_img, (border_w, banner_height + border_w))

    # Outer colored border
    draw.rectangle([(0, 0), (panel.width - 1, panel.height - 1)], outline=color_rgb, width=border_w)
    return panel

def run_adaptive_demo():
    parser = argparse.ArgumentParser(description="Demonstrate Adaptive Degradation-Aware Denoising Pipeline")
    parser.add_argument("--image", type=str, default="test_street.jpg", help="Input clean test image")
    parser.add_argument("--iso", type=float, default=3200.0, help="Camera sensor ISO level")
    parser.add_argument("--sensor", type=str, default="iphone11_main", help="Sensor profile")
    parser.add_argument("--ckpt", type=str, default="checkpoints/Model_E_seed42_best.pth", help="Trained Model E checkpoint")
    parser.add_argument("--output", type=str, default="adaptive_pipeline_demonstration.png", help="Output visual graphic")
    parser.add_argument("--crop_output", type=str, default="zoomed_denoising_comparison.png", help="Zoomed-in crop comparison")
    args = parser.parse_args()

    print("==========================================================================")
    print("      DEMONSTRATION: ADAPTIVE CAMERA-NOISE-AWARE IMAGE DENOISING          ")
    print("==========================================================================\n")

    if not os.path.exists(args.image):
        print(f"[Error] Image '{args.image}' not found!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Clean Reference
    clean_pil = Image.open(args.image).convert("RGB")
    clean_np = np.array(clean_pil, dtype=np.float32) / 255.0
    x_clean = torch.from_numpy(clean_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # 2. Inject SNIC Realistic Noise
    print(f"Step 1: Simulating Camera Noise (Sensor: {args.sensor}, ISO: {args.iso})...")
    snic = SNICNoiseGenerator(sensor_type=args.sensor)
    y_noisy, _ = snic.add_noise(x_clean, iso=args.iso)

    # 3. Load Adaptive Denoiser (Model E)
    print(f"Step 2: Running Adaptive Denoiser (Estimator + Modulation U-Net)...")
    denoiser = AdaptiveDenoiser(base_channels=32, embedding_dim=32).to(device)
    if os.path.exists(args.ckpt):
        denoiser.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"        Loaded trained checkpoint: {args.ckpt}")
    else:
        print(f"        Notice: Using initialized architecture ({args.ckpt} not found).")
    denoiser.eval()

    with torch.no_grad():
        x_hat, m_hat, z = denoiser(y_noisy)

    # 4. Calculate Metrics
    psnr_noisy = calculate_psnr(y_noisy, x_clean)
    ssim_noisy = calculate_ssim(y_noisy, x_clean)
    psnr_restored = calculate_psnr(x_hat, x_clean)
    ssim_restored = calculate_ssim(x_hat, x_clean)

    print("\n--------------------------------------------------------------------------")
    print("                      QUALITY PERFORMANCE METRICS                         ")
    print("--------------------------------------------------------------------------")
    print(f"  Noisy Input (ISO {args.iso}):       PSNR = {psnr_noisy:.2f} dB  |  SSIM = {ssim_noisy:.4f}")
    print(f"  Adaptive Restored Output:        PSNR = {psnr_restored:.2f} dB  |  SSIM = {ssim_restored:.4f}")
    print(f"  Restoration Improvement:         +{psnr_restored - psnr_noisy:.2f} dB PSNR Boost!")
    print(f"  Estimated Noise Embedding z norm: ||z||_2 = {torch.norm(z).item():.2f}")
    print("--------------------------------------------------------------------------\n")

    def tensor_to_pil(t, is_map=False):
        arr = t.squeeze(0).cpu().numpy()
        if is_map:
            arr = arr.squeeze(0)  # [H, W]
            arr = np.clip((arr / (arr.max() + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="L").convert("RGB")
        else:
            arr = np.transpose(arr, (1, 2, 0))  # [H, W, 3]
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)

    noisy_pil = tensor_to_pil(y_noisy)
    m_hat_pil = tensor_to_pil(m_hat, is_map=True)
    restored_pil = tensor_to_pil(x_hat)

    # -------------------------------------------------------------------------
    # PART 1: 2x2 HIGH-CONTRAST MASTER DEMONSTRATION GRAPHIC
    # -------------------------------------------------------------------------
    print("Step 3: Generating High-Clarity 2x2 Master Demonstration Graphic...")
    panel_size = (850, 850)

    # Panel 1: Clean (Emerald Green #10b981)
    p1 = create_labeled_panel(
        clean_pil,
        panel_num=1,
        title="CLEAN REFERENCE IMAGE (X)",
        subtitle="Ground Truth Target | Original High-Resolution Photo",
        color_rgb=(16, 185, 129),
        target_size=panel_size
    )

    # Panel 2: Noisy (Crimson Red #ef4444)
    p2 = create_labeled_panel(
        noisy_pil,
        panel_num=2,
        title="NOISY CAMERA INPUT (Y)",
        subtitle=f"Severe Degradation | Sensor: {args.sensor} | ISO {int(args.iso)}",
        color_rgb=(239, 68, 68),
        target_size=panel_size
    )

    # Panel 3: Estimated Noise Map (Amber / Gold #f59e0b)
    p3 = create_labeled_panel(
        m_hat_pil,
        panel_num=3,
        title="ESTIMATED NOISE-STRENGTH MAP (M_hat)",
        subtitle=f"Spatial Degradation Prior from Scout CNN | ||z|| = {torch.norm(z).item():.2f}",
        color_rgb=(245, 158, 11),
        target_size=panel_size
    )

    # Panel 4: Restored Clean (Electric Blue #3b82f6)
    p4 = create_labeled_panel(
        restored_pil,
        panel_num=4,
        title="ADAPTIVELY RESTORED OUTPUT (X_hat)",
        subtitle=f"Restored via Feature-Modulated U-Net | Residual Subtracted",
        color_rgb=(59, 130, 246),
        target_size=panel_size
    )

    pad = 30
    title_height = 140
    total_w = p1.width * 2 + pad * 3
    total_h = title_height + p1.height * 2 + pad * 3

    canvas = Image.new("RGB", (total_w, total_h), color=(15, 20, 30))
    canvas_draw = ImageDraw.Draw(canvas)

    # Draw Top Master Title Banner
    canvas_draw.rectangle([(0, 0), (total_w, title_height)], fill=(24, 32, 47))
    canvas_draw.rectangle([(0, title_height - 6), (total_w, title_height)], fill=(59, 130, 246))

    font_main = get_font(38, bold=True)
    font_sub_main = get_font(22, bold=False)
    canvas_draw.text((pad, 25), "ADAPTIVE NOISE-AWARE U-NET (MODEL E) — COMPLETE RESTORATION PIPELINE", fill=(255, 255, 255), font=font_main)
    canvas_draw.text((pad, 80), f"Camera Noise Simulation & Restoration Benchmark | Sensor: {args.sensor} | Sensitivity: ISO {int(args.iso)}", fill=(180, 195, 220), font=font_sub_main)

    # Paste 2x2 panels
    x_col1 = pad
    x_col2 = pad * 2 + p1.width
    y_row1 = title_height + pad
    y_row2 = title_height + pad * 2 + p1.height

    canvas.paste(p1, (x_col1, y_row1))
    canvas.paste(p2, (x_col2, y_row1))
    canvas.paste(p3, (x_col1, y_row2))
    canvas.paste(p4, (x_col2, y_row2))

    canvas.save(args.output, quality=95)
    print(f">>> [1/2] Master 2x2 graphic saved to: {args.output} <<<")

    # -------------------------------------------------------------------------
    # PART 2: HIGH-MAGNIFICATION ZOOMED-IN COMPARISON (CROP)
    # -------------------------------------------------------------------------
    print("Step 4: Generating High-Magnification Zoomed Detail Comparison...")
    # Crop a 350x350 textured patch (stone facade and shutters)
    crop_x1, crop_y1 = 350, 350
    crop_x2, crop_y2 = 720, 720
    crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)

    crop_clean = clean_pil.crop(crop_box).resize((600, 600), Image.Resampling.LANCZOS)
    crop_noisy = noisy_pil.crop(crop_box).resize((600, 600), Image.Resampling.LANCZOS)
    crop_restored = restored_pil.crop(crop_box).resize((600, 600), Image.Resampling.LANCZOS)

    c_panel1 = create_labeled_panel(
        crop_clean,
        panel_num=1,
        title="CLEAN REFERENCE (Zoomed)",
        subtitle="Original Texture & Edge Ground Truth",
        color_rgb=(16, 185, 129),
        target_size=(600, 600)
    )

    c_panel2 = create_labeled_panel(
        crop_noisy,
        panel_num=2,
        title="NOISY INPUT (ISO 3200 Zoomed)",
        subtitle="Notice Heavy High-Frequency RGB Noise Grains",
        color_rgb=(239, 68, 68),
        target_size=(600, 600)
    )

    c_panel3 = create_labeled_panel(
        crop_restored,
        panel_num=3,
        title="ADAPTIVE RESTORED (Zoomed)",
        subtitle="Noise Grains Removed & Edges Preserved",
        color_rgb=(59, 130, 246),
        target_size=(600, 600)
    )

    crop_total_w = c_panel1.width * 3 + pad * 4
    crop_total_h = title_height + c_panel1.height + pad * 2
    crop_canvas = Image.new("RGB", (crop_total_w, crop_total_h), color=(15, 20, 30))
    crop_draw = ImageDraw.Draw(crop_canvas)

    crop_draw.rectangle([(0, 0), (crop_total_w, title_height)], fill=(24, 32, 47))
    crop_draw.rectangle([(0, title_height - 6), (crop_total_w, title_height)], fill=(16, 185, 129))

    crop_draw.text((pad, 25), "ZOOMED-IN NOISE VS. RESTORATION TEXTURE COMPARISON (HIGH MAGNIFICATION)", fill=(255, 255, 255), font=font_main)
    crop_draw.text((pad, 80), f"Side-by-Side Texture Crop | Compare the heavy noise grains in (2) against the clean restored facade in (3)", fill=(180, 195, 220), font=font_sub_main)

    cx1 = pad
    cx2 = pad * 2 + c_panel1.width
    cx3 = pad * 3 + c_panel1.width * 2
    cy = title_height + pad

    crop_canvas.paste(c_panel1, (cx1, cy))
    crop_canvas.paste(c_panel2, (cx2, cy))
    crop_canvas.paste(c_panel3, (cx3, cy))

    crop_canvas.save(args.crop_output, quality=95)
    print(f">>> [2/2] Zoomed comparison graphic saved to: {args.crop_output} <<<\n")

if __name__ == "__main__":
    run_adaptive_demo()
