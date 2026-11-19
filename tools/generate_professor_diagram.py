import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_card(ax, x, y, w, h, title, subtitle="", color="#3b82f6", text_color="#ffffff", ec="#0f172a", lw=1.5, radius=0.08, title_size=10.0, sub_size=8.0):
    """Draws a clean, presentation-grade card with title and subtitle."""
    card = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=color, edgecolor=ec, linewidth=lw, zorder=3
    )
    ax.add_patch(card)
    
    if subtitle:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=text_color, zorder=4)
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=sub_size, color=text_color, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=text_color, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color="#475569", lw=1.8, ls="-", label="", label_color=None, label_side="right", label_size=7.5, zorder=2):
    """Draws a crisp directional arrow with optional label."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls, shrinkA=2, shrinkB=2),
        zorder=zorder
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        lc = label_color if label_color else color
        if label_side == "right":
            ax.text(mx + 0.12, my, label, ha="left", va="center", fontsize=label_size, fontweight="bold", color=lc, zorder=zorder+3)
        elif label_side == "left":
            ax.text(mx - 0.12, my, label, ha="right", va="center", fontsize=label_size, fontweight="bold", color=lc, zorder=zorder+3)
        elif label_side == "top":
            ax.text(mx, my + 0.12, label, ha="center", va="bottom", fontsize=label_size, fontweight="bold", color=lc, zorder=zorder+3)
        else:
            ax.text(mx, my - 0.12, label, ha="center", va="top", fontsize=label_size, fontweight="bold", color=lc, zorder=zorder+3)

def generate_base_model_d():
    """Generates standalone Base Model D diagram (ultra-clean U-Net)."""
    fig, ax = plt.subplots(figsize=(12, 13.5), facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 13.5)
    ax.axis("off")
    
    # Title Banner
    head_box = patches.FancyBboxPatch(
        (0.5, 12.1), 10.5, 1.1,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1.5, zorder=1
    )
    ax.add_patch(head_box)
    ax.text(5.75, 12.8, "BASELINE: STANDARD STATIC U-NET (MODEL D)", ha="center", fontsize=15, fontweight="bold", color="#0f172a", zorder=2)
    ax.text(5.75, 12.35, "3,195,331 Parameters | Real SIDD Benchmark: 36.10 dB PSNR | Fixed Static Filters", ha="center", fontsize=9.5, color="#64748b", zorder=2)
    
    # Top Row: Input y & Output x
    draw_card(ax, 0.6, 10.5, 2.8, 0.9, "Noisy Photo (y)", "[3 × H × W] Smartphone Input", color="#ef4444", title_size=11, sub_size=8.5)
    draw_card(ax, 7.6, 10.5, 2.8, 0.9, "Clean Output (x̂)", "clamp(y - r̂) | 36.10 dB PSNR", color="#10b981", title_size=11, sub_size=8.5)
    
    # Global Residual Learning
    ax.annotate("", xy=(7.6, 10.95), xytext=(3.4, 10.95),
                arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2.2, linestyle=":", shrinkA=3, shrinkB=3))
    ax.text(5.5, 11.15, "Residual Learning: x̂ = y - r̂", ha="center", fontsize=9.0, fontweight="bold", color="#dc2626")
    
    # Residual Head
    draw_card(ax, 7.6, 9.75, 2.8, 0.55, "Residual Head (outc)", "Conv 1×1: 32 → 3 ch (noise r̂)", color="#6366f1", title_size=9.5, sub_size=7.5)
    
    # Row 1 (H x W): Enc 1 & Dec 3
    draw_card(ax, 0.6, 8.8, 2.8, 0.8, "Encoder Level 1", "DoubleConv: 3 → 32 ch | H × W", color="#0284c7")
    draw_card(ax, 7.6, 8.8, 2.8, 0.8, "Decoder Level 3", "DoubleConv: 64 → 32 ch | H × W", color="#059669")
    
    # Row 2 (H/2 x W/2): Enc 2 & Dec 2
    draw_card(ax, 0.6, 7.1, 2.8, 0.8, "Encoder Level 2", "DoubleConv: 32 → 64 ch | H/2 × W/2", color="#0284c7")
    draw_card(ax, 7.6, 7.1, 2.8, 0.8, "Decoder Level 2", "DoubleConv: 128 → 64 ch | H/2 × W/2", color="#059669")
    
    # Row 3 (H/4 x W/4): Enc 3 & Dec 1
    draw_card(ax, 0.6, 5.4, 2.8, 0.8, "Encoder Level 3", "DoubleConv: 64 → 128 ch | H/4 × W/4", color="#0284c7")
    draw_card(ax, 7.6, 5.4, 2.8, 0.8, "Decoder Level 1", "DoubleConv: 256 → 128 ch | H/4 × W/4", color="#059669")
    
    # Row 4 (H/8 x W/8): Enc 4, Bottleneck, UpConv Block
    draw_card(ax, 0.6, 3.6, 2.8, 0.85, "Encoder Level 4", "DoubleConv: 128 → 256 ch | H/8 × W/8", color="#0284c7")
    draw_card(ax, 4.1, 3.6, 2.8, 0.85, "Bottleneck", "DoubleConv: 256 → 256 ch (Static)", color="#f59e0b", title_size=11, sub_size=8.5)
    draw_card(ax, 7.6, 3.6, 2.8, 0.85, "UpConv Block", "ConvTranspose 256 → 128 ch", color="#059669")
    
    # Vertical flows
    draw_arrow(ax, 2.0, 10.5, 2.0, 9.6)
    draw_arrow(ax, 2.0, 8.8, 2.0, 7.9, label="Downsample ÷2")
    draw_arrow(ax, 2.0, 7.1, 2.0, 6.2, label="Downsample ÷2")
    draw_arrow(ax, 2.0, 5.4, 2.0, 4.45, label="Downsample ÷2")
    
    draw_arrow(ax, 3.4, 4.02, 4.1, 4.02)
    draw_arrow(ax, 6.9, 4.02, 7.6, 4.02)
    
    draw_arrow(ax, 9.0, 4.45, 9.0, 5.4, label="Up to Dec 1")
    draw_arrow(ax, 9.0, 6.2, 9.0, 7.1, label="Upsample ×2")
    draw_arrow(ax, 9.0, 7.9, 9.0, 8.8, label="Upsample ×2")
    draw_arrow(ax, 9.0, 9.6, 9.0, 9.75)
    draw_arrow(ax, 9.0, 10.3, 9.0, 10.5)
    
    # Horizontal Skip Connections
    draw_arrow(ax, 3.4, 9.2, 7.6, 9.2, color="#94a3b8", ls="--", label="Skip: Concat 32 ch", label_side="top")
    draw_arrow(ax, 3.4, 7.5, 7.6, 7.5, color="#94a3b8", ls="--", label="Skip: Concat 64 ch", label_side="top")
    draw_arrow(ax, 3.4, 5.8, 7.6, 5.8, color="#94a3b8", ls="--", label="Skip: Concat 128 ch", label_side="top")
    
    # Limitation Card
    limit_card = patches.FancyBboxPatch(
        (0.6, 0.4), 9.8, 2.4,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#fef2f2", edgecolor="#fca5a5", linewidth=1.5, zorder=2
    )
    ax.add_patch(limit_card)
    ax.text(5.5, 2.4, "CORE LIMITATION OF BASELINE MODEL D (STATIC U-NET)", ha="center", fontsize=11, fontweight="bold", color="#991b1b")
    ax.text(5.5, 1.95, "• Fixed Static Weights: Filters are frozen after training. Zero adaptability during inference.", ha="center", fontsize=9.0, color="#7f1d1d")
    ax.text(5.5, 1.5, "• Blind to Camera Sensor & ISO: Treats ISO 100 (clean daylight) and ISO 6400 (severe night grain) identically.", ha="center", fontsize=9.0, color="#7f1d1d")
    ax.text(5.5, 1.05, "• Failure Mode: Over-smooths sharp edges on low-noise images; leaves coarse color blotches on high-noise images.", ha="center", fontsize=9.0, color="#7f1d1d")
    ax.text(5.5, 0.65, "• Benchmark Ceiling: Caps out at 36.10 dB PSNR on real smartphone photos (SIDD benchmark).", ha="center", fontsize=9.0, color="#7f1d1d")
    
    plt.tight_layout()
    out = "results/figures/base_model_d_flowchart.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out}")

def generate_proposed_model_e():
    """Generates standalone Proposed Model E flowchart with clean outer-bus FiLM routing."""
    fig, ax = plt.subplots(figsize=(15.5, 16.5), facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(-1.8, 15.8)
    ax.set_ylim(-0.5, 16.5)
    ax.axis("off")
    
    # Title Header
    head_box = patches.FancyBboxPatch(
        (-0.8, 15.1), 15.6, 1.1,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#eff6ff", edgecolor="#bfdbfe", linewidth=1.5, zorder=1
    )
    ax.add_patch(head_box)
    ax.text(7.0, 15.8, "PROPOSED: ADAPTIVE NOISE-AWARE U-NET (MODEL E)", ha="center", fontsize=15, fontweight="bold", color="#1e40af", zorder=2)
    ax.text(7.0, 15.35, "3,241,156 Parameters (+1.43% capacity matched) | Real SIDD: 37.52 dB (+1.41 dB Gain | p = 2.07e-18)", ha="center", fontsize=9.5, color="#2563eb", zorder=2)
    
    # -------------------------------------------------------------------------
    # STAGE 1: NOISE SCOUT (TOP DIAGNOSIS TRACK)
    # -------------------------------------------------------------------------
    scout_box = patches.FancyBboxPatch(
        (-0.8, 13.0), 15.6, 1.8,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#f0fdf4", edgecolor="#86efac", linewidth=1.5, zorder=1
    )
    ax.add_patch(scout_box)
    ax.text(-0.5, 14.45, "STAGE 1: NOISE SCOUT (Ultra-Lightweight Diagnosis: 20,481 params / 0.6% overhead)", fontsize=10.5, fontweight="bold", color="#166534")
    
    draw_card(ax, -0.5, 13.2, 2.4, 0.95, "Noisy Photo (y)", "RGB Image [3×H×W]", color="#ef4444", title_size=10.5, sub_size=8.0)
    draw_card(ax, 2.6, 13.2, 2.6, 0.95, "Scout CNN", "3 Conv: 3→16→16→1", color="#2563eb", title_size=10.5, sub_size=8.0)
    draw_card(ax, 5.9, 13.2, 2.8, 0.95, "Spatial Noise Map (M̂)", "Pixel-wise noise variance", color="#7c3aed", title_size=10.5, sub_size=8.0)
    draw_card(ax, 9.4, 13.2, 2.6, 0.95, "Noise Encoder", "Pool(4×4) + MLP (16→32)", color="#9333ea", title_size=10.5, sub_size=8.0)
    draw_card(ax, 12.7, 13.2, 1.8, 0.95, "Fingerprint z", "32-D Latent Vector", color="#db2777", title_size=10.5, sub_size=7.5)
    
    draw_arrow(ax, 1.9, 13.68, 2.6, 13.68)
    draw_arrow(ax, 5.2, 13.68, 5.9, 13.68)
    draw_arrow(ax, 8.7, 13.68, 9.4, 13.68)
    draw_arrow(ax, 12.0, 13.68, 12.7, 13.68)
    
    # -------------------------------------------------------------------------
    # STAGE 2: ADAPTIVE RESTORATION TRUNK
    # -------------------------------------------------------------------------
    stage2_badge = patches.FancyBboxPatch(
        (2.8, 12.35), 8.4, 0.45,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor="#ecfdf5", edgecolor="#a7f3d0", linewidth=1.0, zorder=2
    )
    ax.add_patch(stage2_badge)
    ax.text(7.0, 12.57, "STAGE 2: ADAPTIVE RESTORATION TRUNK (Identical 3.195M Backbone + 3 FiLM Equalizers)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#047857")
    
    # Input y feed into Stage 2 Encoder
    draw_arrow(ax, 0.7, 13.2, 0.7, 11.45)
    
    # Top Row: Input y & Output x
    draw_card(ax, 0.6, 10.55, 2.8, 0.9, "Noisy Photo (y)", "[3 × H × W] Smartphone Input", color="#ef4444", title_size=11, sub_size=8.5)
    draw_card(ax, 10.6, 10.55, 2.8, 0.9, "Clean Output (x̂)", "clamp(y - r̂) | 37.52 dB (+1.41 dB)", color="#10b981", title_size=11, sub_size=8.5)
    
    # Global Residual Learning Arrow
    ax.annotate("", xy=(10.6, 11.0), xytext=(3.4, 11.0),
                 arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2.2, linestyle=":", shrinkA=3, shrinkB=3))
    ax.text(7.0, 11.2, "Residual Learning: x̂ = y - r̂", ha="center", fontsize=9.0, fontweight="bold", color="#dc2626")
    
    # Residual Head
    draw_card(ax, 10.6, 9.8, 2.8, 0.55, "Residual Head (outc)", "Conv 1×1: 32 → 3 ch (noise r̂)", color="#6366f1", title_size=9.5, sub_size=7.5)
    
    # Row 1 (H x W): Enc 1 & Dec 3
    draw_card(ax, 0.6, 8.85, 2.8, 0.8, "Encoder Level 1", "DoubleConv: 3 → 32 ch | Res: H × W", color="#0284c7")
    draw_card(ax, 10.6, 8.85, 2.8, 0.8, "Decoder Level 3", "DoubleConv: 64 → 32 ch | Res: H × W", color="#059669")
    
    # Row 2 (H/2 x W/2): Enc 2 + FiLM 1 & Dec 2 + FiLM 3
    draw_card(ax, 0.6, 7.65, 2.8, 0.65, "Encoder Level 2", "DoubleConv: 32 → 64 ch", color="#0284c7")
    draw_card(ax, 0.6, 6.85, 2.8, 0.68, "FiLM Layer 1 (MOD 1)", "Modulates 64 ch with vector z", color="#06b6d4")
    
    draw_card(ax, 10.6, 7.65, 2.8, 0.65, "Decoder Level 2", "DoubleConv: 128 → 64 ch", color="#059669")
    draw_card(ax, 10.6, 6.85, 2.8, 0.68, "FiLM Layer 3 (MOD 3)", "Modulates 64 ch with vector z", color="#06b6d4")
    
    # Row 3 (H/4 x W/4): Enc 3 & Dec 1
    draw_card(ax, 0.6, 5.3, 2.8, 0.8, "Encoder Level 3", "128 channels | Res: H/4 × W/4", color="#0284c7")
    draw_card(ax, 10.6, 5.3, 2.8, 0.8, "Decoder Level 1", "128 channels | Res: H/4 × W/4", color="#059669")
    
    # Row 4 (H/8 x W/8): Enc 4, Bottleneck + FiLM 2, UpConv Block
    draw_card(ax, 0.6, 3.5, 2.8, 0.85, "Encoder Level 4", "256 channels | Res: H/8 × W/8", color="#0284c7")
    draw_card(ax, 4.7, 4.0, 4.6, 0.65, "Bottleneck", "DoubleConv: 256 → 256 ch", color="#f59e0b", title_size=10.5, sub_size=7.5)
    draw_card(ax, 4.7, 3.25, 4.6, 0.68, "FiLM Layer 2 (MOD 2)", "Modulates 256 bottleneck ch with vector z", color="#06b6d4")
    draw_card(ax, 10.6, 3.5, 2.8, 0.85, "UpConv Block", "ConvTranspose 256 → 128 ch", color="#059669")
    
    # Flows
    draw_arrow(ax, 2.0, 10.55, 2.0, 9.65)
    draw_arrow(ax, 2.0, 8.85, 2.0, 8.3, label="Downsample ÷2")
    draw_arrow(ax, 2.0, 7.65, 2.0, 7.53)
    draw_arrow(ax, 2.0, 6.85, 2.0, 6.1, label="Downsample ÷2")
    draw_arrow(ax, 2.0, 5.3, 2.0, 4.35, label="Downsample ÷2")
    
    draw_arrow(ax, 3.4, 3.92, 4.7, 4.32)
    draw_arrow(ax, 7.0, 4.0, 7.0, 3.93)
    draw_arrow(ax, 9.3, 3.59, 10.6, 3.92)
    
    draw_arrow(ax, 12.0, 4.35, 12.0, 5.3, label="Up to Dec 1")
    draw_arrow(ax, 12.0, 6.1, 12.0, 6.85, label="Upsample ×2")
    draw_arrow(ax, 12.0, 7.53, 12.0, 7.65)
    draw_arrow(ax, 12.0, 8.3, 12.0, 8.85, label="Upsample ×2")
    draw_arrow(ax, 12.0, 9.65, 12.0, 9.8)
    draw_arrow(ax, 12.0, 10.35, 12.0, 10.55)
    
    # Horizontal Skip Connections
    draw_arrow(ax, 3.4, 9.25, 10.6, 9.25, color="#94a3b8", ls="--", label="Skip: Concat 32 ch", label_side="top")
    draw_arrow(ax, 3.4, 7.25, 10.6, 7.25, color="#94a3b8", ls="--", label="Skip: Concat modulated 64 ch", label_side="top")
    draw_arrow(ax, 3.4, 5.7, 10.6, 5.7, color="#94a3b8", ls="--", label="Skip: Concat 128 ch", label_side="top")
    
    # -------------------------------------------------------------------------
    # CLEAN EXTERNAL PERIMETER BUS FOR FiLM (Zero overlaps with internal nodes)
    # -------------------------------------------------------------------------
    # Route 1: From Fingerprint z down the right margin to FiLM 3
    ax.plot([13.6, 14.8, 14.8], [13.2, 13.2, 7.19], color="#db2777", lw=2.2, zorder=2)
    ax.annotate("", xy=(13.4, 7.19), xytext=(14.8, 7.19), arrowprops=dict(arrowstyle="->", color="#db2777", lw=2.2), zorder=2)
    ax.text(14.9, 8.5, "Tune FiLM 3 (64 ch)", ha="left", va="center", fontsize=8.2, fontweight="bold", color="#be185d", rotation=-90)
    
    # Route 2: Continues down right margin, wraps under U-Net at y=2.65, taps UP into FiLM 2
    ax.plot([14.8, 14.8, 7.0], [7.19, 2.65, 2.65], color="#db2777", lw=2.2, zorder=2)
    ax.annotate("", xy=(7.0, 3.25), xytext=(7.0, 2.65), arrowprops=dict(arrowstyle="->", color="#db2777", lw=2.2), zorder=2)
    ax.text(7.15, 2.9, "Tune FiLM 2 (Bottleneck 256 ch)", ha="left", va="center", fontsize=8.2, fontweight="bold", color="#be185d")
    
    # Route 3: Continues across bottom to left margin (x=-1.2), goes UP to y=7.19, taps RIGHT into FiLM 1
    ax.plot([7.0, -1.2, -1.2], [2.65, 2.65, 7.19], color="#db2777", lw=2.2, zorder=2)
    ax.annotate("", xy=(0.6, 7.19), xytext=(-1.2, 7.19), arrowprops=dict(arrowstyle="->", color="#db2777", lw=2.2), zorder=2)
    ax.text(-1.3, 8.5, "Tune FiLM 1 (64 ch)", ha="right", va="center", fontsize=8.2, fontweight="bold", color="#be185d", rotation=90)
    
    # FiLM Explainer Card (Bottom)
    film_card = patches.FancyBboxPatch(
        (-0.8, 0.4), 15.6, 2.0,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#ecfeff", edgecolor="#06b6d4", linewidth=1.5, zorder=2
    )
    ax.add_patch(film_card)
    ax.text(7.0, 2.05, "HOW FiLM FEATURE MODULATION WORKS (THE 'AUDIO EQUALIZER' PRINCIPLE)", ha="center", fontsize=11.5, fontweight="bold", color="#0e7490")
    ax.text(7.0, 1.65, r"$\mathbf{F}_{\mathrm{out}} = (\gamma(z) + 1.0) \odot \mathbf{F}_{\mathrm{in}} + \beta(z)$", ha="center", fontsize=12.5, fontweight="bold", color="#0f172a")
    ax.text(7.0, 1.2, "• γ(z) = Dynamic Gain ('Volume Knob'): Tunes filter sensitivity up for true edges and down for noisy frequencies.", ha="center", fontsize=9.0, color="#155e75")
    ax.text(7.0, 0.8, "• β(z) = Dynamic Bias ('Tone Knob'): Shifts feature baselines to cancel sensor electronic color casts & dark current.", ha="center", fontsize=9.0, color="#155e75")
    
    plt.tight_layout()
    out = "results/figures/proposed_model_e_flowchart.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out}")

def generate_comparison_master():
    """Generates the unified 2-panel comparison diagram with 100% symmetric backbones and clean routing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(27, 16.5), facecolor="#ffffff")
    
    # =========================================================================
    # PANEL 1: BASELINE MODEL D
    # =========================================================================
    ax1.set_facecolor("#f8fafc")
    ax1.set_xlim(-1.5, 13.5)
    ax1.set_ylim(-0.5, 16.5)
    ax1.axis("off")
    
    head1 = patches.FancyBboxPatch(
        (-0.8, 15.1), 13.8, 1.1,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1.5, zorder=1
    )
    ax1.add_patch(head1)
    ax1.text(6.1, 15.8, "BASELINE: STANDARD STATIC U-NET (MODEL D)", ha="center", fontsize=14, fontweight="bold", color="#0f172a", zorder=2)
    ax1.text(6.1, 15.35, "3,195,331 Parameters | Real SIDD Benchmark: 36.10 dB PSNR | Fixed Static Filters", ha="center", fontsize=9.5, color="#64748b", zorder=2)
    
    # Placeholder card to align vertically with Stage 1 in Model E
    blank_box = patches.FancyBboxPatch(
        (-0.8, 13.0), 13.8, 1.8,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.5, linestyle="--", zorder=1
    )
    ax1.add_patch(blank_box)
    ax1.text(6.1, 14.1, "NO NOISE SCOUT (Model is blind to camera ISO / noise severity)", ha="center", va="center", fontsize=11, fontweight="bold", color="#94a3b8")
    ax1.text(6.1, 13.6, "Static pipeline processes every image blindly with the same fixed weights", ha="center", va="center", fontsize=9.0, color="#94a3b8")
    
    stage1_badge = patches.FancyBboxPatch(
        (2.1, 12.35), 8.0, 0.45,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1.0, zorder=2
    )
    ax1.add_patch(stage1_badge)
    ax1.text(6.1, 12.57, "STATIC RESTORATION TRUNK (Zero Dynamic Feature Modulation)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#475569")
    
    # Input y & Output x
    draw_card(ax1, 0.6, 10.55, 2.8, 0.9, "Noisy Photo (y)", "[3 × H × W] Smartphone Input", color="#ef4444", title_size=11, sub_size=8.5)
    draw_card(ax1, 8.8, 10.55, 2.8, 0.9, "Clean Output (x̂)", "clamp(y - r̂) | 36.10 dB PSNR", color="#10b981", title_size=11, sub_size=8.5)
    
    # Global Residual Learning Arrow
    ax1.annotate("", xy=(8.8, 11.0), xytext=(3.4, 11.0),
                 arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2.2, linestyle=":", shrinkA=3, shrinkB=3))
    ax1.text(6.1, 11.2, "Residual Learning: x̂ = y - r̂", ha="center", fontsize=9.0, fontweight="bold", color="#dc2626")
    
    # Residual Head
    draw_card(ax1, 8.8, 9.8, 2.8, 0.55, "Residual Head (outc)", "Conv 1×1: 32 → 3 ch (noise r̂)", color="#6366f1", title_size=9.5, sub_size=7.5)
    
    # Row 1 (H x W): Enc 1 & Dec 3
    draw_card(ax1, 0.6, 8.85, 2.8, 0.8, "Encoder Level 1", "32 channels | Res: H × W", color="#0284c7")
    draw_card(ax1, 8.8, 8.85, 2.8, 0.8, "Decoder Level 3", "32 channels | Res: H × W", color="#059669")
    
    # Row 2 (H/2 x W/2): Enc 2 & Dec 2
    draw_card(ax1, 0.6, 7.1, 2.8, 0.8, "Encoder Level 2", "64 channels | Res: H/2 × W/2", color="#0284c7")
    draw_card(ax1, 8.8, 7.1, 2.8, 0.8, "Decoder Level 2", "64 channels | Res: H/2 × W/2", color="#059669")
    
    # Row 3 (H/4 x W/4): Enc 3 & Dec 1
    draw_card(ax1, 0.6, 5.3, 2.8, 0.8, "Encoder Level 3", "128 channels | Res: H/4 × W/4", color="#0284c7")
    draw_card(ax1, 8.8, 5.3, 2.8, 0.8, "Decoder Level 1", "128 channels | Res: H/4 × W/4", color="#059669")
    
    # Row 4 (H/8 x W/8): Enc 4, Bottleneck, UpConv Block
    draw_card(ax1, 0.6, 3.5, 2.8, 0.85, "Encoder Level 4", "256 channels | Res: H/8 × W/8", color="#0284c7")
    draw_card(ax1, 4.7, 3.5, 2.8, 0.85, "Bottleneck", "256 channels (Static Locked)", color="#f59e0b", title_size=11, sub_size=8.0)
    draw_card(ax1, 8.8, 3.5, 2.8, 0.85, "UpConv Block", "ConvTranspose 256 → 128 ch", color="#059669")
    
    # Flows
    draw_arrow(ax1, 2.0, 10.55, 2.0, 9.65)
    draw_arrow(ax1, 2.0, 8.85, 2.0, 7.9, label="Downsample ÷2")
    draw_arrow(ax1, 2.0, 7.1, 2.0, 6.1, label="Downsample ÷2")
    draw_arrow(ax1, 2.0, 5.3, 2.0, 4.35, label="Downsample ÷2")
    
    draw_arrow(ax1, 3.4, 3.92, 4.7, 3.92)
    draw_arrow(ax1, 7.5, 3.92, 8.8, 3.92)
    
    draw_arrow(ax1, 10.2, 4.35, 10.2, 5.3, label="Up to Dec 1")
    draw_arrow(ax1, 10.2, 6.1, 10.2, 7.1, label="Upsample ×2")
    draw_arrow(ax1, 10.2, 7.9, 10.2, 8.85, label="Upsample ×2")
    draw_arrow(ax1, 10.2, 9.65, 10.2, 9.8)
    draw_arrow(ax1, 10.2, 10.35, 10.2, 10.55)
    
    # Skips
    draw_arrow(ax1, 3.4, 9.25, 8.8, 9.25, color="#94a3b8", ls="--", label="Skip: Concat 32 ch", label_side="top")
    draw_arrow(ax1, 3.4, 7.5, 8.8, 7.5, color="#94a3b8", ls="--", label="Skip: Concat 64 ch", label_side="top")
    draw_arrow(ax1, 3.4, 5.7, 8.8, 5.7, color="#94a3b8", ls="--", label="Skip: Concat 128 ch", label_side="top")
    
    # Limitation Card
    limit_card = patches.FancyBboxPatch(
        (-0.8, 0.4), 13.8, 2.0,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#fef2f2", edgecolor="#fca5a5", linewidth=1.5, zorder=2
    )
    ax1.add_patch(limit_card)
    ax1.text(6.1, 2.05, "WHY THE BASELINE MODEL FAILS ON REAL SMARTPHONES", ha="center", fontsize=11, fontweight="bold", color="#991b1b")
    ax1.text(6.1, 1.6, "• 100% Fixed Static Weights: Filters never adapt after training.", ha="center", fontsize=9.0, color="#7f1d1d")
    ax1.text(6.1, 1.2, "• Blind to Camera Sensor & ISO: Treats ISO 100 and ISO 6400 identically.", ha="center", fontsize=9.0, color="#7f1d1d")
    ax1.text(6.1, 0.8, "• Result: Blurs sharp edges at low ISO; leaves coarse noise at high ISO.", ha="center", fontsize=9.0, color="#7f1d1d")
    
    # =========================================================================
    # PANEL 2: PROPOSED MODEL E
    # =========================================================================
    ax2.set_facecolor("#f8fafc")
    ax2.set_xlim(-1.8, 15.8)
    ax2.set_ylim(-0.5, 16.5)
    ax2.axis("off")
    
    head2 = patches.FancyBboxPatch(
        (-0.8, 15.1), 15.6, 1.1,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#eff6ff", edgecolor="#bfdbfe", linewidth=1.5, zorder=1
    )
    ax2.add_patch(head2)
    ax2.text(7.0, 15.8, "PROPOSED: ADAPTIVE NOISE-AWARE U-NET (MODEL E)", ha="center", fontsize=14, fontweight="bold", color="#1e40af", zorder=2)
    ax2.text(7.0, 15.35, "3,241,156 Parameters (+1.43% capacity matched) | Real SIDD: 37.52 dB (+1.41 dB Gain | p = 2.07e-18)", ha="center", fontsize=9.5, color="#2563eb", zorder=2)
    
    # Stage 1: Noise Scout
    scout_box2 = patches.FancyBboxPatch(
        (-0.8, 13.0), 15.6, 1.8,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#f0fdf4", edgecolor="#86efac", linewidth=1.5, zorder=1
    )
    ax2.add_patch(scout_box2)
    ax2.text(-0.5, 14.45, "STAGE 1: NOISE SCOUT (Ultra-Lightweight Diagnosis: 20,481 params / 0.6% overhead)", fontsize=10, fontweight="bold", color="#166534")
    
    draw_card(ax2, -0.5, 13.2, 2.4, 0.95, "Noisy Photo (y)", "RGB Image [3×H×W]", color="#ef4444", title_size=10, sub_size=7.5)
    draw_card(ax2, 2.6, 13.2, 2.6, 0.95, "Scout CNN", "3 Conv: 3→16→16→1", color="#2563eb", title_size=10, sub_size=7.5)
    draw_card(ax2, 5.9, 13.2, 2.8, 0.95, "Spatial Noise Map (M̂)", "Pixel-wise noise variance", color="#7c3aed", title_size=10, sub_size=7.5)
    draw_card(ax2, 9.4, 13.2, 2.6, 0.95, "Noise Encoder", "Pool(4×4) + MLP (16→32)", color="#9333ea", title_size=10.5, sub_size=7.5)
    draw_card(ax2, 12.7, 13.2, 1.8, 0.95, "Fingerprint z", "32-D Latent Vector", color="#db2777", title_size=10.5, sub_size=7.5)
    
    draw_arrow(ax2, 1.9, 13.68, 2.6, 13.68)
    draw_arrow(ax2, 5.2, 13.68, 5.9, 13.68)
    draw_arrow(ax2, 8.7, 13.68, 9.4, 13.68)
    draw_arrow(ax2, 12.0, 13.68, 12.7, 13.68)
    
    # Stage 2 Restorer
    stage2_badge2 = patches.FancyBboxPatch(
        (2.8, 12.35), 8.4, 0.45,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor="#ecfdf5", edgecolor="#a7f3d0", linewidth=1.0, zorder=2
    )
    ax2.add_patch(stage2_badge2)
    ax2.text(7.0, 12.57, "STAGE 2: ADAPTIVE RESTORATION TRUNK (Identical 3.195M Backbone + 3 FiLM Equalizers)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#047857")
    
    draw_arrow(ax2, 0.7, 13.2, 0.7, 11.45)
    
    # Global Residual Learning Arrow
    ax2.annotate("", xy=(10.6, 11.0), xytext=(3.4, 11.0),
                 arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2.2, linestyle=":", shrinkA=3, shrinkB=3))
    ax2.text(7.0, 11.2, "Residual Learning: x̂ = y - r̂", ha="center", fontsize=9.0, fontweight="bold", color="#dc2626")
    
    draw_card(ax2, 0.6, 10.55, 2.8, 0.9, "Noisy Photo (y)", "[3 × H × W] Smartphone Input", color="#ef4444", title_size=10.5, sub_size=8.0)
    draw_card(ax2, 10.6, 10.55, 2.8, 0.9, "Clean Output (x̂)", "clamp(y - r̂) | 37.52 dB (+1.41 dB)", color="#10b981", title_size=10.5, sub_size=8.0)
    draw_card(ax2, 10.6, 9.8, 2.8, 0.55, "Residual Head (outc)", "Conv 1×1: 32 → 3 ch (noise r̂)", color="#6366f1", title_size=9.0, sub_size=7.5)
    
    # Row 1: Enc 1 & Dec 3
    draw_card(ax2, 0.6, 8.85, 2.8, 0.8, "Encoder Level 1", "DoubleConv: 3 → 32 ch", color="#0284c7")
    draw_card(ax2, 10.6, 8.85, 2.8, 0.8, "Decoder Level 3", "DoubleConv: 64 → 32 ch", color="#059669")
    
    # Row 2: Enc 2 + FiLM 1 & Dec 2 + FiLM 3
    draw_card(ax2, 0.6, 7.65, 2.8, 0.65, "Encoder Level 2", "DoubleConv: 32 → 64 ch", color="#0284c7")
    draw_card(ax2, 0.6, 6.85, 2.8, 0.68, "FiLM Layer 1 (MOD 1)", "Modulates 64 ch with vector z", color="#06b6d4")
    
    draw_card(ax2, 10.6, 7.65, 2.8, 0.65, "Decoder Level 2", "DoubleConv: 128 → 64 ch", color="#059669")
    draw_card(ax2, 10.6, 6.85, 2.8, 0.68, "FiLM Layer 3 (MOD 3)", "Modulates 64 ch with vector z", color="#06b6d4")
    
    # Row 3: Enc 3 & Dec 1
    draw_card(ax2, 0.6, 5.3, 2.8, 0.8, "Encoder Level 3", "DoubleConv: 64 → 128 ch", color="#0284c7")
    draw_card(ax2, 10.6, 5.3, 2.8, 0.8, "Decoder Level 1", "DoubleConv: 256 → 128 ch", color="#059669")
    
    # Row 4: Enc 4, Bottleneck + FiLM 2, UpConv Block
    draw_card(ax2, 0.6, 3.5, 2.8, 0.85, "Encoder Level 4", "DoubleConv: 128 → 256 ch", color="#0284c7")
    draw_card(ax2, 4.7, 4.0, 4.6, 0.65, "Bottleneck", "DoubleConv: 256 → 256 ch", color="#f59e0b", title_size=10.5, sub_size=7.5)
    draw_card(ax2, 4.7, 3.25, 4.6, 0.68, "FiLM Layer 2 (MOD 2)", "Modulates 256 bottleneck ch with z", color="#06b6d4")
    draw_card(ax2, 10.6, 3.5, 2.8, 0.85, "UpConv Block", "ConvTranspose 256 → 128 ch", color="#059669")
    
    # Flows
    draw_arrow(ax2, 2.0, 10.55, 2.0, 9.65)
    draw_arrow(ax2, 2.0, 8.85, 2.0, 8.3, label="Downsample ÷2")
    draw_arrow(ax2, 2.0, 7.65, 2.0, 7.53)
    draw_arrow(ax2, 2.0, 6.85, 2.0, 6.1, label="Downsample ÷2")
    draw_arrow(ax2, 2.0, 5.3, 2.0, 4.35, label="Downsample ÷2")
    
    draw_arrow(ax2, 3.4, 3.92, 4.7, 4.32)
    draw_arrow(ax2, 7.0, 4.0, 7.0, 3.93)
    draw_arrow(ax2, 9.3, 3.59, 10.6, 3.92)
    
    draw_arrow(ax2, 12.0, 4.35, 12.0, 5.3, label="Up to Dec 1")
    draw_arrow(ax2, 12.0, 6.1, 12.0, 6.85, label="Upsample ×2")
    draw_arrow(ax2, 12.0, 7.53, 12.0, 7.65)
    draw_arrow(ax2, 12.0, 8.3, 12.0, 8.85, label="Upsample ×2")
    draw_arrow(ax2, 12.0, 9.65, 12.0, 9.8)
    draw_arrow(ax2, 12.0, 10.35, 12.0, 10.55)
    
    # Skips
    draw_arrow(ax2, 3.4, 9.25, 10.6, 9.25, color="#94a3b8", ls="--", label="Skip: Concat 32 ch", label_side="top")
    draw_arrow(ax2, 3.4, 7.25, 10.6, 7.25, color="#94a3b8", ls="--", label="Skip: Concat mod 64 ch", label_side="top")
    draw_arrow(ax2, 3.4, 5.7, 10.6, 5.7, color="#94a3b8", ls="--", label="Skip: Concat 128 ch", label_side="top")
    
    # Outer bus lines for FiLM
    ax2.plot([13.6, 14.8, 14.8], [13.2, 13.2, 7.19], color="#db2777", lw=2.2, zorder=2)
    ax2.annotate("", xy=(13.4, 7.19), xytext=(14.8, 7.19), arrowprops=dict(arrowstyle="->", color="#db2777", lw=2.2), zorder=2)
    ax2.text(14.9, 8.5, "Tune FiLM 3 (64 ch)", ha="left", va="center", fontsize=8.2, fontweight="bold", color="#be185d", rotation=-90)
    
    ax2.plot([14.8, 14.8, 7.0], [7.19, 2.65, 2.65], color="#db2777", lw=2.2, zorder=2)
    ax2.annotate("", xy=(7.0, 3.25), xytext=(7.0, 2.65), arrowprops=dict(arrowstyle="->", color="#db2777", lw=2.2), zorder=2)
    ax2.text(7.15, 2.9, "Tune FiLM 2 (Bottleneck 256 ch)", ha="left", va="center", fontsize=8.2, fontweight="bold", color="#be185d")
    
    ax2.plot([7.0, -1.2, -1.2], [2.65, 2.65, 7.19], color="#db2777", lw=2.2, zorder=2)
    ax2.annotate("", xy=(0.6, 7.19), xytext=(-1.2, 7.19), arrowprops=dict(arrowstyle="->", color="#db2777", lw=2.2), zorder=2)
    ax2.text(-1.3, 8.5, "Tune FiLM 1 (64 ch)", ha="right", va="center", fontsize=8.2, fontweight="bold", color="#be185d", rotation=90)
    
    # FiLM Explainer Card
    film_card2 = patches.FancyBboxPatch(
        (-0.8, 0.4), 15.6, 2.0,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#ecfeff", edgecolor="#06b6d4", linewidth=1.5, zorder=2
    )
    ax2.add_patch(film_card2)
    ax2.text(7.0, 2.05, "HOW FiLM FEATURE MODULATION WORKS (THE 'AUDIO EQUALIZER' PRINCIPLE)", ha="center", fontsize=11, fontweight="bold", color="#0e7490")
    ax2.text(7.0, 1.65, r"$\mathbf{F}_{\mathrm{out}} = (\gamma(z) + 1.0) \odot \mathbf{F}_{\mathrm{in}} + \beta(z)$", ha="center", fontsize=12, fontweight="bold", color="#0f172a")
    ax2.text(7.0, 1.2, "• γ(z) = Dynamic Gain ('Volume Knob'): Tunes filter sensitivity up for edges, down for noise.", ha="center", fontsize=8.8, color="#155e75")
    ax2.text(7.0, 0.8, "• β(z) = Dynamic Bias ('Tone Knob'): Shifts feature baselines to cancel sensor electronic color casts.", ha="center", fontsize=8.8, color="#155e75")
    
    plt.tight_layout()
    out = "results/figures/architecture_flowchart_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out}")

def main():
    os.makedirs("results/figures", exist_ok=True)
    generate_base_model_d()
    generate_proposed_model_e()
    generate_comparison_master()

if __name__ == "__main__":
    main()

