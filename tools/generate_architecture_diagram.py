import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pill(ax, x, y, w, h, title, subtitle="", color="#3b82f6", text_color="#ffffff", fontsize=9.5, ec="#1e293b", lw=1.2):
    """Draws a clean, rounded card with crisp text layout."""
    card = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=color, edgecolor=ec, linewidth=lw, zorder=3
    )
    ax.add_patch(card)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=fontsize - 2.0, color=text_color, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color="#475569", lw=1.6, ls="-"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls, shrinkA=3, shrinkB=3),
        zorder=2
    )

def generate_base_model_diagram():
    """Generates dedicated high-res flowchart for Base Model D."""
    fig, ax = plt.subplots(figsize=(11, 10.5), facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 12.0)
    ax.axis("off")

    ax.text(5.0, 11.4, "BASE MODEL (MODEL D) ARCHITECTURE", ha="center", fontsize=16, fontweight="bold", color="#0f172a")
    ax.text(5.0, 10.95, "Standard Static 4-Level U-Net | 3,195,331 Parameters | Real SIDD PSNR: 36.10 dB", ha="center", fontsize=10.5, color="#64748b")

    # Inputs & Outputs
    draw_pill(ax, 0.5, 9.4, 2.6, 0.85, "Noisy Input y", "[3 × H × W]", color="#ef4444")
    draw_pill(ax, 6.9, 9.4, 2.6, 0.85, "Clean Output x̂", "clamp(y - r̂) | 36.10 dB", color="#10b981")
    draw_pill(ax, 6.9, 8.0, 2.6, 0.8, "Residual Sub", "x̂ = y - r̂", color="#475569")
    draw_pill(ax, 6.9, 6.6, 2.6, 0.8, "Residual Head", "Conv2d: 32 → 3 (r̂)", color="#6366f1")

    # Encoder
    draw_pill(ax, 0.5, 8.0, 2.6, 0.8, "Encoder 1 (inc)", "DoubleConv [32 ch]", color="#0ea5e9")
    draw_pill(ax, 0.5, 6.2, 2.6, 0.8, "Encoder 2 (down1)", "DoubleConv [64 ch]", color="#0ea5e9")
    draw_pill(ax, 0.5, 4.4, 2.6, 0.8, "Encoder 3 (down2)", "DoubleConv [128 ch]", color="#0ea5e9")
    draw_pill(ax, 0.5, 2.6, 2.6, 0.8, "Encoder 4 (down3)", "DoubleConv [256 ch]", color="#0ea5e9")

    # Bottleneck
    draw_pill(ax, 3.7, 1.0, 3.0, 0.85, "Bottleneck", "DoubleConv [256 ch] (Static)", color="#f59e0b")

    # Decoder
    draw_pill(ax, 6.9, 2.6, 2.6, 0.8, "Decoder 1 (up1)", "UpConv + DoubleConv [128 ch]", color="#059669")
    draw_pill(ax, 6.9, 4.0, 2.6, 0.8, "Decoder 2 (up2)", "UpConv + DoubleConv [64 ch]", color="#059669")
    draw_pill(ax, 6.9, 5.3, 2.6, 0.8, "Decoder 3 (up3)", "UpConv + DoubleConv [32 ch]", color="#059669")

    # Encoder vertical flow
    draw_arrow(ax, 1.8, 9.4, 1.8, 8.8)
    draw_arrow(ax, 1.8, 8.0, 1.8, 7.0)
    draw_arrow(ax, 1.8, 6.2, 1.8, 5.2)
    draw_arrow(ax, 1.8, 4.4, 1.8, 3.4)
    draw_arrow(ax, 1.8, 2.6, 3.7, 1.42)

    ax.text(1.9, 7.45, "MaxPool ÷2", fontsize=8, color="#64748b")
    ax.text(1.9, 5.65, "MaxPool ÷2", fontsize=8, color="#64748b")
    ax.text(1.9, 3.85, "MaxPool ÷2", fontsize=8, color="#64748b")

    # Decoder vertical flow
    draw_arrow(ax, 6.7, 1.42, 8.2, 2.6)
    draw_arrow(ax, 8.2, 3.4, 8.2, 4.0)
    draw_arrow(ax, 8.2, 4.8, 8.2, 5.3)
    draw_arrow(ax, 8.2, 6.1, 8.2, 6.6)
    draw_arrow(ax, 8.2, 7.4, 8.2, 8.0)
    draw_arrow(ax, 8.2, 8.8, 8.2, 9.4)

    # Subtraction Line (Red dotted)
    ax.annotate("", xy=(6.9, 8.4), xytext=(3.1, 9.8),
                arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.6, linestyle=":", connectionstyle="arc3,rad=0.22"))

    # Skip Connections (Horizontal dashed)
    draw_arrow(ax, 3.1, 3.0, 6.9, 3.0, color="#94a3b8", ls="--")
    draw_arrow(ax, 3.1, 4.8, 6.9, 4.4, color="#94a3b8", ls="--")
    draw_arrow(ax, 3.1, 6.6, 6.9, 5.7, color="#94a3b8", ls="--")
    ax.text(5.0, 3.15, "Skip (256 ch)", ha="center", fontsize=8, color="#64748b")
    ax.text(5.0, 4.75, "Skip (128 ch)", ha="center", fontsize=8, color="#64748b")
    ax.text(5.0, 6.3, "Skip (64 ch)", ha="center", fontsize=8, color="#64748b")

    plt.tight_layout()
    out = "results/figures/base_model_d_flowchart.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out}")

def generate_proposed_model_diagram():
    """Generates dedicated high-res flowchart for Proposed Model E with central routing and zero crossings."""
    fig, ax = plt.subplots(figsize=(13.5, 12.0), facecolor="#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(-0.5, 12.5)
    ax.axis("off")

    ax.text(6.5, 12.0, "PROPOSED ADAPTIVE MODEL (MODEL E) ARCHITECTURE", ha="center", fontsize=16, fontweight="bold", color="#0f172a")
    ax.text(6.5, 11.55, "Adaptive Noise-Aware U-Net | 3,241,156 Parameters (+1.43%) | Real SIDD PSNR: 37.52 dB (+1.41 dB)", ha="center", fontsize=10.5, color="#64748b")

    # =========================================================================
    # BRANCH 1: SENSOR NOISE DIAGNOSIS (TOP BOX)
    # =========================================================================
    diag_box = patches.FancyBboxPatch(
        (0.4, 9.4), 8.8, 1.4,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=1.5, zorder=1
    )
    ax.add_patch(diag_box)
    ax.text(0.65, 10.55, "BRANCH 1: SENSOR NOISE DIAGNOSIS (20,481 Parameters)", fontsize=9.2, fontweight="bold", color="#1e40af", zorder=2)

    draw_pill(ax, 0.6, 9.6, 2.3, 0.75, "Noisy Input y", "[3 × H × W]", color="#ef4444")
    draw_pill(ax, 3.4, 9.6, 2.5, 0.75, "Scout CNN", "3→16→16→1 (14.6K)", color="#2563eb", fontsize=8.5)
    draw_pill(ax, 6.4, 9.6, 2.5, 0.75, "Noise Map M̂", "Spatial [1 × H × W]", color="#7c3aed", fontsize=8.5)

    draw_arrow(ax, 2.9, 9.97, 3.4, 9.97, color="#2563eb", lw=1.8)
    draw_arrow(ax, 5.9, 9.97, 6.4, 9.97, color="#7c3aed", lw=1.8)

    # Noise Encoder situated directly below Noise Map
    draw_pill(ax, 6.4, 7.8, 2.5, 0.75, "Noise Encoder", "Pool+Dense (5.8K)", color="#9333ea", fontsize=8.5)
    draw_arrow(ax, 7.65, 9.6, 7.65, 8.55, color="#9333ea", lw=1.8)
    ax.text(7.75, 9.07, "AdaptiveAvgPool(4×4)", fontsize=7.5, color="#6b21a8")

    # Central Degradation Latent z
    draw_pill(ax, 4.6, 5.4, 3.4, 0.9, "Degradation Latent z", "32-D Sensor Noise Fingerprint", color="#db2777", fontsize=9.2)
    draw_arrow(ax, 7.0, 7.8, 6.5, 6.3, color="#db2777", lw=2.0)
    ax.text(7.1, 7.1, "Latent z", color="#be185d", fontsize=8.0, fontweight="bold")

    # =========================================================================
    # BRANCH 2: ADAPTIVE RESTORATION TRUNK
    # =========================================================================
    ax.text(0.65, 8.85, "BRANCH 2: ADAPTIVE FEATURE-MODULATED RESTORATION (3,220,675 Parameters)", fontsize=9.2, fontweight="bold", color="#047857")

    # Left Column: Encoder
    draw_pill(ax, 0.6, 7.8, 2.5, 0.75, "Encoder 1 (inc)", "DoubleConv [32 ch]", color="#0ea5e9")
    draw_pill(ax, 0.6, 6.5, 2.5, 0.75, "Encoder 2 Conv", "DoubleConv [64 ch]", color="#0ea5e9")
    draw_pill(ax, 0.6, 5.4, 2.5, 0.8, "FiLM 1 (γ₁, β₁)", "Modulated by z [64 ch]", color="#06b6d4")
    draw_pill(ax, 0.6, 4.1, 2.5, 0.75, "Encoder 3 (down2)", "DoubleConv [128 ch]", color="#0ea5e9")
    draw_pill(ax, 0.6, 2.8, 2.5, 0.75, "Encoder 4 (down3)", "DoubleConv [256 ch]", color="#0ea5e9")

    # Bottom: Bottleneck
    draw_pill(ax, 4.6, 1.8, 3.4, 0.75, "Bottleneck Conv", "DoubleConv [256 ch]", color="#f59e0b")
    draw_pill(ax, 4.6, 0.7, 3.4, 0.8, "FiLM 2 (γ₂, β₂)", "Modulated by z [256 ch]", color="#06b6d4")

    # Right Column: Decoder & Outputs
    draw_pill(ax, 9.8, 2.8, 2.7, 0.75, "Decoder 1 (up1)", "UpConv + Conv [128 ch]", color="#059669")
    draw_pill(ax, 9.8, 4.1, 2.7, 0.75, "Decoder 2 Conv", "DoubleConv [64 ch]", color="#059669")
    draw_pill(ax, 9.8, 5.4, 2.7, 0.8, "FiLM 3 (γ₃, β₃)", "Modulated by z [64 ch]", color="#06b6d4")
    draw_pill(ax, 9.8, 6.7, 2.7, 0.75, "Decoder 3 (up3)", "UpConv + Conv [32 ch]", color="#059669")
    draw_pill(ax, 9.8, 7.8, 2.7, 0.75, "Residual Head", "Conv2d: 32 → 3 (r̂)", color="#6366f1")
    draw_pill(ax, 9.8, 8.9, 2.7, 0.75, "Residual Subtraction", "x̂ = y - r̂", color="#475569")
    draw_pill(ax, 9.8, 10.0, 2.7, 0.8, "Clean Output x̂", "37.52 dB (+1.41 dB)", color="#10b981")

    # Vertical flows in Encoder
    draw_arrow(ax, 1.85, 9.6, 1.85, 8.55)
    draw_arrow(ax, 1.85, 7.8, 1.85, 7.25)
    draw_arrow(ax, 1.85, 6.5, 1.85, 6.2)
    draw_arrow(ax, 1.85, 5.4, 1.85, 4.85)
    draw_arrow(ax, 1.85, 4.1, 1.85, 3.55)
    draw_arrow(ax, 1.85, 2.8, 4.6, 2.1)

    # Bottleneck flows
    draw_arrow(ax, 6.3, 1.8, 6.3, 1.5)
    draw_arrow(ax, 8.0, 1.1, 9.8, 2.8)

    # Vertical flows in Decoder
    draw_arrow(ax, 11.15, 3.55, 11.15, 4.1)
    draw_arrow(ax, 11.15, 4.85, 11.15, 5.4)
    draw_arrow(ax, 11.15, 6.2, 11.15, 6.7)
    draw_arrow(ax, 11.15, 7.45, 11.15, 7.8)
    draw_arrow(ax, 11.15, 8.55, 11.15, 8.9)
    draw_arrow(ax, 11.15, 9.65, 11.15, 10.0)

    # Skip connections
    draw_arrow(ax, 3.1, 3.17, 9.8, 3.17, color="#94a3b8", ls="--")
    draw_arrow(ax, 3.1, 4.47, 9.8, 4.47, color="#94a3b8", ls="--")
    ax.text(3.8, 3.3, "Skip (256 ch)", color="#64748b", fontsize=7.5)
    ax.text(3.8, 4.6, "Skip (128 ch)", color="#64748b", fontsize=7.5)

    # Top-routed y skip
    ax.plot([1.75, 1.75, 11.15, 11.15], [10.35, 11.1, 11.1, 9.65], color="#ef4444", lw=1.5, ls=":", zorder=2)
    ax.annotate("", xy=(11.15, 9.65), xytext=(11.15, 9.75),
                arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.5, shrinkA=0, shrinkB=0))
    ax.text(5.5, 11.22, "Noisy Input y (Skip for Residual Subtraction x̂ = y - r̂)", color="#b91c1c", fontsize=7.5, fontweight="bold", ha="center")

    # Direct FiLM modulation arrows
    draw_arrow(ax, 4.6, 5.8, 3.1, 5.8, color="#db2777", lw=2.4)
    ax.text(3.85, 6.05, "z → FiLM 1", color="#db2777", fontsize=8.5, fontweight="bold", ha="center")

    draw_arrow(ax, 6.3, 5.4, 6.3, 1.5, color="#db2777", lw=2.4)
    ax.text(6.65, 3.5, "z → FiLM 2", color="#db2777", fontsize=8.5, fontweight="bold", va="center")

    draw_arrow(ax, 8.0, 5.8, 9.8, 5.8, color="#db2777", lw=2.4)
    ax.text(8.9, 6.05, "z → FiLM 3", color="#db2777", fontsize=8.5, fontweight="bold", ha="center")

    # FiLM Formula Box (Bottom Left)
    film_math_box = patches.FancyBboxPatch(
        (0.6, 0.4), 3.4, 1.8,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#ecfeff", edgecolor="#06b6d4", linewidth=1.2, zorder=2
    )
    ax.add_patch(film_math_box)
    ax.text(2.3, 1.9, "FiLM Modulation Math", fontsize=8.5, fontweight="bold", color="#0e7490", ha="center", zorder=3)
    ax.text(2.3, 1.35, r"$\mathbf{F}_{\mathrm{out}} = \gamma(z) \odot \mathbf{F}_{\mathrm{in}} + \beta(z)$", fontsize=10, color="#0f172a", ha="center", zorder=3)
    ax.text(2.3, 0.75, r"$\gamma(z)$: Channel-wise scaling" + "\n" + r"$\beta(z)$: Channel-wise bias shift", fontsize=7.5, color="#334155", ha="center", zorder=3)

    plt.tight_layout()
    out = "results/figures/proposed_model_e_flowchart.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out}")

def generate_comparison_diagram():
    """Generates the combined side-by-side comparison diagram with zero overlaps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12.5), facecolor="#ffffff")

    # ---------------- PANEL 1: BASE MODEL D ----------------
    ax1.set_facecolor("#f8fafc")
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-0.5, 12.4)
    ax1.axis("off")

    ax1.text(5.0, 11.8, "BASE MODEL (MODEL D)", ha="center", fontsize=16, fontweight="bold", color="#0f172a")
    ax1.text(5.0, 11.35, "Standard Static 4-Level U-Net | 3,195,331 Parameters\nReal SIDD Benchmark: 36.10 dB PSNR | 0.8847 SSIM", ha="center", fontsize=10.5, color="#64748b")

    draw_pill(ax1, 0.5, 9.4, 2.6, 0.85, "Noisy Input y", "[3 × H × W]", color="#ef4444")
    draw_pill(ax1, 6.9, 9.4, 2.6, 0.85, "Clean Output x̂", "clamp(y - r̂) | 36.10 dB", color="#10b981")
    draw_pill(ax1, 6.9, 8.0, 2.6, 0.8, "Residual Sub", "x̂ = y - r̂", color="#475569")
    draw_pill(ax1, 6.9, 6.6, 2.6, 0.8, "Residual Head", "Conv2d: 32 → 3 (r̂)", color="#6366f1")

    draw_pill(ax1, 0.5, 8.0, 2.6, 0.8, "Encoder 1 (inc)", "DoubleConv [32 ch]", color="#0ea5e9")
    draw_pill(ax1, 0.5, 6.2, 2.6, 0.8, "Encoder 2 (down1)", "DoubleConv [64 ch]", color="#0ea5e9")
    draw_pill(ax1, 0.5, 4.4, 2.6, 0.8, "Encoder 3 (down2)", "DoubleConv [128 ch]", color="#0ea5e9")
    draw_pill(ax1, 0.5, 2.6, 2.6, 0.8, "Encoder 4 (down3)", "DoubleConv [256 ch]", color="#0ea5e9")

    draw_pill(ax1, 3.7, 1.0, 3.0, 0.85, "Bottleneck", "DoubleConv [256 ch]\n(Fixed Static Filters)", color="#f59e0b")

    draw_pill(ax1, 6.9, 2.6, 2.6, 0.8, "Decoder 1 (up1)", "UpConv + DoubleConv [128 ch]", color="#059669")
    draw_pill(ax1, 6.9, 4.0, 2.6, 0.8, "Decoder 2 (up2)", "UpConv + DoubleConv [64 ch]", color="#059669")
    draw_pill(ax1, 6.9, 5.3, 2.6, 0.8, "Decoder 3 (up3)", "UpConv + DoubleConv [32 ch]", color="#059669")

    draw_arrow(ax1, 1.8, 9.4, 1.8, 8.8)
    draw_arrow(ax1, 1.8, 8.0, 1.8, 7.0)
    draw_arrow(ax1, 1.8, 6.2, 1.8, 5.2)
    draw_arrow(ax1, 1.8, 4.4, 1.8, 3.4)
    draw_arrow(ax1, 1.8, 2.6, 3.7, 1.42)
    ax1.text(1.9, 7.45, "Pool ÷2", fontsize=8, color="#64748b")
    ax1.text(1.9, 5.65, "Pool ÷2", fontsize=8, color="#64748b")
    ax1.text(1.9, 3.85, "Pool ÷2", fontsize=8, color="#64748b")

    draw_arrow(ax1, 6.7, 1.42, 8.2, 2.6)
    draw_arrow(ax1, 8.2, 3.4, 8.2, 4.0)
    draw_arrow(ax1, 8.2, 4.8, 8.2, 5.3)
    draw_arrow(ax1, 8.2, 6.1, 8.2, 6.6)
    draw_arrow(ax1, 8.2, 7.4, 8.2, 8.0)
    draw_arrow(ax1, 8.2, 8.8, 8.2, 9.4)

    ax1.annotate("", xy=(6.9, 8.4), xytext=(3.1, 9.8),
                 arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.6, linestyle=":", connectionstyle="arc3,rad=0.22"))

    draw_arrow(ax1, 3.1, 3.0, 6.9, 3.0, color="#94a3b8", ls="--")
    draw_arrow(ax1, 3.1, 4.8, 6.9, 4.4, color="#94a3b8", ls="--")
    draw_arrow(ax1, 3.1, 6.6, 6.9, 5.7, color="#94a3b8", ls="--")
    ax1.text(5.0, 3.15, "Skip (256 ch)", ha="center", fontsize=8, color="#64748b")
    ax1.text(5.0, 4.75, "Skip (128 ch)", ha="center", fontsize=8, color="#64748b")
    ax1.text(5.0, 6.3, "Skip (64 ch)", ha="center", fontsize=8, color="#64748b")

    # ---------------- PANEL 2: PROPOSED MODEL E ----------------
    ax2.set_facecolor("#f8fafc")
    ax2.set_xlim(-0.5, 13.5)
    ax2.set_ylim(-0.5, 12.4)
    ax2.axis("off")

    ax2.text(6.5, 11.9, "PROPOSED MODEL (MODEL E)", ha="center", fontsize=16, fontweight="bold", color="#0f172a")
    ax2.text(6.5, 11.45, "Adaptive Noise-Aware U-Net | 3,241,156 Parameters (+1.43%)\nReal SIDD Benchmark: 37.52 dB PSNR (+1.41 dB) | p = 2.07e-18", ha="center", fontsize=10.5, color="#64748b")

    # Branch 1 diagnosis box
    diag_box = patches.FancyBboxPatch(
        (0.4, 9.35), 8.8, 1.35,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=1.5, zorder=1
    )
    ax2.add_patch(diag_box)
    ax2.text(0.65, 10.4, "Branch 1: Noise Diagnosis (20,481 params)", fontsize=9, fontweight="bold", color="#1e40af", zorder=2)

    draw_pill(ax2, 0.6, 9.5, 2.3, 0.75, "Noisy Input y", "[3 × H × W]", color="#ef4444")
    draw_pill(ax2, 3.4, 9.5, 2.5, 0.75, "Scout CNN", "3→16→16→1 (14.6K)", color="#2563eb", fontsize=8.5)
    draw_pill(ax2, 6.4, 9.5, 2.5, 0.75, "Noise Map M̂", "Spatial [1 × H × W]", color="#7c3aed", fontsize=8.5)

    draw_arrow(ax2, 2.9, 9.87, 3.4, 9.87, color="#2563eb", lw=1.8)
    draw_arrow(ax2, 5.9, 9.87, 6.4, 9.87, color="#7c3aed", lw=1.8)

    draw_pill(ax2, 6.4, 7.8, 2.5, 0.75, "Noise Encoder", "Pool+Dense (5.8K)", color="#9333ea", fontsize=8.5)
    draw_arrow(ax2, 7.65, 9.5, 7.65, 8.55, color="#9333ea", lw=1.8)

    draw_pill(ax2, 4.6, 5.4, 3.4, 0.9, "Fingerprint z", "32-D Latent Vector", color="#db2777", fontsize=9.2)
    draw_arrow(ax2, 7.0, 7.8, 6.5, 6.3, color="#db2777", lw=2.0)

    ax2.text(0.65, 8.85, "Branch 2: Adaptive FiLM U-Net (3.22M params)", fontsize=9, fontweight="bold", color="#047857")

    # Encoder
    draw_pill(ax2, 0.6, 7.8, 2.5, 0.75, "Encoder 1", "DoubleConv [32 ch]", color="#0ea5e9")
    draw_pill(ax2, 0.6, 6.5, 2.5, 0.75, "Encoder 2 Conv", "DoubleConv [64 ch]", color="#0ea5e9")
    draw_pill(ax2, 0.6, 5.4, 2.5, 0.8, "FiLM 1 (γ₁, β₁)", "Modulated by z", color="#06b6d4")
    draw_pill(ax2, 0.6, 4.1, 2.5, 0.75, "Encoder 3", "DoubleConv [128 ch]", color="#0ea5e9")
    draw_pill(ax2, 0.6, 2.8, 2.5, 0.75, "Encoder 4", "DoubleConv [256 ch]", color="#0ea5e9")

    # Bottleneck
    draw_pill(ax2, 4.6, 1.8, 3.4, 0.75, "Bottleneck Conv", "DoubleConv [256 ch]", color="#f59e0b")
    draw_pill(ax2, 4.6, 0.7, 3.4, 0.8, "FiLM 2 (γ₂, β₂)", "Modulated by z", color="#06b6d4")

    # Decoder & Heads
    draw_pill(ax2, 9.8, 2.8, 2.7, 0.75, "Decoder 1", "UpConv [128 ch]", color="#059669")
    draw_pill(ax2, 9.8, 4.1, 2.7, 0.75, "Decoder 2 Conv", "DoubleConv [64 ch]", color="#059669")
    draw_pill(ax2, 9.8, 5.4, 2.7, 0.8, "FiLM 3 (γ₃, β₃)", "Modulated by z", color="#06b6d4")
    draw_pill(ax2, 9.8, 6.7, 2.7, 0.75, "Decoder 3", "UpConv [32 ch]", color="#059669")
    draw_pill(ax2, 9.8, 7.8, 2.7, 0.75, "Residual Head", "Conv2d: 32 → 3 (r̂)", color="#6366f1")
    draw_pill(ax2, 9.8, 8.9, 2.7, 0.75, "Residual Sub", "x̂ = y - r̂", color="#475569")
    draw_pill(ax2, 9.8, 10.0, 2.7, 0.8, "Clean Output x̂", "37.52 dB (+1.41 dB)", color="#10b981")

    # Encoder vertical flow
    draw_arrow(ax2, 1.85, 9.5, 1.85, 8.55)
    draw_arrow(ax2, 1.85, 7.8, 1.85, 7.25)
    draw_arrow(ax2, 1.85, 6.5, 1.85, 6.2)
    draw_arrow(ax2, 1.85, 5.4, 1.85, 4.85)
    draw_arrow(ax2, 1.85, 4.1, 1.85, 3.55)
    draw_arrow(ax2, 1.85, 2.8, 4.6, 2.1)

    draw_arrow(ax2, 6.3, 1.8, 6.3, 1.5)
    draw_arrow(ax2, 8.0, 1.1, 9.8, 2.8)

    draw_arrow(ax2, 11.15, 3.55, 11.15, 4.1)
    draw_arrow(ax2, 11.15, 4.85, 11.15, 5.4)
    draw_arrow(ax2, 11.15, 6.2, 11.15, 6.7)
    draw_arrow(ax2, 11.15, 7.45, 11.15, 7.8)
    draw_arrow(ax2, 11.15, 8.55, 11.15, 8.9)
    draw_arrow(ax2, 11.15, 9.65, 11.15, 10.0)

    # Top-routed y skip
    ax2.plot([1.75, 1.75, 11.15, 11.15], [10.25, 10.95, 10.95, 9.65], color="#ef4444", lw=1.5, ls=":", zorder=2)
    ax2.annotate("", xy=(11.15, 9.65), xytext=(11.15, 9.75),
                arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.5, shrinkA=0, shrinkB=0))

    # Direct FiLM modulation arrows
    draw_arrow(ax2, 4.6, 5.8, 3.1, 5.8, color="#db2777", lw=2.4)
    ax2.text(3.85, 6.05, "z → FiLM 1", color="#db2777", fontsize=8.5, fontweight="bold", ha="center")

    draw_arrow(ax2, 6.3, 5.4, 6.3, 1.5, color="#db2777", lw=2.4)
    ax2.text(6.65, 3.5, "z → FiLM 2", color="#db2777", fontsize=8.5, fontweight="bold", va="center")

    draw_arrow(ax2, 8.0, 5.8, 9.8, 5.8, color="#db2777", lw=2.4)
    ax2.text(8.9, 6.05, "z → FiLM 3", color="#db2777", fontsize=8.5, fontweight="bold", ha="center")

    plt.suptitle("Architecture Flowchart Comparison: Base Model D vs. Proposed Adaptive Model E", fontsize=18, fontweight="bold", y=0.985)
    plt.tight_layout()
    out = "results/figures/architecture_flowchart_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out}")

def main():
    os.makedirs("results/figures", exist_ok=True)
    generate_base_model_diagram()
    generate_proposed_model_diagram()
    generate_comparison_diagram()

if __name__ == "__main__":
    main()

