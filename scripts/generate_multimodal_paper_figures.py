"""
Generate publication-grade vector PDF/PNG figures for the Multimodal RGB+LWIR Fusion IEEE Manuscript:
  1. fig_multimodal_ablation_progression.pdf/.png:
     (a) Multimodal Architecture & Registration Progression (Val & Test mAP@50 and Test mAP@50-95)
     (b) Per-Class Test AP50 Breakdown (YOLOv8n vs v30 Thermal vs v36_main Dual-Stream vs v38 CWAR/MACAF)
  2. fig_multimodal_arch_diagram.pdf/.png:
     Clean block diagram of the Registered Dual-Stream RGB+LWIR + Cross-Modal Attention Gate + 2D-Grid CWAR/MACAF pipeline.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = Path(r"D:\Yolo\Data\project_flir\thermal-yolo\IEEE_Sensor_Thermal")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "legend.fontsize": 8.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def make_progression_and_per_class_fig():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8), gridspec_kw={"width_ratios": [1.22, 1.0]})

    # Panel (a): Pure Architectural Progression from Early Stem Concat -> Single-Modality -> Dual-Stream CMAF -> CWAR/MACAF
    stages = [
        "v34 Early\n(4-ch Stem)",
        "v0 YOLOv8n\n(LWIR Only)",
        "v30 Champion\n(LWIR Dil-ECA)",
        "v38_ca_dil\n(Sym-Dil+Neck)",
        "v36_main\n(Asym CA+ECA)",
        "v36_eca_rgb\n(Asym ECA)",
        "CWAR Dual\n(No Flip)",
        "MACAF Core4\n(Class-Routed)",
        "Proposed CWAR\nTriad (v38)",
    ]
    test_map50 = [83.68, 86.22, 88.15, 88.74, 89.75, 89.86, 91.52, 92.12, 92.33]
    val_map50  = [84.20, 86.25, 87.52, 89.33, 90.53, 89.75, 91.90, 92.68, 92.86]
    test_map95 = [58.67, 62.92, 64.73, 66.52, 66.99, 66.76, 68.75, 69.47, 69.02]

    x = np.arange(len(stages))
    w = 0.36
    bars_test = ax1.bar(x - w/2, test_map50, width=w, color="#1f4e78", edgecolor="black", linewidth=0.6, label="Test mAP@50 (%)")
    bars_val  = ax1.bar(x + w/2, val_map50,  width=w, color="#5b9bd5", edgecolor="black", linewidth=0.6, label="Val mAP@50 (%)")

    ax1_r = ax1.twinx()
    line_95 = ax1_r.plot(x, test_map95, color="#c00000", marker="o", linewidth=2.0, markersize=5.5, label="Test mAP@50–95 (%)")
    ax1_r.set_ylabel("Strict Localization mAP@50–95 (%)", color="#c00000", fontweight="bold")
    ax1_r.tick_params(axis="y", labelcolor="#c00000")
    ax1_r.set_ylim(54, 74)

    ax1.axhline(90.0, color="#2e7d32", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.axhline(92.0, color="#d32f2f", linestyle=":", linewidth=1.1, alpha=0.85)
    ax1.set_ylim(80.0, 95.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, rotation=24, ha="right", fontsize=8.3)
    ax1.set_ylabel("Detection Accuracy mAP@50 (%)", fontweight="bold")
    ax1.set_title("(a) Multimodal Architectural & Pre-NMS Fusion Progression", fontweight="bold", pad=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    for b in bars_test:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 0.22, f"{h:.1f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#102a43")

    # Combined legend on ax1
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    # Panel (b): Per-Class Test AP50 comparison
    classes = ["Person\n(Warm Silhouette)", "Trolley\n(Metallic Frame)", "Bag\n(Carried Luggage)", "Unattended\n(Security Anomaly)", "Overall\nmAP@50"]
    m_v8n  = [76.29, 97.60, 89.20, 81.79, 86.22]
    m_v30  = [77.76, 97.96, 92.66, 84.21, 88.15]
    m_v36m = [82.42, 98.36, 92.10, 86.10, 89.75]
    m_v38c = [87.03, 98.65, 96.74, 86.92, 92.33]

    xc = np.arange(len(classes))
    bw = 0.19
    b1 = ax2.bar(xc - 1.5*bw, m_v8n,  width=bw, color="#b0bec5", edgecolor="black", linewidth=0.6, label="Stock YOLOv8n (Thermal, 86.22%)")
    b2 = ax2.bar(xc - 0.5*bw, m_v30,  width=bw, color="#ffb74d", edgecolor="black", linewidth=0.6, label="v30 Hybrid Dilated ECA (Thermal, 88.15%)")
    b3 = ax2.bar(xc + 0.5*bw, m_v36m, width=bw, color="#4fc3f7", edgecolor="black", linewidth=0.6, label="v36_main Dual-Stream (RGB+LWIR, 89.75%)")
    b4 = ax2.bar(xc + 1.5*bw, m_v38c, width=bw, color="#1b5e20", edgecolor="black", linewidth=0.6, label="Proposed v38 CWAR Triad (RGB+LWIR, 92.33%)")

    for b in b4:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h + 0.35, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1b5e20")

    ax2.set_ylim(70.0, 102.5)
    ax2.set_xticks(xc)
    ax2.set_xticklabels(classes, fontsize=8.8)
    ax2.set_ylabel("Per-Class Test AP@50 (%)", fontweight="bold")
    ax2.set_title("(b) Per-Class Test Split AP@50 Breakdown", fontweight="bold", pad=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.0)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_multimodal_ablation_progression.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_multimodal_ablation_progression.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_multimodal_ablation_progression.pdf/.png")


def make_architecture_diagram():
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def draw_box(x, y, w, h, text, fc="#e3f2fd", ec="#0d47a1", fontsize=8.8, bold=True):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                      facecolor=fc, edgecolor=ec, linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if bold else "normal", color="#0a192f")

    def arrow(x1, y1, x2, y2, color="#263238"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.28,head_length=0.45", color=color, lw=1.4))

    # Title banner
    ax.text(50, 40.2, "Proposed Asymmetric Dual-Stream RGB–LWIR Architecture with Cross-Modal Attention Gating & 2D-Grid CWAR/MACAF Fusion",
            ha="center", va="center", fontsize=11.2, fontweight="bold", color="#0d47a1")

    # Input Stage
    draw_box(1.5, 25.5, 13.5, 10.5, "Visible RGB Stream\n(Optical Sensor)\n$I_{rgb} \\in \\mathbb{R}^{3\\times 640\\times 640}$\nHigh-Freq Texture\n(Straps, Zippers, Fabric)", fc="#e8f5e9", ec="#1b5e20", fontsize=8.2)
    draw_box(1.5, 5.5,  13.5, 10.5, "LWIR Thermal Stream\n($7.5\\text{--}13.5\\,\\mu$m Radiometric)\n$I_{thm} \\in \\mathbb{R}^{3\\times 640\\times 640}$\nIllumination-Invariant\nBody & Trolley Heat", fc="#fff3e0", ec="#e65100", fontsize=8.2)

    # Dual Asymmetric Backbones
    draw_box(18.5, 25.5, 18.0, 10.5, "RGB Backbone (Dense Texture)\n• Stem: Conv $3\\times3$ ($P_1/2, P_2/4$)\n• $P_2, P_3$: C2f_CA_ECA / C2f_ECA\n• $P_4, P_5$: Dense ($d=1$) or Aligned ($d=2$)\n• SPPF ($k=5$)", fc="#c8e6c9", ec="#1b5e20", fontsize=8.1)
    draw_box(18.5, 5.5,  18.0, 10.5, "Thermal Backbone (Dilated Context)\n• Stem: Conv $3\\times3$ ($P_1/2, P_2/4$)\n• $P_2, P_3$: Dense C2f_ECA ($d=1$)\n• $P_4, P_5$: C2f_ECA_Dilated ($d=2, D_{eff}=5$)\n• SPPF ($k=5$)", fc="#ffe0b2", ec="#e65100", fontsize=8.1)

    arrow(15.0, 30.7, 18.5, 30.7, "#1b5e20")
    arrow(15.0, 10.7, 18.5, 10.7, "#e65100")

    # Multi-scale CMAF Gates
    draw_box(40.5, 28.5, 16.5, 7.5, "CMAF Gate @ $P_3$ ($80\\times80$)\n$\\alpha_3 = \\sigma(\\mathrm{Conv}([F_{rgb}^3, F_{thm}^3]))$\n$F_{fuse}^3 = \\alpha_3 F_{rgb}^3 + (1-\\alpha_3)F_{thm}^3 + F_{red}^3$", fc="#e1f5fe", ec="#0277bd", fontsize=7.8)
    draw_box(40.5, 17.0, 16.5, 7.5, "CMAF Gate @ $P_4$ ($40\\times40$)\n$\\alpha_4 = \\sigma(\\mathrm{Conv}([F_{rgb}^4, F_{thm}^4]))$\n$F_{fuse}^4 = \\alpha_4 F_{rgb}^4 + (1-\\alpha_4)F_{thm}^4 + F_{red}^4$", fc="#e1f5fe", ec="#0277bd", fontsize=7.8)
    draw_box(40.5, 5.5,  16.5, 7.5, "CMAF Gate @ $P_5$ ($20\\times20$)\n$\\alpha_5 = \\sigma(\\mathrm{Conv}([F_{rgb}^5, F_{thm}^5]))$\n$F_{fuse}^5 = \\alpha_5 F_{rgb}^5 + (1-\\alpha_5)F_{thm}^5 + F_{red}^5$", fc="#e1f5fe", ec="#0277bd", fontsize=7.8)

    for y_dst in (32.2, 20.7, 9.2):
        arrow(36.5, 30.7, 40.5, y_dst, "#1b5e20")
        arrow(36.5, 10.7, 40.5, y_dst, "#e65100")

    # ECA-PANet Neck
    draw_box(60.5, 10.5, 15.5, 20.5, "ECA-Enhanced PANet Neck\n(PANetNeckECA)\n\n• Top-Down Path:\n  Upsample + Concat\n  C2f_ECA ($P_5 \\to P_4 \\to P_3$)\n\n• Bottom-Up Path:\n  Conv $3\\times3, s=2$ + Concat\n  C2f_ECA ($P_3 \\to P_4 \\to P_5$)\n\n• 100% COCO Weight Transfer", fc="#ede7f6", ec="#4527a0", fontsize=8.0)

    for y_src in (32.2, 20.7, 9.2):
        arrow(57.0, y_src, 60.5, 20.7, "#0277bd")

    # 2D-Grid CWAR / MACAF Head
    draw_box(79.5, 8.5, 19.0, 24.5, "Decoupled Detect Head &\n2D-Grid Anchor-Aligned Fusion\n(CWAR / MACAF @ 8,400 Anchors)\n\n• 1-to-1 Spatial Grid Alignment:\n  $(80^2 + 40^2 + 20^2 = 8,400)$\n• Confidence-Weighted Box:\n  $\\hat{b}_i = \\frac{\\sum_m b_m (\\max_c w_{m,c} p_{m,i,c})^\\gamma b_{m,i}}{\\sum_m b_m (\\max_c w_{m,c} p_{m,i,c})^\\gamma}$\n• Class-Aware Modality Routing:\n  $\\hat{p}_i = \\sum_m w_m \\odot p_{m,i}$\n\nResult: 92.33% Test / 92.89% Val mAP@50", fc="#fce4ec", ec="#880e4f", fontsize=7.9)

    arrow(76.0, 20.7, 79.5, 20.7, "#4527a0")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig_multimodal_arch_diagram.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_multimodal_arch_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_multimodal_arch_diagram.pdf/.png")


if __name__ == "__main__":
    make_progression_and_per_class_fig()
    make_architecture_diagram()
