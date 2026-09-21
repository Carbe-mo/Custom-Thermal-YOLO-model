"""
scripts/generate_architecture_figures.py
=========================================
Draws publication-grade vector architectural diagrams (.pdf and .png) using Matplotlib:
  - Fig 2: Macro System Architecture Flowchart
  - Fig 3: Micro-Architecture Residual Bottleneck Comparison
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "paper_assets" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
})


def draw_micro_bottlenecks():
    """Figure 3: Micro-Architecture Bottleneck Comparison (Standard vs C2f_ECA vs DilatedECABottleneck)"""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.8), sharey=True)

    # Styling colors
    c_blue = "#dbeafe"
    c_blue_b = "#2563eb"
    c_green = "#d1fae5"
    c_green_b = "#059669"
    c_gold = "#fef3c7"
    c_gold_b = "#d97706"
    c_conv = "#f1f5f9"
    c_conv_b = "#475569"

    titles = [
        "(a) Standard Bottleneck\n(Stock YOLOv8 Baseline)",
        "(b) Dense C2f_ECA Bottleneck\n(Proposed P2 & P3 Stages)",
        "(c) Dilated C2f_ECA Bottleneck\n(Proposed P4 & P5 Stages)",
    ]

    for idx, ax in enumerate(axes):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")
        ax.set_title(titles[idx], fontsize=9.5, pad=10)

        # Draw shortcut line
        ax.annotate("", xy=(3.0, 1.2), xytext=(3.0, 10.8),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.5, linestyle="--"))
        ax.text(2.6, 6.0, "Shortcut Addition (Identity)", rotation=90, va="center", ha="center", fontsize=7.5, color="#64748b")

        # Main residual path line
        ax.annotate("", xy=(6.5, 1.5), xytext=(6.5, 10.8),
                    arrowprops=dict(arrowstyle="-", color="#334155", lw=1.5))

        # Input & Output nodes
        ax.text(5.0, 11.2, "Input Tensor $X$", ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(5.0, 0.4, "Output Tensor $Y$", ha="center", va="center", fontsize=8.5, fontweight="bold")

        # Conv 1
        p_cv1 = patches.FancyBboxPatch((4.5, 8.5), 4.0, 1.3, boxstyle="round,pad=0.2",
                                       fc=c_conv, ec=c_conv_b, lw=1.2)
        ax.add_patch(p_cv1)
        ax.text(6.5, 9.15, "Conv $3\\times 3$\n(Dense, $s=1$)", ha="center", va="center", fontsize=8)

        # Conv 2 / Dilated Conv
        if idx < 2:
            p_cv2 = patches.FancyBboxPatch((4.5, 5.8), 4.0, 1.3, boxstyle="round,pad=0.2",
                                           fc=c_blue if idx == 0 else c_green,
                                           ec=c_blue_b if idx == 0 else c_green_b, lw=1.2)
            ax.add_patch(p_cv2)
            ax.text(6.5, 6.45, "Conv $3\\times 3$\n(Dense, $d=1$)", ha="center", va="center", fontsize=8)
        else:
            p_cv2 = patches.FancyBboxPatch((4.5, 5.8), 4.0, 1.3, boxstyle="round,pad=0.2",
                                           fc=c_gold, ec=c_gold_b, lw=1.5)
            ax.add_patch(p_cv2)
            ax.text(6.5, 6.45, "Dilated Conv $3\\times 3$\n($d=2$, RF=$5\\times 5$)", ha="center", va="center", fontsize=8, fontweight="bold")

        # ECA block
        if idx == 0:
            # No ECA
            ax.text(6.5, 3.8, "[No Attention]", ha="center", va="center", fontsize=8, color="#94a3b8", fontstyle="italic")
        else:
            p_eca = patches.FancyBboxPatch((4.5, 3.2), 4.0, 1.3, boxstyle="round,pad=0.2",
                                           fc="#e0e7ff", ec="#4338ca", lw=1.3)
            ax.add_patch(p_eca)
            ax.text(6.5, 3.85, "ECA Module (1D, $k=3$)\n$\\omega = \\sigma(\\text{Conv1D}(y))$", ha="center", va="center", fontsize=7.5, fontweight="bold")

        # Add node (+)
        circ = patches.Circle((5.0, 1.2), 0.45, fc="#ffffff", ec="#1e293b", lw=1.5)
        ax.add_patch(circ)
        ax.text(5.0, 1.2, "+", ha="center", va="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    pdf_out = OUTPUT_DIR / "fig3_micro_bottleneck_architecture.pdf"
    png_out = OUTPUT_DIR / "fig3_micro_bottleneck_architecture.png"
    plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {pdf_out}")


def draw_macro_system_architecture():
    """Figure 2: Complete Macro System Architecture Diagram"""
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def draw_box(x, y, w, h, text, fc, ec, title=""):
        p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=1.2)
        ax.add_patch(p)
        if title:
            ax.text(x + w/2, y + h - 2.5, title, ha="center", va="center", fontsize=7.5, fontweight="bold", color=ec)
            ax.text(x + w/2, y + h/2 - 1.2, text, ha="center", va="center", fontsize=7)
        else:
            ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=7.5)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#475569", lw=1.2))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 1.5, label, ha="center", va="center", fontsize=6.5, color="#64748b")

    # Group backgrounds
    bb_bg = patches.Rectangle((2, 28), 96, 20, fill=True, fc="#f8fafc", ec="#cbd5e1", lw=1.0, linestyle="--")
    ax.add_patch(bb_bg)
    ax.text(5, 46.5, "HYBRID DILATED BACKBONE", fontsize=8, fontweight="bold", color="#334155")

    neck_bg = patches.Rectangle((2, 2), 96, 23, fill=True, fc="#fbfbfe", ec="#e2e8f0", lw=1.0, linestyle="--")
    ax.add_patch(neck_bg)
    ax.text(5, 22.5, "CLEAN LINEAR PANet NECK & DETECTION HEAD", fontsize=8, fontweight="bold", color="#4338ca")

    # Backbone stages
    draw_box(4, 32, 12, 11, "Input 640x640\nStem Convs", "#ffffff", "#475569", "P1/P2")
    draw_arrow(16, 37.5, 20, 37.5)

    draw_box(20, 32, 17, 11, "C2f_ECA (n=1)\nDense 3x3 + ECA", "#d1fae5", "#059669", "P2 (Layer 2)")
    draw_arrow(37, 37.5, 41, 37.5)

    draw_box(41, 32, 17, 11, "C2f_ECA (n=2)\nDense 3x3 + ECA", "#d1fae5", "#059669", "P3 (Layer 4)")
    draw_arrow(58, 37.5, 62, 37.5)

    draw_box(62, 32, 17, 11, "Dilated_ECA (n=2)\nDilated (d=2) + ECA", "#fef3c7", "#d97706", "P4 (Layer 6)")
    draw_arrow(79, 37.5, 83, 37.5)

    draw_box(83, 32, 14, 11, "Dilated_ECA\n+ SPPF (5,9,13)", "#fef3c7", "#d97706", "P5 (Layer 8-9)")

    # Skip connections from backbone to neck
    draw_arrow(50, 32, 50, 15, "P3 Skip")
    draw_arrow(71, 32, 71, 15, "P4 Skip")
    draw_arrow(90, 32, 90, 15, "P5 Top")

    # Neck stages
    draw_box(14, 6, 20, 9, "P3 Detect (Small)\n64ch @ 80x80", "#fee2e2", "#dc2626", "Detect P3")
    draw_box(42, 6, 20, 9, "P4 Detect (Medium)\n128ch @ 40x40", "#fee2e2", "#dc2626", "Detect P4")
    draw_box(70, 6, 20, 9, "P5 Detect (Large)\n256ch @ 20x20", "#fee2e2", "#dc2626", "Detect P5")

    # Connect to final detection
    draw_arrow(50, 15, 24, 15)
    draw_arrow(24, 15, 24, 15)
    draw_arrow(71, 15, 52, 15)
    draw_arrow(90, 15, 80, 15)

    plt.tight_layout()
    pdf_out = OUTPUT_DIR / "fig2_system_architecture.pdf"
    png_out = OUTPUT_DIR / "fig2_system_architecture.png"
    plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {pdf_out}")


if __name__ == "__main__":
    draw_micro_bottlenecks()
    draw_macro_system_architecture()
    print("All architecture diagrams successfully generated in paper_assets/figures/!")
