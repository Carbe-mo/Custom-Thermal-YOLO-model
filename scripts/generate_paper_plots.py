"""
scripts/generate_paper_plots.py
================================
Generates high-resolution IEEE-style vector plots (.pdf and .png) for the manuscript:
  1. Training dynamics comparison over 100 epochs (Losses & mAP curves)
  2. Per-class AP50 and AP50-95 performance comparison
"""

from __future__ import annotations

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "paper_assets" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set IEEE Journal styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "lines.linewidth": 1.8,
    "lines.markersize": 4,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# Color palette: IEEE friendly (Navy, Crimson, Forest Green, Amber)
COLOR_BASELINE = "#2b5c8f"    # Deep Navy
COLOR_PROPOSED = "#d9381e"    # Crimson Red
COLOR_ECA = "#1b8a5a"         # Forest Green
COLOR_FILL = "#f8f9fa"


def plot_training_convergence():
    v30_path = PROJECT_ROOT / "runs" / "v30_v27_cosine_100ep" / "train" / "results.csv"
    v0b_path = PROJECT_ROOT / "runs" / "v0b_baseline_100ep" / "train" / "results.csv"

    if not v30_path.exists() or not v0b_path.exists():
        print(f"Missing run results.csv files at {v30_path} or {v0b_path}")
        return

    df30 = pd.read_csv(v30_path)
    df30.columns = [c.strip() for c in df30.columns]

    df0b = pd.read_csv(v0b_path)
    df0b.columns = [c.strip() for c in df0b.columns]

    epochs = df30["epoch"].values

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)

    # 1. Box Loss
    ax = axes[0, 0]
    ax.plot(epochs, df0b["train/box_loss"], label="Stock Baseline (v0b)", color=COLOR_BASELINE, linestyle="--")
    ax.plot(epochs, df30["train/box_loss"], label="Proposed Model (v30)", color=COLOR_PROPOSED)
    ax.set_ylabel("Train Box Loss")
    ax.set_title("(a) Bounding Box Loss")
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    # 2. Class Loss
    ax = axes[0, 1]
    ax.plot(epochs, df0b["train/cls_loss"], label="Stock Baseline (v0b)", color=COLOR_BASELINE, linestyle="--")
    ax.plot(epochs, df30["train/cls_loss"], label="Proposed Model (v30)", color=COLOR_PROPOSED)
    ax.set_ylabel("Train Class Loss")
    ax.set_title("(b) Classification Loss")
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    # 3. Validation mAP@50
    ax = axes[1, 0]
    ax.plot(epochs, df0b["metrics/mAP50(B)"] * 100, label="Stock Baseline (v0b)", color=COLOR_BASELINE, linestyle="--")
    ax.plot(epochs, df30["metrics/mAP50(B)"] * 100, label="Proposed Model (v30)", color=COLOR_PROPOSED)
    # Highlight final 20 epochs where mosaic was closed
    ax.axvspan(80, 100, color="#fef3c7", alpha=0.5, label="Mosaic Closed (20ep)")
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Validation mAP@50 (%)")
    ax.set_title("(c) Validation mAP@50 Convergence")
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right")

    # 4. Validation mAP@50-95
    ax = axes[1, 1]
    ax.plot(epochs, df0b["metrics/mAP50-95(B)"] * 100, label="Stock Baseline (v0b)", color=COLOR_BASELINE, linestyle="--")
    ax.plot(epochs, df30["metrics/mAP50-95(B)"] * 100, label="Proposed Model (v30)", color=COLOR_PROPOSED)
    ax.axvspan(80, 100, color="#fef3c7", alpha=0.5, label="Mosaic Closed (20ep)")
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Validation mAP@50-95 (%)")
    ax.set_title("(d) Validation mAP@50-95 Convergence")
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right")

    plt.tight_layout()
    pdf_out = OUTPUT_DIR / "fig4_training_loss_convergence.pdf"
    png_out = OUTPUT_DIR / "fig4_training_loss_convergence.png"
    plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {pdf_out}")


def plot_per_class_comparison():
    # Per-class AP50 on official held-out test split
    classes = ["Person", "Trolley", "Bag", "Unattended"]
    
    # Values extracted directly from test split evaluations
    ap50_v0b = [72.6, 97.4, 89.2, 82.8]
    ap50_v30 = [76.9, 97.9, 92.7, 84.1]
    
    ap95_v0b = [46.6, 81.0, 66.7, 55.8]
    ap95_v30 = [51.1, 82.5, 66.0, 59.3]

    x = np.arange(len(classes))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)

    # Subplot 1: mAP@50
    rects1 = ax1.bar(x - width/2, ap50_v0b, width, label="Stock Baseline (v0b)", color=COLOR_BASELINE, alpha=0.85, edgecolor="black", linewidth=0.8)
    rects2 = ax1.bar(x + width/2, ap50_v30, width, label="Proposed Model (v30)", color=COLOR_PROPOSED, alpha=0.9, edgecolor="black", linewidth=0.8)

    ax1.set_ylabel("mAP@50 (%)")
    ax1.set_title("(a) Detection Accuracy (mAP@50)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_ylim(60, 105)
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper left", framealpha=0.9)

    # Add delta labels on top of bars
    for i in range(len(classes)):
        diff = ap50_v30[i] - ap50_v0b[i]
        sign = "+" if diff >= 0 else ""
        ax1.annotate(f"{sign}{diff:.1f}%",
                     xy=(x[i] + width/2, ap50_v30[i]),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, fontweight="bold", color=COLOR_PROPOSED)

    # Subplot 2: mAP@50-95
    rects3 = ax2.bar(x - width/2, ap95_v0b, width, label="Stock Baseline (v0b)", color=COLOR_BASELINE, alpha=0.85, edgecolor="black", linewidth=0.8)
    rects4 = ax2.bar(x + width/2, ap95_v30, width, label="Proposed Model (v30)", color=COLOR_PROPOSED, alpha=0.9, edgecolor="black", linewidth=0.8)

    ax2.set_ylabel("mAP@50-95 (%)")
    ax2.set_title("(b) Localization Precision (mAP@50-95)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.set_ylim(35, 95)
    ax2.grid(True, axis="y")
    ax2.legend(loc="upper left", framealpha=0.9)

    for i in range(len(classes)):
        diff = ap95_v30[i] - ap95_v0b[i]
        sign = "+" if diff >= 0 else ""
        ax2.annotate(f"{sign}{diff:.1f}%",
                     xy=(x[i] + width/2, ap95_v30[i]),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, fontweight="bold", color=COLOR_PROPOSED)

    plt.tight_layout()
    pdf_out = OUTPUT_DIR / "fig5_per_class_comparison.pdf"
    png_out = OUTPUT_DIR / "fig5_per_class_comparison.png"
    plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {pdf_out}")


def plot_receptive_field_motivation():
    """Figure 1: Receptive field and dilation motivation diagram."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))

    # Grid 1: Standard 3x3 Conv
    ax = axes[0]
    grid1 = np.zeros((7, 7))
    grid1[2:5, 2:5] = 1.0  # Dense 3x3
    ax.imshow(grid1, cmap="Blues", vmin=0, vmax=1.2)
    ax.set_title("Standard Conv (d=1)\nRF = 3x3, Weights = 9", fontsize=9.5)
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.grid(color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Label "Tunnel Vision"
    ax.set_xlabel("Dense Local Scope\n(Tunnel Vision on Heat)", color="#2b5c8f", fontsize=8.5)

    # Grid 2: Dilated 3x3 Conv (d=2)
    ax = axes[1]
    grid2 = np.zeros((7, 7))
    for r in [1, 3, 5]:
        for c in [1, 3, 5]:
            grid2[r, c] = 1.0  # Dilated 3x3 with d=2
    ax.imshow(grid2, cmap="Oranges", vmin=0, vmax=1.2)
    ax.set_title("Dilated Conv (d=2)\nRF = 5x5, Weights = 9", fontsize=9.5)
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.grid(color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("Expanded Context\n(Captures Radiant Blooms)", color="#d9381e", fontsize=8.5)

    # Grid 3: Proposed Hybrid Stage Mapping
    ax = axes[2]
    ax.axis("off")
    stages = ["P1/P2 (Stem)", "P3 (Small Obj)", "P4 (Medium Obj)", "P5 (Large Obj)"]
    types = ["Dense (d=1)", "Dense (d=1)", "Dilated (d=2)", "Dilated (d=2)"]
    colors = ["#94a3b8", "#1b8a5a", "#d9381e", "#d9381e"]

    for i in range(4):
        ax.barh(3 - i, 1, color=colors[i], height=0.6, edgecolor="black", linewidth=0.8)
        ax.text(0.05, 3 - i, f"{stages[i]}", va="center", ha="left", color="white", fontweight="bold", fontsize=8)
        ax.text(0.95, 3 - i, f"{types[i]}", va="center", ha="right", color="white", fontweight="bold", fontsize=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 3.5)
    ax.set_title("Proposed Stage-Selective\nHybrid Allocation", fontsize=9.5)

    plt.tight_layout()
    pdf_out = OUTPUT_DIR / "fig1_receptive_field_motivation.pdf"
    png_out = OUTPUT_DIR / "fig1_receptive_field_motivation.png"
    plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {pdf_out}")


if __name__ == "__main__":
    plot_training_convergence()
    plot_per_class_comparison()
    plot_receptive_field_motivation()
    print("All vector figures successfully generated in paper_assets/figures/!")
