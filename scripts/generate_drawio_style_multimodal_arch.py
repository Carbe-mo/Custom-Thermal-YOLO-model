"""
Generate a draw.io-style 5-panel architecture diagram (matching Dilated_ECA_Yolo.pdf from the
thermal-only IEEE paper) for the Multimodal RGB-LWIR + CWAR/MACAF manuscript:
  Panel (a): Baseline Early-Fusion / Single-Stream YOLO (with red X markers)
  Panel (b): Proposed Dual-Stream Asymmetric RGB-LWIR + CMAF + PANetNeckECA Macro-Architecture (with green checkmarks)
  Panel (c): RGB C2f_CA_ECA & Thermal C2f_ECA_Dilated (d=2) Bottleneck Micro-Architecture
  Panel (d): Multi-Scale Cross-Modal Attention Fusion (CMAF) Gate Micro-Architecture
  Panel (e): 2D-Grid Anchor-Aligned CWAR / MACAF Fusion Pipeline (8,400 Spatial Anchors)
"""
from __future__ import annotations
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = Path(r"D:\Yolo\Data\project_flir\thermal-yolo\IEEE_Sensor_Thermal")
ART_DIR = Path(r"C:\Users\satya\.gemini\antigravity\brain\8aed0bbc-dce3-4a5a-b2c4-58c864f9a829")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
})


def rbox(ax, x, y, w, h, text, fc="#dae8fc", ec="#000000", lw=1.1, fontsize=7.5, bold=False, rad=0.45):
    p = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.08,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold" if bold else "normal", color="#000000",
        linespacing=1.12, zorder=4
    )


def arrow(ax, x1, y1, x2, y2, ls="-", lw=1.15, color="#000000"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.34", color=color, lw=lw, linestyle=ls),
        zorder=5
    )


def polyline_arrow(ax, pts, ls="-", lw=1.15, color="#000000"):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], linestyle=ls, linewidth=lw, color=color, zorder=4)
    arrow(ax, xs[-2], ys[-2], xs[-1], ys[-1], ls=ls, lw=lw, color=color)


def red_x(ax, x, y, s=0.38):
    """Draw a crisp vector red X with black outline like draw.io."""
    ax.plot([x - s, x + s], [y - s, y + s], color="#000000", linewidth=3.4, solid_capstyle="round", zorder=6)
    ax.plot([x - s, x + s], [y + s, y - s], color="#000000", linewidth=3.4, solid_capstyle="round", zorder=6)
    ax.plot([x - s, x + s], [y - s, y + s], color="#ff1744", linewidth=2.0, solid_capstyle="round", zorder=7)
    ax.plot([x - s, x + s], [y + s, y - s], color="#ff1744", linewidth=2.0, solid_capstyle="round", zorder=7)


def green_check(ax, x, y, s=0.45):
    """Draw a crisp vector green checkmark with black outline like draw.io."""
    xs = [x - 0.85 * s, x - 0.15 * s, x + 1.05 * s]
    ys = [y - 0.05 * s, y - 0.75 * s, y + 0.75 * s]
    ax.plot(xs, ys, color="#000000", linewidth=3.6, solid_capstyle="round", solid_joinstyle="round", zorder=6)
    ax.plot(xs, ys, color="#00c853", linewidth=2.1, solid_capstyle="round", solid_joinstyle="round", zorder=7)


def op_circle(ax, x, y, r=0.58, sym="+"):
    c = patches.Circle((x, y), r, facecolor="#ffffff", edgecolor="#000000", linewidth=1.15, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, sym, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#000000", zorder=6)


def build_diagram():
    fig, ax = plt.subplots(figsize=(18.0, 5.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30.5)
    ax.axis("off")

    # =========================================================================
    # Outer 5 Panels matching Dilated_ECA_Yolo_rendered.png layout
    # =========================================================================
    # Panel (a): x=[0.2, 17.8], y=[0.3, 30.2]
    ax.add_patch(patches.Rectangle((0.2, 0.3), 17.6, 29.9, facecolor="#fdf2d0", edgecolor="#000000", linewidth=1.5, zorder=1))
    ax.text(0.7, 29.5, "(a)", color="#ff0000", fontsize=15, fontweight="bold", ha="left", va="top", zorder=6)

    # Panel (b): x=[18.0, 64.8], y=[0.3, 30.2]
    ax.add_patch(patches.Rectangle((18.0, 0.3), 46.8, 29.9, facecolor="#ffffff", edgecolor="#000000", linewidth=1.5, zorder=1))
    ax.text(18.4, 29.5, "(b)", color="#ff0000", fontsize=15, fontweight="bold", ha="left", va="top", zorder=6)

    # Panel (c): x=[65.0, 82.3], y=[10.2, 30.2]
    ax.add_patch(patches.Rectangle((65.0, 10.2), 17.3, 20.0, facecolor="#c5b4d9", edgecolor="#000000", linewidth=1.5, zorder=1))
    ax.text(81.8, 29.5, "(c)", color="#ff0000", fontsize=15, fontweight="bold", ha="right", va="top", zorder=6)

    # Panel (d): x=[82.5, 99.8], y=[10.2, 30.2]
    ax.add_patch(patches.Rectangle((82.5, 10.2), 17.3, 20.0, facecolor="#d9d2e9", edgecolor="#000000", linewidth=1.5, zorder=1))
    ax.text(99.3, 29.5, "(d)", color="#ff0000", fontsize=15, fontweight="bold", ha="right", va="top", zorder=6)

    # Panel (e): x=[65.0, 99.8], y=[0.3, 10.0]
    ax.add_patch(patches.Rectangle((65.0, 0.3), 34.8, 9.7, facecolor="#fce5cd", edgecolor="#000000", linewidth=1.5, zorder=1))
    ax.text(83.8, 0.8, "(e)", color="#ff0000", fontsize=15, fontweight="bold", ha="center", va="bottom", zorder=6)

    # =========================================================================
    # PANEL (a): Baseline Early-Fusion / Single-Stream YOLO (What Fails)
    # =========================================================================
    ax.text(9.5, 29.1, "Baseline Early-Stream Concat", ha="center", va="top", fontsize=8.3, fontweight="bold")
    rbox(ax, 1.0, 25.0, 7.4, 2.5, "Visible RGB\n640×640×3\n(Optical)", fc="#ffffff", fontsize=6.9)
    rbox(ax, 9.6, 25.0, 7.4, 2.5, "LWIR Thermal\n640×640×3\n(Radiometric)", fc="#ffffff", fontsize=6.9)

    rbox(ax, 2.2, 21.3, 13.6, 2.4, "Early Stem Concat (4-ch / 6-ch)\n3×3, s=2 (Mixed Modalities)", fc="#dae8fc", fontsize=7.0)
    red_x(ax, 15.1, 23.2)
    arrow(ax, 4.7, 25.0, 6.5, 23.7)
    arrow(ax, 13.3, 25.0, 11.5, 23.7)

    # 2 columns inside Panel (a) like original diagram
    rbox(ax, 0.9, 16.5, 7.4, 3.3, "Conv 128 ch\n3×3, s=2, P2/4", fc="#dae8fc", fontsize=6.9)
    rbox(ax, 0.9, 11.8, 7.4, 3.3, "C2f (n=1)\n128 ch (No Att.)", fc="#b0e3e6", fontsize=6.9)
    red_x(ax, 7.7, 14.5)
    rbox(ax, 0.9, 7.1, 7.4, 3.3, "Conv 256 ch\n3×3, s=2, P3/8", fc="#dae8fc", fontsize=6.9)
    rbox(ax, 0.9, 2.2, 7.4, 3.3, "C2f (n=2)\n256 ch (d=1)", fc="#b0e3e6", fontsize=6.9)
    red_x(ax, 7.7, 4.9)

    rbox(ax, 9.7, 16.5, 7.4, 3.3, "Conv 512 ch\n3×3, s=2, P4/16", fc="#dae8fc", fontsize=6.9)
    rbox(ax, 9.7, 11.8, 7.4, 3.3, "C2f (n=2)\n512 ch (d=1)", fc="#b0e3e6", fontsize=6.9)
    red_x(ax, 16.5, 14.5)
    rbox(ax, 9.7, 7.1, 7.4, 3.3, "Conv 1024 ch\n3×3, s=2, P5/32", fc="#dae8fc", fontsize=6.9)
    rbox(ax, 9.7, 2.2, 7.4, 3.3, "SPPF (k=5)\n1024 ch\n(83.7% mAP50)", fc="#f8cecc", fontsize=6.8)

    arrow(ax, 5.5, 21.3, 4.6, 19.8)
    arrow(ax, 4.6, 16.5, 4.6, 15.1)
    arrow(ax, 4.6, 11.8, 4.6, 10.4)
    arrow(ax, 4.6, 7.1, 4.6, 5.5)
    polyline_arrow(ax, [(8.3, 3.8), (9.0, 3.8), (9.0, 18.2), (9.7, 18.2)])
    arrow(ax, 13.4, 16.5, 13.4, 15.1)
    arrow(ax, 13.4, 11.8, 13.4, 10.4)
    arrow(ax, 13.4, 7.1, 13.4, 5.5)

    # =========================================================================
    # PANEL (b): Proposed Dual-Stream RGB-LWIR + CMAF + PANetNeckECA
    # =========================================================================
    ax.add_patch(patches.FancyBboxPatch((19.5, 1.2), 16.0, 25.0, boxstyle="round,pad=0.2,rounding_size=0.6",
                                        facecolor="#fff8e1", edgecolor="#e6ccb2", linewidth=0.9, zorder=2))

    # Top input boxes (shifted right of (b) label)
    rbox(ax, 19.8, 26.8, 7.3, 2.2, "Visible RGB I_rgb\n640×640×3 (Optical)", fc="#d5e8d4", fontsize=6.7, bold=True)
    rbox(ax, 27.8, 26.8, 7.3, 2.2, "LWIR Thermal I_thm\n640×640×3 (7.5–13.5µm)", fc="#ffe6cc", fontsize=6.7, bold=True)

    # Stream 1: RGB Backbone (x=19.8, w=7.2)
    rbox(ax, 19.8, 22.6, 7.2, 3.0, "RGB Stem Conv\n64→128ch, P2/4", fc="#dae8fc", fontsize=6.7)
    green_check(ax, 26.5, 25.1, 0.36)
    rbox(ax, 19.8, 18.4, 7.2, 3.0, "C2f_CA_ECA\n128ch (P2) + 256ch (P3)", fc="#b49bc8", fontsize=6.5)
    green_check(ax, 26.5, 20.9, 0.38)
    rbox(ax, 19.8, 14.2, 7.2, 3.0, "Conv 512 ch\n3×3, s=2, P4/16", fc="#dae8fc", fontsize=6.7)
    rbox(ax, 19.8, 10.0, 7.2, 3.0, "C2f_CA_ECA / Dil\n512ch (P4, d=1/2)", fc="#d5b8d8", fontsize=6.5)
    green_check(ax, 26.5, 12.5, 0.38)
    rbox(ax, 19.8, 5.8,  7.2, 3.0, "C2f_CA_ECA / Dil\n1024ch (P5, d=1/2)", fc="#d5b8d8", fontsize=6.5)
    green_check(ax, 26.5, 8.3, 0.38)
    rbox(ax, 19.8, 1.6,  7.2, 2.9, "SPPF (k=5)\n1024 ch (RGB P5)", fc="#f8cecc", fontsize=6.7)

    arrow(ax, 23.4, 26.8, 23.4, 25.6)
    arrow(ax, 23.4, 22.6, 23.4, 21.4)
    arrow(ax, 23.4, 18.4, 23.4, 17.2)
    arrow(ax, 23.4, 14.2, 23.4, 13.0)
    arrow(ax, 23.4, 10.0, 23.4, 8.8)
    arrow(ax, 23.4, 5.8,  23.4, 4.5)

    # Stream 2: Thermal Backbone (x=27.8, w=7.2)
    rbox(ax, 27.8, 22.6, 7.2, 3.0, "LWIR Stem Conv\n64→128ch, P2/4", fc="#dae8fc", fontsize=6.7)
    rbox(ax, 27.8, 18.4, 7.2, 3.0, "C2f_ECA (Dense)\n128ch (P2) + 256ch (P3)", fc="#b49bc8", fontsize=6.5)
    green_check(ax, 34.5, 20.9, 0.38)
    rbox(ax, 27.8, 14.2, 7.2, 3.0, "Conv 512 ch\n3×3, s=2, P4/16", fc="#dae8fc", fontsize=6.7)
    rbox(ax, 27.8, 10.0, 7.2, 3.0, "C2f_ECA_Dilated\n512ch (P4, d=2)", fc="#e1d5e7", fontsize=6.5)
    green_check(ax, 34.5, 12.5, 0.38)
    rbox(ax, 27.8, 5.8,  7.2, 3.0, "C2f_ECA_Dilated\n1024ch (P5, d=2)", fc="#e1d5e7", fontsize=6.5)
    green_check(ax, 34.5, 8.3, 0.38)
    rbox(ax, 27.8, 1.6,  7.2, 2.9, "SPPF (k=5)\n1024 ch (LWIR P5)", fc="#f8cecc", fontsize=6.7)

    arrow(ax, 31.4, 26.8, 31.4, 25.6)
    arrow(ax, 31.4, 22.6, 31.4, 21.4)
    arrow(ax, 31.4, 18.4, 31.4, 17.2)
    arrow(ax, 31.4, 14.2, 31.4, 13.0)
    arrow(ax, 31.4, 10.0, 31.4, 8.8)
    arrow(ax, 31.4, 5.8,  31.4, 4.5)

    # Column 3 in Panel (b): Multi-Scale CMAF Gates + Top-Down PANetNeckECA (x=36.5, w=7.0)
    ax.text(50.2, 28.8, "Cross-Modal Gates, ECA-PANet Neck & CWAR Head", ha="center", va="top", fontsize=9.0, fontweight="bold")

    rbox(ax, 36.5, 22.4, 7.0, 3.4, "CMAF Gate @ P5\n1024ch (20×20)", fc="#fff2cc", fontsize=6.7, bold=True)
    green_check(ax, 43.0, 25.3, 0.36)
    rbox(ax, 36.5, 18.0, 7.0, 2.9, "Upsample 2×\n1024 ch", fc="#fff2cc", fontsize=6.7)
    rbox(ax, 36.5, 14.3, 7.0, 2.4, "Concat + CMAF P4\n512 ch (40×40)", fc="#d5e8d4", fontsize=6.6, bold=True)
    rbox(ax, 36.5, 9.9,  7.0, 3.1, "C2f_ECA (Neck)\n512 ch (td_p4)", fc="#b0e3e6", fontsize=6.7)
    green_check(ax, 43.0, 12.5, 0.36)
    rbox(ax, 36.5, 6.0,  7.0, 2.7, "Upsample 2×\n512 ch", fc="#fff2cc", fontsize=6.7)
    rbox(ax, 36.5, 1.6,  7.0, 3.2, "CMAF P3 + C2f_ECA\n256 ch (td_p3)", fc="#b0e3e6", fontsize=6.6, bold=True)
    green_check(ax, 43.0, 4.3, 0.36)

    arrow(ax, 40.0, 22.4, 40.0, 20.9)
    arrow(ax, 40.0, 18.0, 40.0, 16.7)
    arrow(ax, 40.0, 14.3, 40.0, 13.0)
    arrow(ax, 40.0, 9.9,  40.0, 8.7)
    arrow(ax, 40.0, 6.0,  40.0, 4.8)

    # Clean orthogonal skip connections through horizontal corridors (no text crossing!)
    polyline_arrow(ax, [(35.0, 3.0), (35.7, 3.0), (35.7, 24.1), (36.5, 24.1)], ls="-", lw=1.0)
    polyline_arrow(ax, [(35.0, 11.5), (35.9, 11.5), (35.9, 15.5), (36.5, 15.5)], ls=":", lw=1.0)
    polyline_arrow(ax, [(27.0, 11.5), (27.4, 11.5), (27.4, 9.3), (35.9, 9.3), (35.9, 15.5), (36.5, 15.5)], ls=":", lw=1.0)
    polyline_arrow(ax, [(27.0, 19.9), (27.4, 19.9), (27.4, 17.7), (35.9, 17.7), (35.9, 3.2), (36.5, 3.2)], ls=":", lw=1.0)

    # Column 4 in Panel (b): Bottom-Up PANetNeckECA (x=45.0, w=7.0)
    rbox(ax, 45.0, 22.4, 7.0, 3.4, "Conv 256 ch\n3×3, s=2", fc="#dae8fc", fontsize=6.8)
    rbox(ax, 45.0, 18.4, 7.0, 2.5, "Concat (P4)", fc="#d5e8d4", fontsize=6.8)
    rbox(ax, 45.0, 13.8, 7.0, 3.3, "C2f_ECA (Neck)\n512 ch (bu_p4)", fc="#b0e3e6", fontsize=6.7)
    green_check(ax, 51.5, 16.6, 0.36)
    rbox(ax, 45.0, 9.3,  7.0, 3.2, "Conv 512 ch\n3×3, s=2", fc="#dae8fc", fontsize=6.8)
    rbox(ax, 45.0, 5.8,  7.0, 2.3, "Concat (P5)", fc="#d5e8d4", fontsize=6.8)
    rbox(ax, 45.0, 1.6,  7.0, 3.1, "C2f_ECA (Neck)\n1024 ch (bu_p5)", fc="#b0e3e6", fontsize=6.7)
    green_check(ax, 51.5, 4.3, 0.36)

    polyline_arrow(ax, [(43.5, 3.2), (44.2, 3.2), (44.2, 24.1), (45.0, 24.1)])
    arrow(ax, 48.5, 22.4, 48.5, 20.9)
    arrow(ax, 48.5, 18.4, 48.5, 17.1)
    arrow(ax, 48.5, 13.8, 48.5, 12.5)
    arrow(ax, 48.5, 9.3,  48.5, 8.1)
    arrow(ax, 48.5, 5.8,  48.5, 4.7)
    polyline_arrow(ax, [(43.5, 11.4), (44.4, 11.4), (44.4, 19.6), (45.0, 19.6)], ls=":")
    polyline_arrow(ax, [(43.5, 24.1), (44.6, 24.1), (44.6, 6.9), (45.0, 6.9)], ls=":")

    # Column 5 in Panel (b): Detect Heads + 2D-Grid CWAR Output (x=53.8)
    rbox(ax, 53.8, 18.5, 4.4, 6.8, "Detect\nP4\nMedium\n(40×40)", fc="#d5e8d4", fontsize=6.8)
    rbox(ax, 53.8, 10.1, 4.4, 6.8, "Detect\nP5\nLarge\n(20×20)", fc="#d5e8d4", fontsize=6.8)
    rbox(ax, 53.8, 1.6,  4.4, 6.8, "Detect\nP3\nSmall\n(80×80)", fc="#d5e8d4", fontsize=6.8)

    rbox(ax, 59.4, 4.8, 4.8, 20.0, "2D-Grid\nCWAR &\nMACAF\nOutput\n(8,400\nAnchors)\n\n92.33%\nmAP@50", fc="#ffffff", fontsize=7.0, bold=True)

    polyline_arrow(ax, [(52.0, 15.4), (53.0, 15.4), (53.0, 21.9), (53.8, 21.9)], ls="--")
    polyline_arrow(ax, [(52.0, 3.1),  (53.2, 3.1),  (53.2, 13.5), (53.8, 13.5)], ls="--")
    polyline_arrow(ax, [(43.5, 2.4),  (53.8, 2.4)], ls="--")

    arrow(ax, 58.2, 21.9, 59.4, 18.5)
    arrow(ax, 58.2, 13.5, 59.4, 13.5)
    arrow(ax, 58.2, 5.0,  59.4, 9.5)

    # =========================================================================
    # PANEL (c): Asymmetric Bottleneck Micro-Architecture (RGB CA+ECA / LWIR Dilated d=2)
    # =========================================================================
    rbox(ax, 66.2, 27.2, 13.2, 1.8, "Input Shape X: B × C × H × W", fc="#ffffff", fontsize=6.9)
    rbox(ax, 66.8, 23.3, 12.0, 2.8, "Conv 3×3, s=1 (Dense)\nC → C (100% COCO Transfer)", fc="#dae8fc", fontsize=6.7)
    rbox(ax, 66.8, 19.3, 12.0, 2.9, "RGB: CoordAtt + Dense 3×3 (d=1)\nLWIR: Dilated Conv 3×3 (d=2, p=2)", fc="#b8c9d9", fontsize=6.6, bold=True)
    rbox(ax, 66.8, 15.2, 12.0, 2.9, "ECA Module (1D Conv, k=3)\nChannel Calibration", fc="#ffe6cc", fontsize=6.7)
    op_circle(ax, 72.8, 13.4, r=0.62, sym="+")
    rbox(ax, 68.8, 10.7, 8.0, 1.6, "Bottleneck Output", fc="#ffffff", fontsize=6.9)

    arrow(ax, 72.8, 27.2, 72.8, 26.1)
    arrow(ax, 72.8, 23.3, 72.8, 22.2)
    arrow(ax, 72.8, 19.3, 72.8, 18.1)
    arrow(ax, 72.8, 15.2, 72.8, 14.05)
    arrow(ax, 72.8, 12.75, 72.8, 12.3)
    polyline_arrow(ax, [(79.4, 28.1), (80.3, 28.1), (80.3, 13.4), (73.45, 13.4)])
    ax.text(81.1, 20.8, "Residual Connection", rotation=270, ha="center", va="center", fontsize=6.8)

    # =========================================================================
    # PANEL (d): Cross-Modal Attention Fusion (CMAF) Gate Micro-Architecture
    # =========================================================================
    rbox(ax, 83.2, 27.1, 7.1, 1.9, "F_rgb [B,C,H,W]", fc="#d5e8d4", fontsize=6.6, bold=True)
    rbox(ax, 90.9, 27.1, 7.1, 1.9, "F_thm [B,C,H,W]", fc="#ffe6cc", fontsize=6.6, bold=True)
    rbox(ax, 84.3, 23.5, 12.6, 2.4, "Channel Concat [B, 2C, H, W]\n+ 1×1 Conv Reduction (F_red)", fc="#dae8fc", fontsize=6.6)
    rbox(ax, 84.3, 19.6, 12.6, 2.7, "Spatial Attention Gate α_s\nConv1×1(2C→C/4)→ReLU→Conv1×1→σ", fc="#ffe6cc", fontsize=6.5, bold=True)
    rbox(ax, 84.3, 15.4, 12.6, 2.9, "Modality Gating:\nα_s · F_rgb + (1 - α_s) · F_thm", fc="#e1d5e7", fontsize=6.7, bold=True)
    op_circle(ax, 90.6, 13.4, r=0.62, sym="+")
    rbox(ax, 86.6, 10.7, 8.0, 1.6, "Fused Output F_fuse", fc="#ffffff", fontsize=6.9)

    arrow(ax, 86.7, 27.1, 88.5, 25.9)
    arrow(ax, 94.4, 27.1, 92.5, 25.9)
    arrow(ax, 90.6, 23.5, 90.6, 22.3)
    arrow(ax, 90.6, 19.6, 90.6, 18.3)
    arrow(ax, 90.6, 15.4, 90.6, 14.05)
    arrow(ax, 90.6, 12.75, 90.6, 12.3)
    polyline_arrow(ax, [(96.9, 24.7), (98.0, 24.7), (98.0, 13.4), (91.25, 13.4)])
    ax.text(98.8, 19.0, "F_red Residual", rotation=270, ha="center", va="center", fontsize=6.8)

    # =========================================================================
    # PANEL (e): 2D-Grid Anchor-Aligned CWAR / MACAF Fusion Pipeline
    # =========================================================================
    rbox(ax, 65.6, 5.4, 7.6, 3.8, "Multi-Stream Grid\nPredictions (M Models)\nShape=[B, 4+C, 8400]\n(Strides 8, 16, 32)", fc="#ffffff", fontsize=6.5)
    rbox(ax, 74.4, 5.8, 7.8, 3.2, "2D-Grid Flip Unwarp\nReshape [B,8,Hs,Ws]\nFlip Ws → 1-to-1 Align", fc="#d5b8d8", fontsize=6.5)
    rbox(ax, 83.4, 5.8, 7.6, 3.2, "Class-Specific Routing\nW [M×4] (MACAF)\n& Peak Conf (max p)^γ", fc="#e69138", fontsize=6.5, bold=True)
    rbox(ax, 92.2, 2.2, 6.8, 6.8, "Confidence-\nWeighted Box\nRegression\n(CWAR)\nb_hat = Σ(ψ_m·b_m)\n        / Σψ_m", fc="#cfe2f3", fontsize=6.5, bold=True)

    rbox(ax, 65.6, 0.9, 7.6, 3.6, "Fused Pre-NMS Grid\nShape=[B, 4+C, 8400]\n(92.33% Test mAP50)", fc="#ffffff", fontsize=6.5, bold=True)
    op_circle(ax, 76.2, 2.7, r=0.65, sym="×")
    rbox(ax, 78.2, 1.3, 4.6, 2.8, "Normalize\nWeights\nψ_m / Σψ", fc="#d5a6bd", fontsize=6.4)
    rbox(ax, 85.4, 1.3, 5.6, 2.8, "Anchor Grid\nBroadcast\n[B, 4, 8400]", fc="#e69138", fontsize=6.4)

    arrow(ax, 73.2, 7.4, 74.4, 7.4)
    arrow(ax, 82.2, 7.4, 83.4, 7.4)
    arrow(ax, 91.0, 7.4, 92.2, 7.4)
    arrow(ax, 92.2, 2.7, 91.0, 2.7)
    arrow(ax, 85.4, 2.7, 82.8, 2.7)
    arrow(ax, 78.2, 2.7, 76.85, 2.7)
    arrow(ax, 73.2, 5.6, 75.7, 3.2)
    arrow(ax, 75.55, 2.7, 73.2, 2.7)

    plt.tight_layout(pad=0.1)
    pdf_out = OUT_DIR / "Multimodal_Dilated_ECA_Yolo.pdf"
    png_out = OUT_DIR / "Multimodal_Dilated_ECA_Yolo.png"
    fig.savefig(pdf_out, dpi=300, bbox_inches="tight")
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    shutil.copy2(png_out, ART_DIR / "Multimodal_Dilated_ECA_Yolo.png")
    shutil.copy2(pdf_out, ART_DIR / "Multimodal_Dilated_ECA_Yolo.pdf")
    print("Saved", pdf_out, "and", png_out)


if __name__ == "__main__":
    build_diagram()
