"""
scripts/generate_fig6_qualitative_detections.py
=================================================
Generates Figure 6 for the IEEE Sensors Journal manuscript:
Side-by-side qualitative detection comparison on challenging thermal test frames:
  Column 1: Ground Truth
  Column 2: Stock YOLOv8n Baseline (v0b)
  Column 3: Proposed Hybrid Dilated ECA-YOLO (v30)
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from common.modules.c2f_eca import register_c2f_eca
from common.modules.c2f_eca_dilated import register_c2f_eca_dilated
register_c2f_eca()
register_c2f_eca_dilated()

from ultralytics import YOLO

IMG_DIR = REPO_ROOT / "data" / "thermal" / "images"
LBL_DIR = REPO_ROOT / "data" / "thermal" / "labels"
OUT_DIR = REPO_ROOT / "paper_assets" / "figures"

V0B_WEIGHTS = REPO_ROOT / "runs" / "v0b_baseline_100ep" / "train" / "weights" / "best.pt"
V30_WEIGHTS = REPO_ROOT / "runs" / "v30_v27_cosine_100ep" / "train" / "weights" / "best.pt"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "axes.titlesize": 10.5,
})

CLASS_META = {
    0: ("Person", "#1d4ed8"),      # Blue
    1: ("Trolley", "#0284c7"),     # Cyan
    2: ("Bag", "#d97706"),         # Amber
    3: ("Unattended", "#dc2626")   # Crimson Red
}

def draw_gt(ax, img_path, lbl_path):
    im = Image.open(img_path)
    w, h = im.size
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if not lbl_path.exists(): return
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = (xc - bw / 2.0) * w
            y1 = (yc - bh / 2.0) * h
            box_w = bw * w
            box_h = bh * h
            
            name, color = CLASS_META.get(cls_id, ("Obj", "white"))
            rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2.0, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x1 + 2, max(14, y1 - 4), name, color="white", fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.15", facecolor=color, edgecolor="none", alpha=0.92))

def draw_pred(ax, img_path, model, conf_thresh=0.35):
    im = Image.open(img_path)
    w, h = im.size
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    
    res = model(img_path, conf=conf_thresh, verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0: return
    
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].tolist()
        x1, y1, x2, y2 = xyxy
        box_w = x2 - x1
        box_h = y2 - y1
        
        name, color = CLASS_META.get(cls_id, ("Obj", "white"))
        rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2.0, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        label = f"{name} {conf:.2f}"
        ax.text(x1 + 2, max(14, y1 - 4), label, color="white", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.15", facecolor=color, edgecolor="none", alpha=0.92))

def main():
    model_v0b = YOLO(str(V0B_WEIGHTS))
    model_v30 = YOLO(str(V30_WEIGHTS))
    
    test_scenes = [
        ("FLIR8796", "Scene 1: Concourse\n(Thermal Camouflage)"),
        ("FLIR9970", "Scene 2: Distant Luggage\n(Weak Boundary)"),
        ("FLIR9010", "Scene 3: Waiting Lounge\n(Dense Occlusion)"),
    ]
    
    n_rows = len(test_scenes)
    fig, axes = plt.subplots(n_rows, 3, figsize=(7.2, 5.6), gridspec_kw={'wspace': 0.04, 'hspace': 0.08})
    
    col_titles = ["(a) Ground Truth", "(b) Stock Baseline (v0b)", "(c) Proposed Model (v30)"]
    
    for row_idx, (img_id, scene_name) in enumerate(test_scenes):
        img_path = IMG_DIR / f"{img_id}.jpg"
        lbl_path = LBL_DIR / f"{img_id}.txt"
        
        draw_gt(axes[row_idx, 0], img_path, lbl_path)
        draw_pred(axes[row_idx, 1], img_path, model_v0b, conf_thresh=0.35)
        draw_pred(axes[row_idx, 2], img_path, model_v30, conf_thresh=0.35)
        
        axes[row_idx, 0].set_ylabel(scene_name, fontsize=8.5, fontweight="bold", labelpad=4)
        
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, pad=5, fontweight="bold")
        
    pdf_path = OUT_DIR / "fig6_qualitative_detections.pdf"
    png_path = OUT_DIR / "fig6_qualitative_detections.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {pdf_path}")
    print(f"Generated: {png_path}")

if __name__ == "__main__":
    main()
