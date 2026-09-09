"""
scripts/generate_fig1_dataset_samples.py
==========================================
Generates Figure 1 for the IEEE Sensors Journal manuscript:
Thermal perception dataset samples across the four target categories:
  (a) Pedestrian Mobility (Person)
  (b) Wheeled Luggage (Trolley)
  (c) Carried & Handheld Bags (Bag)
  (d) Unattended Luggage Anomaly (Unattended)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

IMG_DIR = Path("data/thermal/images")
LBL_DIR = Path("data/thermal/labels")
OUT_DIR = Path("paper_assets/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.titlesize": 11,
})

CLASS_META = {
    0: ("Person", "#1d4ed8"),      # Royal Blue
    1: ("Trolley", "#0284c7"),     # Cyan / Sky Blue
    2: ("Bag", "#d97706"),         # Amber / Warm Gold
    3: ("Unattended", "#dc2626")   # Crimson Red
}

def draw_frame(ax, img_id, title, target_cls):
    img_path = IMG_DIR / f"{img_id}.jpg"
    lbl_path = LBL_DIR / f"{img_id}.txt"
    
    im = Image.open(img_path)
    w, h = im.size
    ax.imshow(im)
    ax.set_title(title, pad=6, fontweight="bold")
    ax.axis("off")
    
    if not lbl_path.exists():
        return
        
    with open(lbl_path) as f:
        lines = [line.strip().split() for line in f if line.strip()]
        
    boxes = []
    for parts in lines:
        cls_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2.0) * w
        y1 = (yc - bh / 2.0) * h
        box_w = bw * w
        box_h = bh * h
        boxes.append((cls_id, x1, y1, box_w, box_h))
        
    # Sort so target class is drawn on top
    boxes.sort(key=lambda b: (b[0] == target_cls, b[3]*b[4]))
    
    placed_labels = []
    for cls_id, x1, y1, box_w, box_h in boxes:
        is_target = (cls_id == target_cls)
        name, color = CLASS_META.get(cls_id, ("Obj", "#ffffff"))
        
        lw = 2.4 if is_target else 1.3
        alpha = 1.0 if is_target else 0.45
        ls = "-" if is_target else "--"
        
        rect = patches.Rectangle(
            (x1, y1), box_w, box_h,
            linewidth=lw, edgecolor=color, facecolor="none",
            alpha=alpha, linestyle=ls
        )
        ax.add_patch(rect)
        
        # Calculate non-overlapping label position
        label_y = max(16, y1 - 5)
        for px, py in placed_labels:
            if abs(px - x1) < 55 and abs(py - label_y) < 18:
                label_y = y1 + box_h + 14  # place below box
                break
                
        placed_labels.append((x1, label_y))
        ax.text(
            x1 + 2, label_y,
            name,
            color="white", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.2", facecolor=color, edgecolor="none", alpha=alpha)
        )

def main():
    samples = [
        ("FLIR8282", "(a) Person: Pedestrian & Luggage Interaction", 0),
        ("FLIR7988", "(b) Trolley: Wheeled Luggage in Transit", 1),
        ("FLIR9010", "(c) Bag: Handheld & Worn Backpacks", 2),
        ("FLIR8796", "(d) Unattended: Stationary Thermal Anomaly", 3),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axes = axes.flatten()
    
    for idx, (img_id, title, cls_id) in enumerate(samples):
        draw_frame(axes[idx], img_id, title, cls_id)
        
    plt.tight_layout(pad=0.8)
    
    pdf_path = OUT_DIR / "fig1_dataset_samples.pdf"
    png_path = OUT_DIR / "fig1_dataset_samples.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: {pdf_path}")
    print(f"Generated: {png_path}")

if __name__ == "__main__":
    main()
