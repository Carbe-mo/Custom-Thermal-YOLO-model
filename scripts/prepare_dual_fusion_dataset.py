"""
scripts/prepare_dual_fusion_dataset.py
======================================
Prepares the 6-channel dataset for Dual-Backbone RGB-Thermal Fusion (v35):
  - Channels 0-2: Visible RGB image (FLIR{N+1}.jpg)
  - Channels 3-5: Thermal pseudo-color image (FLIR{N}.jpg)

Saves as native NumPy .npy arrays [480, 640, 6] uint8:
  Ultralytics BaseDataset automatically loads .npy files with 6 channels
  without OpenCV 3-channel truncation.
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

def main():
    project_root = Path(__file__).resolve().parent.parent
    thermal_img_dir = project_root / 'data' / 'thermal' / 'images'
    thermal_lbl_dir = project_root / 'data' / 'thermal' / 'labels'
    rgb_source_dir = Path(r"D:\Yolo\Data\project_flir\100_FLIR")

    fusion_dual_dir = project_root / 'data' / 'fusion_dual'
    img_dir = fusion_dual_dir / 'images'
    lbl_dir = fusion_dual_dir / 'labels'

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    thermal_files = sorted(list(thermal_img_dir.glob('*.jpg')))
    print(f"Preparing 6-channel arrays for {len(thermal_files)} image pairs...")

    for t_path in tqdm(thermal_files, desc="Writing 6-channel .npy arrays"):
        stem = t_path.stem
        match = re.search(r'\d+', stem)
        if not match:
            continue

        idx_str = match.group()
        rgb_idx = int(idx_str) + 1
        rgb_stem = stem[:match.start()] + str(rgb_idx).zfill(len(idx_str)) + stem[match.end():]
        rgb_path = rgb_source_dir / f"{rgb_stem}.jpg"

        if not rgb_path.exists():
            print(f"Warning: Missing RGB pair {rgb_path}")
            continue

        t_img = cv2.imread(str(t_path))        # [480, 640, 3] BGR Thermal
        rgb_img = cv2.imread(str(rgb_path))    # [480, 640, 3] BGR Visual

        if t_img is None or rgb_img is None:
            continue

        if t_img.shape[:2] != rgb_img.shape[:2]:
            rgb_img = cv2.resize(rgb_img, (t_img.shape[1], t_img.shape[0]))

        # Stack into 6-channel array: [BGR_rgb, BGR_thermal]
        arr_6ch = np.concatenate([rgb_img, t_img], axis=2).astype(np.uint8)  # [480, 640, 6]

        # Save as .npy
        npy_path = img_dir / f"{stem}.npy"
        np.save(str(npy_path), arr_6ch)

        # Also write a dummy tiny .jpg so Ultralytics find_images discovers the file
        jpg_path = img_dir / f"{stem}.jpg"
        if not jpg_path.exists():
            cv2.imwrite(str(jpg_path), t_img)

        # Copy label
        lbl_path = thermal_lbl_dir / f"{stem}.txt"
        if lbl_path.exists():
            shutil.copy(lbl_path, lbl_dir / f"{stem}.txt")

    print("Generating train / val / test splits...")
    splits_dir = project_root / 'data' / 'splits'

    for split_name in ['train.txt', 'val.txt', 'test.txt']:
        src_split = splits_dir / split_name
        if src_split.exists():
            with open(src_split, 'r') as f:
                stems = [line.strip() for line in f if line.strip()]

            out_lines = [str(img_dir / f"{s}.jpg") for s in stems]
            dst_split = fusion_dual_dir / split_name
            with open(dst_split, 'w') as f:
                f.write('\n'.join(out_lines) + '\n')

    print(f"\nSuccessfully prepared 6-channel dataset at {fusion_dual_dir}")
    print(f"Total .npy files: {len(list(img_dir.glob('*.npy')))}")

if __name__ == '__main__':
    main()
