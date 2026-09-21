import os
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm
import re

def main():
    project_root = Path(__file__).resolve().parent.parent
    thermal_img_dir = project_root / 'data' / 'thermal' / 'images'
    thermal_lbl_dir = project_root / 'data' / 'thermal' / 'labels'
    rgb_source_dir = Path(r"D:\Yolo\Data\project_flir\100_FLIR")
    
    fusion_dir = project_root / 'data' / 'fusion'
    fusion_img_dir = fusion_dir / 'images'
    fusion_lbl_dir = fusion_dir / 'labels'
    
    fusion_img_dir.mkdir(parents=True, exist_ok=True)
    fusion_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    if not thermal_img_dir.exists():
        print(f"Error: {thermal_img_dir} does not exist.")
        return
        
    thermal_files = list(thermal_img_dir.glob('*.jpg'))
    print(f"Found {len(thermal_files)} thermal images.")
    
    for t_path in tqdm(thermal_files, desc="Creating 4-channel fusion images"):
        stem = t_path.stem
        match = re.search(r'\d+', stem)
        if not match:
            print(f"Warning: Could not parse index from {stem}")
            continue
            
        idx_str = match.group()
        rgb_idx = int(idx_str) + 1
        rgb_stem = stem[:match.start()] + str(rgb_idx).zfill(len(idx_str)) + stem[match.end():]
        rgb_path = rgb_source_dir / f"{rgb_stem}.jpg"
        
        if not rgb_path.exists():
            print(f"Warning: RGB pair {rgb_path} not found for {t_path}")
            continue
            
        t_img = cv2.imread(str(t_path))
        rgb_img = cv2.imread(str(rgb_path))
        
        if t_img is None or rgb_img is None:
            continue
            
        if t_img.shape[:2] != rgb_img.shape[:2]:
            rgb_img = cv2.resize(rgb_img, (t_img.shape[1], t_img.shape[0]))
            
        t_gray = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(rgb_img)
        
        # Stack as B, G, R, T_gray so that when loaded by OpenCV it corresponds to standard structure
        # (Actually, OpenCV loads as BGRA, so A is T_gray. Then Ultralytics will load image natively using OpenCV)
        fusion_img = cv2.merge([b, g, r, t_gray])
        
        out_path = fusion_img_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), fusion_img)
        
        lbl_path = thermal_lbl_dir / f"{stem}.txt"
        if lbl_path.exists():
            out_lbl_path = fusion_lbl_dir / f"{stem}.txt"
            shutil.copy(lbl_path, out_lbl_path)

    print("Copying labels and creating splits...")
    splits_dir = project_root / 'data' / 'splits'
    
    total_in_splits = 0
    for split_name in ['train.txt', 'val.txt', 'test.txt']:
        split_path = splits_dir / split_name
        if split_path.exists():
            with open(split_path, 'r') as f:
                stems = [line.strip() for line in f if line.strip()]
            
            out_lines = []
            for stem in stems:
                img_path = fusion_img_dir / f"{stem}.png"
                out_lines.append(str(img_path))
                
            out_split = fusion_dir / split_name
            with open(out_split, 'w') as f:
                f.write('\n'.join(out_lines) + '\n')
            total_in_splits += len(out_lines)
                
    print(f"Created fusion dataset at {fusion_dir}")
    print(f"Total processed images: {len(list(fusion_img_dir.glob('*.png')))}")
    print(f"Total images in splits: {total_in_splits}")

if __name__ == '__main__':
    main()
