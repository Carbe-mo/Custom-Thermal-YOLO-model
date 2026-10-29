"""
scripts/build_fusion6_dataset.py
================================
Build 6-channel multi-page TIFF images for dual-stream RGB+thermal training.

Channel (page) order:  [R_rgb, G_rgb, B_rgb, R_thm, G_thm, B_thm]
(RGB order, because Ultralytics does not BGR->RGB flip non-3-channel images.)

Modes:
  registered   (default) RGB frame is cropped/resampled into the thermal FOV using
               the transform from scripts/register_rgb_thermal.py
               (data/rgb_thermal_registration.json). Output: data/fusion6/
  unregistered RGB frame is simply resized to 640x480 (the naive pairing used by
               the earlier v34/v35 runs). Output: data/fusion6_unreg/   [control]

Labels are copied from data/thermal/labels (they are defined in thermal coordinates,
which after registration are also valid for the RGB channels).
Split lists data/<out>/{train,val,test}.txt follow data/splits/*.txt.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from common.dataset import generate_splits, CLASS_NAMES, NC  # noqa: E402

THM_IMG = PROJECT_ROOT / "data" / "thermal" / "images"
THM_LBL = PROJECT_ROOT / "data" / "thermal" / "labels"
RGB_DIR = Path(r"D:\Yolo\Data\project_flir\100_FLIR")
REG_JSON = PROJECT_ROOT / "data" / "rgb_thermal_registration.json"


def build(mode: str = "registered") -> Path:
    out = PROJECT_ROOT / "data" / ("fusion6" if mode == "registered" else "fusion6_unreg")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    reg = json.loads(REG_JSON.read_text())
    M = np.float32([[reg["scale"], 0, reg["x0"]], [0, reg["scale"], reg["y0"]]])

    splits = generate_splits()
    missing_rgb = []
    for split, stems in splits.items():
        lines = []
        for stem in stems:
            tid = int(stem.replace("FLIR", ""))
            thm = cv2.imread(str(THM_IMG / f"{stem}.jpg"))
            rgb = cv2.imread(str(RGB_DIR / f"FLIR{tid + 1}.jpg"))
            if thm is None:
                continue
            thm = cv2.resize(thm, (640, 480))
            if rgb is None:
                missing_rgb.append(stem)
                rgb = np.zeros_like(thm)
            else:
                rgb = cv2.resize(rgb, (640, 480))
                if mode == "registered":
                    rgb = cv2.warpAffine(rgb, M, (640, 480),
                                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_REPLICATE)
            # BGR (cv2) -> RGB page order
            pages = [rgb[..., 2], rgb[..., 1], rgb[..., 0], thm[..., 2], thm[..., 1], thm[..., 0]]
            dst = out / "images" / f"{stem}.tiff"
            if not dst.exists():
                cv2.imwritemulti(str(dst), pages)
            lbl = THM_LBL / f"{stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, out / "labels" / f"{stem}.txt")
            lines.append(str(dst))
        (out / f"{split}.txt").write_text("\n".join(lines) + "\n")
        print(f"[{mode}] {split}: {len(lines)} images")

    cfg = {
        "path": str(out), "train": str(out / "train.txt"), "val": str(out / "val.txt"),
        "test": str(out / "test.txt"), "nc": NC, "names": CLASS_NAMES, "channels": 6,
    }
    yml = PROJECT_ROOT / "data" / f"{out.name}_data.yaml"
    yml.write_text(yaml.dump(cfg, sort_keys=False))
    print(f"[{mode}] missing RGB pairs: {len(missing_rgb)} {missing_rgb[:10]}")
    print(f"[{mode}] wrote {yml}")
    return yml


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "registered")
