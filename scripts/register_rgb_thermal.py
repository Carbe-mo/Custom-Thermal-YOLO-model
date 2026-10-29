"""
scripts/register_rgb_thermal.py
===============================
Estimate the geometric mapping between FLIR C5 thermal frames and their paired
RGB frames. The RGB camera has a wider FOV, so the thermal view corresponds to a
scaled sub-window of the RGB image.

Method: FLIR MSX blends visible-camera edges into the thermal image, so both
modalities share edge structure. For each pair we compute gradient-magnitude
maps and run multi-scale normalized cross-correlation template matching of the
(overlay-free) thermal interior against the RGB edge map.

Output: scratch CSV of per-pair (scale, x0, y0, score) and the median transform,
saved to data/rgb_thermal_registration.json:
    thermal pixel (u, v) in [0,W]x[0,H]  ->  RGB pixel (x0 + s*u, y0 + s*v)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THM_DIR = PROJECT_ROOT / "data" / "thermal" / "images"
RGB_DIR = Path(r"D:\Yolo\Data\project_flir\100_FLIR")
OUT_JSON = PROJECT_ROOT / "data" / "rgb_thermal_registration.json"

# Thermal interior free of overlays (temp bar on the left, FLIR logo top-right)
CROP = (70, 420, 100, 600)  # y1, y2, x1, x2 in 640x480 thermal coordinates


def edges(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    return m / (m.max() + 1e-6)


def register_pair(tid: int, scales=np.arange(0.50, 1.001, 0.01)):
    t = cv2.imread(str(THM_DIR / f"FLIR{tid}.jpg"), cv2.IMREAD_GRAYSCALE)
    r = cv2.imread(str(RGB_DIR / f"FLIR{tid + 1}.jpg"), cv2.IMREAD_GRAYSCALE)
    if t is None or r is None:
        return None
    t = cv2.resize(t, (640, 480))
    r = cv2.resize(r, (640, 480))
    te, re_ = edges(t), edges(r)
    y1, y2, x1, x2 = CROP
    tmpl_full = te[y1:y2, x1:x2]
    best = (-1.0, None, None, None)
    for s in scales:
        tw, th = int(round((x2 - x1) * s)), int(round((y2 - y1) * s))
        if tw >= 640 or th >= 480 or tw < 20:
            continue
        tmpl = cv2.resize(tmpl_full, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(re_, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            # loc is position of crop's top-left; convert to full-thermal origin
            x0 = loc[0] - s * x1
            y0 = loc[1] - s * y1
            best = (mx, float(s), float(x0), float(y0))
    return best


def main(step: int = 10):
    ids = sorted(int(p.stem.replace("FLIR", "")) for p in THM_DIR.glob("*.jpg"))
    rows = []
    for tid in ids[::step]:
        res = register_pair(tid)
        if res is None or res[1] is None:
            continue
        score, s, x0, y0 = res
        rows.append((tid, score, s, x0, y0))
        print(f"FLIR{tid}: score={score:.3f} s={s:.2f} x0={x0:.1f} y0={y0:.1f}")
    arr = np.array([r[1:] for r in rows])
    good = arr[arr[:, 0] > np.percentile(arr[:, 0], 25)]  # drop weakest 25% matches
    med = np.median(good, axis=0)
    mad = np.median(np.abs(good - med), axis=0)
    result = {
        "scale": float(med[1]), "x0": float(med[2]), "y0": float(med[3]),
        "mad_scale": float(mad[1]), "mad_x0": float(mad[2]), "mad_y0": float(mad[3]),
        "n_pairs": int(len(rows)), "n_used": int(len(good)),
        "median_score": float(med[0]),
        "convention": "rgb_px = (x0 + scale*u, y0 + scale*v), u,v thermal px at 640x480",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    np.savetxt(PROJECT_ROOT / "data" / "rgb_thermal_registration_pairs.csv",
               np.array(rows), delimiter=",", header="tid,score,scale,x0,y0", fmt="%.4f")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
