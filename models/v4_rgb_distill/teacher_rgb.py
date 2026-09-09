"""
models/v4_rgb_distill/teacher_rgb.py — RGB Teacher Model Training / Loading
===========================================================================

Trains (or loads) a YOLOv8n detector on the visible RGB paired dataset to serve
as a frozen knowledge teacher for cross-modal distillation into the thermal student.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.dataset import build_yaml
from ultralytics import YOLO

TEACHER_DIR = PROJECT_ROOT / "runs" / "v4_rgb_distill" / "teacher_rgb"
TEACHER_WEIGHTS = TEACHER_DIR / "weights" / "best.pt"


def train_or_load_teacher(epochs: int = 30, batch: int = 16, imgsz: int = 640) -> YOLO:
    """
    Returns a trained YOLO RGB teacher model.
    If the best.pt checkpoint already exists, loads it directly;
    otherwise trains a YOLOv8n model on data/rgb_data.yaml.
    """
    if TEACHER_WEIGHTS.exists():
        print(f"[teacher_rgb] Loading existing RGB teacher weights from {TEACHER_WEIGHTS}")
        teacher = YOLO(str(TEACHER_WEIGHTS))
        return teacher

    print("[teacher_rgb] Training RGB teacher model on data/rgb_data.yaml ...")
    rgb_yaml = build_yaml("rgb")
    
    teacher = YOLO("yolov8n.pt")
    t0 = time.time()
    teacher.train(
        data=str(rgb_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(TEACHER_DIR.parent),
        name="teacher_rgb",
        exist_ok=True,
        workers=0,
        device=0,
        verbose=True,
    )
    print(f"[teacher_rgb] RGB Teacher training complete in {(time.time() - t0)/60.0:.1f} min")
    return teacher


if __name__ == "__main__":
    teacher = train_or_load_teacher(epochs=30)
    print("RGB Teacher is ready!")
