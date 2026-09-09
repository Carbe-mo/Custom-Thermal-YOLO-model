"""
models/v0b_baseline_100ep/train.py
===================================
Stock YOLOv8n baseline trained under the EXACT SAME CONDITIONS as v30 and v33:
- 100 Epochs
- Cosine Annealing Learning Rate Schedule (cos_lr=True, lrf=0.01)
- 20 Epochs Un-augmented Fine-Tuning (close_mosaic=20)
- Identical batch=16, imgsz=640, device=0, pretrained=yolov8n.pt

Provides the rigorous controlled baseline to measure the true net gain of our
architectural enhancements under identical training dynamics.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from common.dataset import build_yaml
from common.results_logger import log_run

VARIANT_ID = "v0b"
VARIANT_NAME = "v0b_baseline_100ep"
MODEL_WEIGHTS = "yolov8n.pt"
RUNS_DIR = PROJECT_ROOT / "runs" / VARIANT_NAME


def train(epochs: int = 100, batch: int = 16, imgsz: int = 640) -> None:
    data_yaml = build_yaml("thermal")

    model = YOLO(MODEL_WEIGHTS)

    t0 = time.time()
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(RUNS_DIR),
        name="train",
        exist_ok=True,
        pretrained=True,
        cos_lr=True,
        lrf=0.01,
        close_mosaic=20,
        workers=0,
        device=0,
        verbose=True,
    )
    train_minutes = (time.time() - t0) / 60.0

    metrics_val = model.val(data=str(data_yaml), split="val", workers=0, device=0)
    mAP50_val = float(metrics_val.box.map50)
    mAP50_95_val = float(metrics_val.box.map)
    prec_val = float(metrics_val.box.mp)
    rec_val = float(metrics_val.box.mr)

    metrics_test = model.val(data=str(data_yaml), split="test", workers=0, device=0)
    mAP50_test = float(metrics_test.box.map50)
    mAP50_95_test = float(metrics_test.box.map)
    prec_test = float(metrics_test.box.mp)
    rec_test = float(metrics_test.box.mr)
    inference_ms = metrics_test.speed.get("inference", 0.0)

    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6

    log_run(
        model_id=VARIANT_ID,
        model_name=f"{VARIANT_NAME} (Stock YOLOv8n + 100ep Cosine LR)",
        folder=f"models/{VARIANT_NAME}",
        params_M=round(n_params, 2),
        gflops=8.1,
        epochs=epochs,
        mAP50=round(mAP50_val, 4),
        mAP50_95=round(mAP50_95_val, 4),
        precision=round(prec_val, 4),
        recall=round(rec_val, 4),
        inference_ms=round(inference_ms, 2),
        train_time_min=round(train_minutes, 1),
        notes="Stock YOLOv8n baseline trained under identical 100ep Cosine LR conditions as v30 and v33",
    )

    log_run(
        model_id=f"{VARIANT_ID}_eval",
        model_name=f"{VARIANT_NAME} (100ep Cosine Baseline) [Official Test]",
        folder=f"models/{VARIANT_NAME}",
        params_M=round(n_params, 2),
        gflops=8.1,
        epochs=epochs,
        mAP50=round(mAP50_test, 4),
        mAP50_95=round(mAP50_95_test, 4),
        precision=round(prec_test, 4),
        recall=round(rec_test, 4),
        inference_ms=round(inference_ms, 2),
        train_time_min=round(train_minutes, 1),
        notes="official test split evaluation on held-out data/splits/test.txt (108 images)",
    )

    print(f"\n[{VARIANT_NAME}] Val mAP50: {mAP50_val*100:.2f}% | Test mAP50: {mAP50_test*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
