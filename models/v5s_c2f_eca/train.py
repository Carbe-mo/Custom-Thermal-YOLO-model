"""
models/v5s_c2f_eca/train.py
===========================
Train YOLOv8s (Scale 's' 2x Channel Dimension Capacity) with C2f_ECA Backbone.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# --- ensure project root is importable --------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.modules.c2f_eca import register_c2f_eca
from common.dataset import build_yaml
from common.results_logger import log_run
from ultralytics import YOLO

# Register custom module before loading model
register_c2f_eca()

VARIANT_ID = "v5s"
VARIANT_NAME = "v5s_c2f_eca"
MODEL_YAML = PROJECT_ROOT / "models" / VARIANT_NAME / "model.yaml"
PRETRAINED_WEIGHTS = "yolov8s.pt"
RUNS_DIR = PROJECT_ROOT / "runs" / VARIANT_NAME


def train(epochs: int = 50, batch: int = 16, imgsz: int = 640) -> None:
    data_yaml = build_yaml("thermal")

    # Load architecture
    model = YOLO(str(MODEL_YAML))

    t0 = time.time()
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(RUNS_DIR),
        name="train",
        exist_ok=True,
        workers=0,
        device=0,
        pretrained=PRETRAINED_WEIGHTS,
        verbose=True,
    )
    train_minutes = (time.time() - t0) / 60.0

    # 1. Evaluate on val split (validation metrics)
    metrics_val = model.val(
        data=str(data_yaml),
        split="val",
        workers=0,
        device=0,
    )

    mAP50_val = float(metrics_val.box.map50)
    mAP50_95_val = float(metrics_val.box.map)
    prec_val = float(metrics_val.box.mp)
    rec_val = float(metrics_val.box.mr)

    # 2. Evaluate on held-out test split (Official test metrics)
    metrics_test = model.val(
        data=str(data_yaml),
        split="test",
        workers=0,
        device=0,
    )

    mAP50_test = float(metrics_test.box.map50)
    mAP50_95_test = float(metrics_test.box.map)
    prec_test = float(metrics_test.box.mp)
    rec_test = float(metrics_test.box.mr)
    speed = metrics_test.speed
    inference_ms = speed.get("inference", 0.0)

    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6

    # Log training run
    log_run(
        model_id=VARIANT_ID,
        model_name=f"{VARIANT_NAME} (YOLOv8s + C2f_ECA Backbone 2x Dimension)",
        folder=f"models/{VARIANT_NAME}",
        params_M=round(n_params, 2),
        gflops=28.7,
        epochs=epochs,
        mAP50=round(mAP50_val, 4),
        mAP50_95=round(mAP50_95_val, 4),
        precision=round(prec_val, 4),
        recall=round(rec_val, 4),
        inference_ms=round(inference_ms, 2),
        train_time_min=round(train_minutes, 1),
        notes="Scale 's' channel capacity doubling (11.14M params) with C2f_ECA backbone and yolov8s.pt transfer",
    )

    # Log official eval run
    log_run(
        model_id=f"{VARIANT_ID}_eval",
        model_name=f"{VARIANT_NAME} (Scale s C2f_ECA) [Official Test]",
        folder=f"models/{VARIANT_NAME}",
        params_M=round(n_params, 2),
        gflops=28.7,
        epochs=epochs,
        mAP50=round(mAP50_test, 4),
        mAP50_95=round(mAP50_95_test, 4),
        precision=round(prec_test, 4),
        recall=round(rec_test, 4),
        inference_ms=round(inference_ms, 2),
        train_time_min=round(train_minutes, 1),
        notes="official test split evaluation on held-out data/splits/test.txt (108 images)",
    )

    print(f"\n[v5s_c2f_eca] Val mAP50: {mAP50_val*100:.2f}% | Test mAP50: {mAP50_test*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
