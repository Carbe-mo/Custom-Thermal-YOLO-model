"""
models/v24_copy_paste_aug/train.py
====================================
v5b_c2f_eca architecture (current #1 leader) trained with copy-paste
augmentation (copy_paste=0.3) to address class imbalance.

Single variable vs. v5b: copy_paste augmentation only.

Class distribution: trolley (2145) >> person (1076) >> bag (353) >> unattended (275)
copy_paste=0.3 randomly copies/pastes instances from random images into the current
training image, effectively over-sampling rare classes (bag, unattended) by
synthesising extra instances from existing images.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.modules.c2f_eca import register_c2f_eca
from common.dataset import build_yaml
from common.results_logger import log_run
from ultralytics import YOLO

register_c2f_eca()

VARIANT_ID = "v24"
VARIANT_NAME = "v24_copy_paste_aug"
MODEL_YAML = PROJECT_ROOT / "models" / "v5b_c2f_eca" / "model.yaml"
PRETRAINED_WEIGHTS = "yolov8n.pt"
RUNS_DIR = PROJECT_ROOT / "runs" / VARIANT_NAME


def train(epochs: int = 50, batch: int = 16, imgsz: int = 640) -> None:
    data_yaml = build_yaml("thermal")

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
        # --- Class imbalance augmentation ---
        copy_paste=0.3,  # probability of copy-pasting instances from other images
    )
    train_minutes = (time.time() - t0) / 60.0

    metrics_val = model.val(data=str(data_yaml), split="val", workers=0, device=0)
    mAP50_val    = float(metrics_val.box.map50)
    mAP50_95_val = float(metrics_val.box.map)
    prec_val     = float(metrics_val.box.mp)
    rec_val      = float(metrics_val.box.mr)

    metrics_test = model.val(data=str(data_yaml), split="test", workers=0, device=0)
    mAP50_test    = float(metrics_test.box.map50)
    mAP50_95_test = float(metrics_test.box.map)
    prec_test     = float(metrics_test.box.mp)
    rec_test      = float(metrics_test.box.mr)
    inference_ms  = metrics_test.speed.get("inference", 0.0)

    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6

    log_run(
        model_id=VARIANT_ID,
        model_name=f"{VARIANT_NAME} (v5b arch + copy_paste=0.3)",
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
        notes="v5b arch; copy_paste=0.3 to over-sample rare bag/unattended classes",
    )

    log_run(
        model_id=f"{VARIANT_ID}_eval",
        model_name=f"{VARIANT_NAME} (CopyPaste Aug ECA) [Official Test]",
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--imgsz",  type=int, default=640)
    args = parser.parse_args()
    train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
