"""
models/v21_dilated_hybrid/train.py
====================================
Train YOLOv8n with Hybrid Dilated ECA Backbone:
  - Standard ECA bottlenecks at P2/P3 (fine-edge, small object detection)
  - Dilated d=2 ECA bottlenecks at P4/P5 (large context, thermal object segmentation)

Hypothesis: dilation in shallow stages blurs fine-grained boundaries needed for
small person detection. By restricting dilation to deep stages only we preserve
the tight 3x3 edge priors while still gaining contextual receptive field for
large composite objects (trolleys, bags) at P4/P5.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.modules.c2f_eca import register_c2f_eca
from common.modules.c2f_eca_dilated import register_c2f_eca_dilated
from common.dataset import build_yaml
from common.results_logger import log_run
from ultralytics import YOLO

# Register both modules
register_c2f_eca()
register_c2f_eca_dilated()

VARIANT_ID = "v21"
VARIANT_NAME = "v21_dilated_hybrid"
MODEL_YAML = PROJECT_ROOT / "models" / VARIANT_NAME / "model.yaml"
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
    )
    train_minutes = (time.time() - t0) / 60.0

    # Val split
    metrics_val = model.val(data=str(data_yaml), split="val", workers=0, device=0)
    mAP50_val    = float(metrics_val.box.map50)
    mAP50_95_val = float(metrics_val.box.map)
    prec_val     = float(metrics_val.box.mp)
    rec_val      = float(metrics_val.box.mr)

    # Official test split
    metrics_test = model.val(data=str(data_yaml), split="test", workers=0, device=0)
    mAP50_test    = float(metrics_test.box.map50)
    mAP50_95_test = float(metrics_test.box.map)
    prec_test     = float(metrics_test.box.mp)
    rec_test      = float(metrics_test.box.mr)
    inference_ms  = metrics_test.speed.get("inference", 0.0)

    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6

    log_run(
        model_id=VARIANT_ID,
        model_name=f"{VARIANT_NAME} (Hybrid: ECA@P2/P3 + Dilated-ECA@P4/P5)",
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
        notes="Standard ECA P2/P3 + Dilated d=2 ECA P4/P5; best of both worlds",
    )

    log_run(
        model_id=f"{VARIANT_ID}_eval",
        model_name=f"{VARIANT_NAME} (Hybrid Dilated ECA) [Official Test]",
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
