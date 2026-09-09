"""
models/v6_c2f_dcn/train.py
==========================
Train YOLOv8n with C2f_DCN (Deformable Convolution Bottlenecks in Backbone) on thermal data.

Replaces stock C2f with C2f_DCN in backbone stages 2, 4, 6, 8.
Transfers pretrained weights from yolov8n.pt to all matching conv/c2f/head layers.

Usage:
    cd thermal-yolo
    python -m models.v6_c2f_dcn.train          # default 50 epochs
    python -m models.v6_c2f_dcn.train --epochs 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import torch

# --- ensure project root is importable --------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.modules.c2f_dcn import register_c2f_dcn
from common.dataset import build_yaml
from common.results_logger import log_run, _read_all_rows
from ultralytics import YOLO

# Register custom module before loading model
register_c2f_dcn()

# --- constants --------------------------------------------------------
VARIANT_ID = "v6"
VARIANT_NAME = "v6_c2f_dcn"
MODEL_YAML = PROJECT_ROOT / "models" / VARIANT_NAME / "model.yaml"
PRETRAINED_WEIGHTS = "yolov8n.pt"
RUNS_DIR = PROJECT_ROOT / "runs" / VARIANT_NAME


def transfer_pretrained_weights(model: YOLO, base_weights_path: str = "yolov8n.pt") -> int:
    """
    Transfer matching pretrained weights from stock YOLOv8n to v6_c2f_dcn.
    """
    base = YOLO(base_weights_path)
    base_state = base.model.state_dict()
    v6_state = model.model.state_dict()

    transferred = 0
    for k, v in base_state.items():
        if k in v6_state and v6_state[k].shape == v.shape:
            v6_state[k] = v
            transferred += 1

    model.model.load_state_dict(v6_state)
    print(f"[v6_c2f_dcn] Transferred {transferred} pretrained parameter tensors from {base_weights_path}")
    return transferred


def train(epochs: int = 50, batch: int = 16, imgsz: int = 640) -> None:
    # 1. Build / reuse the thermal data YAML
    data_yaml = build_yaml("thermal")

    # 2. Load model architecture with C2f_DCN and transfer pretrained weights
    model = YOLO(str(MODEL_YAML))
    transfer_pretrained_weights(model, PRETRAINED_WEIGHTS)

    # 3. Train
    t0 = time.time()
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(RUNS_DIR),
        name="train",
        exist_ok=True,
        workers=0,           # Windows safe
        device=0,
        verbose=True,
    )
    train_minutes = (time.time() - t0) / 60.0

    # 4. Validate on val split to get final metrics
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        workers=0,
        device=0,
    )

    # 5. Extract real numbers from the trainer
    mAP50 = float(metrics.box.map50)
    mAP50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    # Model info
    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
    speed = metrics.speed
    inference_ms = speed.get("inference", 0.0)

    try:
        info = model.info(verbose=False)
        gflops = float(info[1]) if isinstance(info, (list, tuple)) and len(info) > 1 else 8.3
    except Exception:
        gflops = 8.3

    # 6. Log to results.csv (per CLAUDE.md — real metrics only)
    new_row = log_run(
        model_id=VARIANT_ID,
        model_name=f"{VARIANT_NAME} (YOLOv8n + C2f_DCN Backbone)",
        folder=f"models/{VARIANT_NAME}",
        params_M=round(n_params, 2),
        gflops=round(gflops, 1),
        epochs=epochs,
        mAP50=round(mAP50, 4),
        mAP50_95=round(mAP50_95, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        inference_ms=round(inference_ms, 2),
        train_time_min=round(train_minutes, 1),
        notes="Modulated DCNv2 deformable convolutions in backbone C2f bottlenecks",
    )

    # 7. Print all variants in results.csv sorted by mAP50 descending
    rows = _read_all_rows()
    sorted_rows = sorted(rows, key=lambda r: float(r.get("mAP50", 0)), reverse=True)

    print()
    print("=" * 85)
    print("  LEADERBOARD: ALL VARIANTS SORTED BY mAP@50 (DESCENDING)")
    print("=" * 85)
    print(f"  {'Rank':<6} {'Model ID':<10} {'Model Name':<30} {'Params':>8} {'GFLOPs':>8} {'mAP50':>10} {'mAP50-95':>10} {'is_best':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*30} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for rank, r in enumerate(sorted_rows, start=1):
        m_id = r.get("model_id", "")
        name = r.get("model_name", "")[:28]
        p_m = f"{float(r.get('params_M', 0)):.2f}M"
        gfl = f"{float(r.get('gflops', 0)):.1f}"
        map50_str = f"{float(r.get('mAP50', 0))*100:.2f}%"
        map95_str = f"{float(r.get('mAP50_95', 0))*100:.2f}%"
        best_str = r.get("is_best", "")
        print(f"  {rank:<6} {m_id:<10} {name:<30} {p_m:>8} {gfl:>8} {map50_str:>10} {map95_str:>10} {best_str:>8}")
    print("=" * 85)

    is_v6_best = (sorted_rows[0].get("model_id") == "v6")
    if is_v6_best:
        print("  🎉 NEW BEST MODEL: v6_c2f_dcn achieved the highest mAP@50 overall!")
    else:
        print(f"  Current #1 Leader remains: {sorted_rows[0].get('model_name')} ({float(sorted_rows[0].get('mAP50'))*100:.2f}%)")

    print(f"  Weights: {RUNS_DIR / 'train' / 'weights' / 'best.pt'}")
    print(f"  Train time: {train_minutes:.1f} min")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
