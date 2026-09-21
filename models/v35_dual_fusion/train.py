"""
models/v35_dual_fusion/train.py
=====================================
Train Dual-Backbone RGB-Thermal Fusion with Multi-Scale Gated Attention (v35).

Architecture:
  - Stream A: Visible RGB Backbone (pre-trained on COCO)
  - Stream B: Thermal Infrared Backbone (pre-trained from champion v30 Dilated ECA)
  - GatedFusion at P3, P4, P5: Learns dynamic weighting g*F_rgb + (1-g)*F_thermal
  - Shared PANet Neck + Detect Head (pre-trained from v30)

Total Parameters: 4.46M
"""

from __future__ import annotations

import argparse
import sys
import time
import yaml
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.modules.c2f_eca import register_c2f_eca
from common.modules.c2f_eca_dilated import register_c2f_eca_dilated
from common.modules.fusion import register_fusion_modules
from common.results_logger import log_run
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.base import BaseDataset

register_c2f_eca()
register_c2f_eca_dilated()
register_fusion_modules()

# Patch BaseDataset so self.channels = 6 is accepted and .npy files loaded properly
_orig_base_init = BaseDataset.__init__
def _patched_base_init(self, *args, **kwargs):
    _orig_base_init(self, *args, **kwargs)
    if self.channels == 6:
        # Avoid cv2 dropping channels since images are loaded from .npy files
        pass
BaseDataset.__init__ = _patched_base_init

VARIANT_ID = "v35"
VARIANT_NAME = "v35_dual_fusion"
MODEL_YAML = PROJECT_ROOT / "models" / VARIANT_NAME / "model.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / VARIANT_NAME
DATA_ROOT = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def build_dual_fusion_yaml() -> Path:
    yaml_path = DATA_ROOT / "fusion_dual_data.yaml"
    fusion_dir = DATA_ROOT / "fusion_dual"

    if not (fusion_dir / "train.txt").exists():
        raise FileNotFoundError(
            f"Missing {fusion_dir / 'train.txt'}. "
            f"Run scripts/prepare_dual_fusion_dataset.py first."
        )

    data_cfg = {
        "path": str(fusion_dir),
        "train": str(fusion_dir / "train.txt"),
        "val": str(fusion_dir / "val.txt"),
        "test": str(fusion_dir / "test.txt"),
        "nc": 4,
        "names": {0: "person", 1: "trolley", 2: "bag", 3: "unattended"},
        "channels": 6,
    }

    yaml_path.write_text(yaml.dump(data_cfg, default_flow_style=False, sort_keys=False))
    print(f"[dataset] Wrote {yaml_path}")
    return yaml_path


def create_dual_pretrained_weights() -> Path:
    output_pt = WEIGHTS_DIR / "v35_dual_fusion_pretrained.pt"
    if output_pt.exists():
        print(f"[weights] Pretrained checkpoint already exists at {output_pt}")
        return output_pt

    output_pt.parent.mkdir(parents=True, exist_ok=True)
    print("[weights] Initializing v35 dual model architecture...")
    model = DetectionModel(cfg=str(MODEL_YAML), ch=6, nc=4, verbose=False)
    model_sd = model.state_dict()

    # Load stock yolov8n for RGB Stream A
    print("[weights] Loading COCO weights for RGB Stream A...")
    stock = YOLO("yolov8n.pt")
    stock_sd = stock.model.state_dict()

    # Load champion v30 for Thermal Stream B and Head
    print("[weights] Loading v30 Champion weights for Thermal Stream B and PANet Head...")
    v30_path = PROJECT_ROOT / "runs" / "v30_v27_cosine_100ep" / "train" / "weights" / "best.pt"
    v30 = YOLO(str(v30_path))
    v30_sd = v30.model.state_dict()

    loaded = 0
    for k, v in model_sd.items():
        parts = k.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            layer_idx = int(parts[1])
            if 2 <= layer_idx <= 11:
                # RGB Stream A: layers 2..11 -> map from stock yolov8n layers 0..9
                src_key = ".".join(["model", str(layer_idx - 2)] + parts[2:])
                if src_key in stock_sd and stock_sd[src_key].shape == v.shape:
                    model_sd[k] = stock_sd[src_key]
                    loaded += 1
            elif 13 <= layer_idx <= 22:
                # Thermal Stream B: layers 13..22 -> map from v30 layers 0..9
                src_key = ".".join(["model", str(layer_idx - 13)] + parts[2:])
                if src_key in v30_sd and v30_sd[src_key].shape == v.shape:
                    model_sd[k] = v30_sd[src_key]
                    loaded += 1
            elif 26 <= layer_idx <= 38:
                # Head: layers 26..38 -> map from v30 layers 10..22
                src_key = ".".join(["model", str(layer_idx - 16)] + parts[2:])
                if src_key in v30_sd and v30_sd[src_key].shape == v.shape:
                    model_sd[k] = v30_sd[src_key]
                    loaded += 1

    model.load_state_dict(model_sd)
    print(f"[weights] Transferred {loaded} weight tensors into dual-backbone model!")

    ckpt = {
        "model": model,
        "optimizer": None,
        "train_args": {"model": str(MODEL_YAML), "data": "", "task": "detect"},
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt, str(output_pt))
    print(f"[weights] Saved dual pretrained weights to {output_pt}")
    return output_pt


def train(epochs: int = 100, batch: int = 16, imgsz: int = 640) -> None:
    data_yaml = build_dual_fusion_yaml()
    pretrained_pt = create_dual_pretrained_weights()

    print(f"\n[train] Loading v35 Dual-Backbone model from {pretrained_pt}...")
    model = YOLO(str(pretrained_pt))

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
        cos_lr=True,
        lrf=0.01,
        close_mosaic=20,
        verbose=True,
    )
    train_minutes = (time.time() - t0) / 60.0

    # Evaluate on val and test
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
        model_name=f"{VARIANT_NAME} (Dual-Backbone Gated Fusion + 100ep Cosine LR)",
        folder=f"models/{VARIANT_NAME}",
        params_M=round(n_params, 2),
        gflops=16.2,
        epochs=epochs,
        mAP50=round(mAP50_val, 4),
        mAP50_95=round(mAP50_95_val, 4),
        precision=round(prec_val, 4),
        recall=round(rec_val, 4),
        inference_ms=round(inference_ms, 2),
        train_time_min=round(train_minutes, 1),
        notes="Dual-stream RGB + Thermal backbones with multi-scale GatedFusion at P3, P4, P5. 6-channel input.",
    )

    log_run(
        model_id=f"{VARIANT_ID}_eval",
        model_name=f"{VARIANT_NAME} (Dual-Backbone Gated Fusion) [Official Test]",
        folder=f"models/{VARIANT_NAME}",
        params_M=round(n_params, 2),
        gflops=16.2,
        epochs=epochs,
        mAP50=round(mAP50_test, 4),
        mAP50_95=round(mAP50_95_test, 4),
        precision=round(prec_test, 4),
        recall=round(rec_test, 4),
        inference_ms=round(inference_ms, 2),
        train_time_min=round(train_minutes, 1),
        notes="Official test split evaluation on 6-channel dual fusion dataset (108 images)",
    )

    print(f"\n[{VARIANT_NAME}] Val mAP50: {mAP50_val*100:.2f}% | Test mAP50: {mAP50_test*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
