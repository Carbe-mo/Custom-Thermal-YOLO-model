"""
models/v34_early_fusion/train.py
======================================
Train Hybrid Dilated ECA-YOLO with 4-channel Early Fusion (RGB + Thermal Gray).

Strategy: Since Ultralytics' YOLO() doesn't expose ch=4, we:
1. Build the standard 3ch model from the v30 YAML
2. Replace Layer 0 conv in-place with a 4ch version (weight-surgically initialized)
3. Save as a custom checkpoint
4. Load that checkpoint for training

Requires: Run scripts/prepare_fusion_dataset.py first to create 4-channel images.
"""

from __future__ import annotations

import argparse
import sys
import time
import yaml
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.modules.c2f_eca import register_c2f_eca
from common.modules.c2f_eca_dilated import register_c2f_eca_dilated
from common.results_logger import log_run
from ultralytics import YOLO
from ultralytics.data.base import BaseDataset
import cv2

register_c2f_eca()
register_c2f_eca_dilated()

# Patch BaseDataset to preserve 4th channel (Alpha = Thermal Gray)
_orig_base_init = BaseDataset.__init__
def _patched_base_init(self, *args, **kwargs):
    _orig_base_init(self, *args, **kwargs)
    if self.channels == 4:
        self.cv2_flag = cv2.IMREAD_UNCHANGED
BaseDataset.__init__ = _patched_base_init

VARIANT_ID = "v34"
VARIANT_NAME = "v34_early_fusion"
V30_MODEL_YAML = PROJECT_ROOT / "models" / "v30_v27_cosine_100ep" / "model.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / VARIANT_NAME
DATA_ROOT = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def build_fusion_yaml() -> Path:
    """Create Ultralytics-compatible data YAML pointing to 4-channel fusion images."""
    yaml_path = DATA_ROOT / "fusion_data.yaml"
    fusion_dir = DATA_ROOT / "fusion"

    if not (fusion_dir / "train.txt").exists():
        raise FileNotFoundError(
            f"Missing {fusion_dir / 'train.txt'}. "
            f"Run scripts/prepare_fusion_dataset.py first."
        )

    data_cfg = {
        "path": str(fusion_dir),
        "train": str(fusion_dir / "train.txt"),
        "val": str(fusion_dir / "val.txt"),
        "test": str(fusion_dir / "test.txt"),
        "nc": 4,
        "names": {0: "person", 1: "trolley", 2: "bag", 3: "unattended"},
        "channels": 4,
    }

    yaml_path.write_text(yaml.dump(data_cfg, default_flow_style=False, sort_keys=False))
    print(f"[dataset] Wrote {yaml_path}")
    return yaml_path


def create_4ch_model() -> Path:
    """
    Create a 4-channel pretrained model checkpoint.

    1. Load stock yolov8n.pt (3ch, standard arch)
    2. Build our Dilated ECA model from v30 YAML (3ch)
    3. Transfer pretrained weights to our model
    4. Replace Layer 0 conv: 3ch → 4ch with weight surgery
    5. Save checkpoint
    """
    output_pt = WEIGHTS_DIR / "v34_early_fusion_4ch.pt"

    if output_pt.exists():
        print(f"[weights] 4ch checkpoint already exists at {output_pt}")
        return output_pt

    output_pt.parent.mkdir(parents=True, exist_ok=True)

    # Build our custom model from v30 YAML (3ch)
    print("[weights] Building Dilated ECA model from v30 YAML...")
    model = YOLO(str(V30_MODEL_YAML))

    # Load stock yolov8n pretrained weights (partial match)
    print("[weights] Loading yolov8n.pt pretrained weights...")
    stock = YOLO("yolov8n.pt")
    stock_sd = stock.model.state_dict()
    model_sd = model.model.state_dict()

    # Transfer matching weights (skip mismatched shapes)
    transferred = 0
    for k in model_sd:
        if k in stock_sd and stock_sd[k].shape == model_sd[k].shape:
            model_sd[k] = stock_sd[k]
            transferred += 1

    model.model.load_state_dict(model_sd)
    print(f"[weights] Transferred {transferred}/{len(model_sd)} pretrained layers")

    # Now replace Layer 0 conv: in_channels 3 → 4
    old_conv = model.model.model[0].conv  # nn.Conv2d(3, 16, 3, 2, 1)
    old_bn = model.model.model[0].bn
    old_act = model.model.model[0].act

    # Create new conv with 4 input channels
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    # Weight surgery: copy 3ch weights + mean for 4th channel
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    print(f"[weights] Layer 0 conv: {old_conv.weight.shape} → {new_conv.weight.shape}")

    # Replace in-place
    model.model.model[0].conv = new_conv

    # Update the yaml stored in the model to reflect ch=4
    model.model.yaml["channels"] = 4

    # Save checkpoint
    ckpt = {
        "model": model.model,
        "optimizer": None,
        "train_args": {
            "model": str(V30_MODEL_YAML),
            "data": "",
            "task": "detect",
        },
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt, str(output_pt))
    print(f"[weights] Saved 4ch checkpoint to {output_pt}")

    # Verify forward pass
    model.model.eval()
    x = torch.randn(1, 4, 640, 640)
    with torch.no_grad():
        out = model.model(x)
    print(f"[weights] Forward pass verified with 4ch input!")

    return output_pt


def train(epochs: int = 100, batch: int = 16, imgsz: int = 640) -> None:
    data_yaml = build_fusion_yaml()

    # Step 1: Create 4ch pretrained checkpoint (weight surgery)
    pretrained_4ch = create_4ch_model()

    # Step 2: Load from checkpoint (this gives us the full 4ch model)
    print(f"\n[train] Loading 4ch model from {pretrained_4ch}...")
    model = YOLO(str(pretrained_4ch))

    t0 = time.time()
    # Step 3: Train with identical hyperparameters to v30
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

    # Step 4: Evaluate on val and test
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
        model_name=f"{VARIANT_NAME} (Early Fusion 4ch + 100ep Cosine LR)",
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
        notes="Early fusion 4-channel [R,G,B,T_gray]. Weight surgery: 4th ch = mean(RGB) from yolov8n.pt",
    )

    log_run(
        model_id=f"{VARIANT_ID}_eval",
        model_name=f"{VARIANT_NAME} (Early Fusion) [Official Test]",
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
        notes="Official test split evaluation on 4-channel fusion dataset (108 images)",
    )

    print(f"\n[{VARIANT_NAME}] Val mAP50: {mAP50_val*100:.2f}% | Test mAP50: {mAP50_test*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
