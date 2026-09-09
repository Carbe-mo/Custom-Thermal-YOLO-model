# CLAUDE.md — Project Rules for thermal-yolo

## Project Overview

Multi-modal (Thermal + RGB) object detection research project using **Ultralytics YOLOv8**
as the base library. The goal is to iteratively improve detection of 4 classes
(`person`, `trolley`, `bag`, `unattended`) on paired FLIR thermal/visible imagery through
a series of controlled architectural experiments.

---

## Rules (strictly enforced)

### 1. Model Variant Isolation

Every model variant lives in its own `models/<id>_<name>/` folder with its own `model.yaml`
and `train.py`. **Never edit one variant's folder while working on another.**

```
models/
├── v0_baseline/          # Unmodified YOLOv8 baseline
│   ├── model.yaml
│   └── train.py
├── v1_thermal_stem/      # Custom thermal-aware input stem
│   ├── model.yaml
│   └── train.py
├── v2_cbam_backbone/     # CBAM attention in the backbone
│   ├── model.yaml
│   └── train.py
├── v3_bifpn_neck/        # BiFPN neck replacement
│   ├── model.yaml
│   └── train.py
├── v4_rgb_distill/       # Knowledge distillation from RGB teacher
│   ├── model.yaml
│   └── train.py
└── v5_combo_best/        # Best combination of above improvements
    ├── model.yaml
    └── train.py
```

### 2. Reusable Modules

Reusable blocks (attention modules, custom necks/stems, losses) go in `common/modules/`
and are **imported, never duplicated**.

```
common/modules/
├── cbam.py             # Convolutional Block Attention Module
├── bifpn.py            # Bi-directional Feature Pyramid Network
├── thermal_stem.py     # Thermal-aware input stem
└── distill_loss.py     # Knowledge distillation loss
```

### 3. Single Canonical Splits

All variants train/val/test on the **same split files** in `data/splits/` — never
regenerate splits per-model. The split files are created once and shared by every
experiment.

### 4. Automated Results Logging

Every `train.py` must call `common.results_logger.log_run(...)` at the end with **real
metrics pulled from the trainer**, not hand-typed numbers. The logger writes to
`results/results.csv`.

### 5. Append-Only Results

`results/results.csv` is **append-only**, written **only** by `results_logger.py` — never
hand-edited. Each row records:

| Field             | Source                                    |
|-------------------|-------------------------------------------|
| `timestamp`       | Automatic (UTC)                           |
| `variant`         | e.g. `v0_baseline`                        |
| `epochs`          | From trainer config                       |
| `precision`       | From `trainer.metrics`                    |
| `recall`          | From `trainer.metrics`                    |
| `mAP50`           | From `trainer.metrics`                    |
| `mAP50_95`        | From `trainer.metrics`                    |
| `params_M`        | From model summary                        |
| `gflops`          | From model summary                        |
| `best_weight`     | Path to `best.pt`                         |

---

## Base Library

- **Ultralytics YOLOv8** (`pip install ultralytics`)
- All custom YAML configs extend `yolov8n.yaml` or `yolov8s.yaml`
- Custom modules are registered via Ultralytics' module registration system

## Dataset

- 4 classes: `person` (0), `trolley` (1), `bag` (2), `unattended` (3)
- Source: FLIR paired thermal/visible imagery in `data/thermal/` and `data/rgb/`
- Labels: YOLO-format `.txt` files
- Splits: `data/splits/train.txt`, `data/splits/val.txt`, `data/splits/test.txt`
