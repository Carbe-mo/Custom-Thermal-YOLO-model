# Hybrid Dilated ECA-YOLO: Lightweight Thermal Infrared Object Detection for Edge Surveillance

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-blue.svg?style=flat)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An ultra-lightweight, zero-parameter-overhead object detection architecture specifically engineered to resolve **radiant heat diffusion**, **sensor static**, and **small-object edge preservation** in thermal infrared (TIR) surveillance imagery.

Developed through a systematic **29-iteration empirical research campaign**, this architecture introduces **stage-selective dilated convolutions** combined with **Efficient Channel Attention (ECA)**, achieving **87.89% test mAP@50** and **64.73% mAP@50-95** on unseen real-world thermal surveillance scenes—outperforming the stock YOLOv8n baseline by **+2.39pp mAP@50** and **+7.52pp Recall** under identical training dynamics.

---

## 🌟 Key Highlights

- **Zero-Parameter Overhead**: Fixed strictly at **3.01M parameters / 8.1 GFLOPs**, identical to stock YOLOv8n.
- **100% Pretrained Prior Transfer**: Preserves standard tensor dimensions, allowing full transfer from COCO-pretrained weights.
- **Stage-Selective Hybrid Dilation**: Keeps shallow stages (P2/P3) dense ($d=1$) for small pedestrians while expanding deep stages (P4/P5) via dilated convolutions ($d=2$, effective receptive field $5\times 5$) to capture composite heat signatures.
- **ECA Channel Calibration**: Fast 1D convolution ($k=3$) adaptively scales informative thermal features and suppresses microbolometer sensor noise.
- **Complete Open Research**: Includes all 38 experimental models, full research iteration notes (`notes.md`), and quantitative benchmarking tables.

---

## 📊 Benchmark Results (Official Held-Out Test Split)

Evaluated on official held-out test split (`data/splits/test.txt`, 108 unseen thermal surveillance scenes) under strictly controlled identical conditions (100 Epochs, Cosine Annealing, `close_mosaic=20`):

| Model Architecture | Params (M) | GFLOPs | Precision (%) | Recall (%) | **mAP@50 (%)** | **mAP@50-95 (%)** | Checkpoint Size | Net Gain vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 👑 **Hybrid Dilated ECA (`v30`)** | **3.01** | **8.1** | **87.22** | **83.38** | **87.89** | **64.73** | **5.97 MB** | 🚀 **+2.39pp (+2.18pp)** |
| **Stock YOLOv8n (`v0b`, 100ep)** | 3.01 | 8.1 | 86.31 | 75.86 | 85.50 | 62.55 | 5.97 MB | *Controlled Baseline* |
| **Stock YOLOv8n (`v0`, 50ep)** | 3.01 | 8.1 | 82.74 | 83.35 | 86.22 | 62.92 | 5.96 MB | - |
| **Pure C2f_ECA (`v5b`, 50ep)** | 3.01 | 8.1 | 87.22 | 80.77 | 87.39 | 63.41 | 5.97 MB | +1.17pp |
| **ECA SPPF Neck (`v31`, 75ep)** | 3.01 | 8.1 | 81.85 | **84.95** | 87.13 | 63.34 | 5.97 MB | 🎯 *Peak Recall (84.95%)* |
| **Linear Mixup (`v32`, 75ep)** | 3.01 | 8.1 | **90.80** | 78.36 | 86.35 | 63.65 | 5.97 MB | 🎯 *Peak Precision (90.80%)* |

---

## 🏗️ Architecture Overview

```
                                  BACKBONE TOPOLOGY
                                  =================
  Thermal Input (640x640)
         │
         ▼
  [Layer 0-1: Stem Convs (P1, P2)]
         │
         ▼
  [Layer 2: C2f_ECA (P2/4)] ──────────> Dense 3x3 Conv + ECA (k=3)  [Preserves small edge priors]
         │
         ▼
  [Layer 4: C2f_ECA (P3/8)] ──────────> Dense 3x3 Conv + ECA (k=3)  [Preserves small Person/Bag]
         │
         ▼
  [Layer 6: C2f_ECA_Dilated (P4/16)] ──> Dilated 3x3 (d=2, RF=5x5) + ECA [Captures medium blooms]
         │
         ▼
  [Layer 8: C2f_ECA_Dilated (P5/32)] ──> Dilated 3x3 (d=2, RF=5x5) + ECA [Panoramic context]
         │
         ▼
  [Layer 9: SPPF (1024, 5)]
         │
         ▼
  [Layers 10-21: Clean Linear PANet Neck] ──> [Layer 22: Decoupled Detect Head]
```

### Micro-Architecture: Residual Bottleneck Innovation
1. **Shallow Bottlenecks (P2, P3)**: Standard $3\times 3$ Conv $\rightarrow$ Standard $3\times 3$ Conv $\rightarrow$ **ECA ($k=3$)** $\rightarrow$ Shortcut Addition.
2. **Deep Bottlenecks (P4, P5)**: Standard $3\times 3$ Conv $\rightarrow$ **Dilated $3\times 3$ Conv ($d=2$, padding $p=2$)** $\rightarrow$ **ECA ($k=3$)** $\rightarrow$ Shortcut Addition.

---

## 📁 Repository Structure

```
thermal-yolo/
├── common/                        # Core architectural modules & plumbing
│   ├── modules/
│   │   ├── c2f_eca.py             # C2f with standard ECA channel attention
│   │   ├── c2f_eca_dilated.py     # Dilated C2f_ECA module (d=2)
│   │   ├── eca.py                 # Efficient Channel Attention (1D conv)
│   │   ├── sppf_eca.py            # SPPF with post-pyramid ECA gating
│   │   └── concat_eca.py          # Cross-scale concat channel gating
│   ├── dataset.py                 # Thermal dataset YAML builder & split resolver
│   └── results_logger.py          # Automated metric logger (CSV & Excel)
│
├── models/                        # 38 Model variants & training scripts
│   ├── v0_baseline/               # Stock YOLOv8n baseline
│   ├── v0b_baseline_100ep/        # Controlled 100-epoch baseline
│   ├── v5b_c2f_eca/               # Base ECA model
│   ├── v21_dilated_hybrid/        # First hybrid dilated architecture
│   ├── v27_v21_cosine/            # 75-epoch cosine hybrid model
│   ├── v30_v27_cosine_100ep/      # 👑 Champion Model (87.89% mAP50)
│   └── ...                        # Full ablation spectrum (v1 to v33)
│
├── weights/                       # Pretrained champion model weights
│   └── v30_champion_best.pt       # Ready-to-use trained weights (5.97 MB)
│
├── results/                       # Official evaluation metrics & logs
│   ├── results.csv                # Complete database of all experimental runs
│   └── results_highlighted.xlsx   # Color-formatted evaluation spreadsheet
│
├── scripts/                       # Automated execution and comparison runners
│   ├── run_v21_to_v25_sequential.py
│   ├── run_v21_combinations.py
│   ├── run_v27_next_combinations.py
│   └── run_v33_and_baseline_100ep.py
│
├── data/splits/                   # Official split partitions
│   ├── train.txt                  # 746 training image paths
│   ├── val.txt                    # 213 validation image paths
│   └── test.txt                   # 108 held-out test image paths
│
└── notes.md                       # Comprehensive log of Iterations 0 through 29
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone <your-repo-url>.git
cd thermal-yolo
pip install ultralytics torch torchvision
```

### 2. Inference with Champion Model
```python
from ultralytics import YOLO
from common.modules.c2f_eca import register_c2f_eca
from common.modules.c2f_eca_dilated import register_c2f_eca_dilated

# Register custom architectural modules
register_c2f_eca()
register_c2f_eca_dilated()

# Load trained champion checkpoint
model = YOLO("weights/v30_champion_best.pt")

# Run inference on a thermal image
results = model.predict("path/to/thermal_image.jpg", conf=0.25, imgsz=640)
results[0].show()
```

### 3. Training from Scratch / Pretrained
```bash
# Train the champion v30 model (100 Epochs Cosine Annealing)
python -m models.v30_v27_cosine_100ep.train --epochs 100 --batch 16
```

### 4. Official Test Split Evaluation
```bash
python scripts/evaluate_all.py
```

---

## 📜 Citation

If you find this architecture, empirical findings, or benchmark useful in your research, please cite our manuscript:

```bibtex
@article{thermal_hybrid_yolo_2026,
  author    = {Satyam and Contributors},
  title     = {Overcoming Radiant Heat Diffusion: Stage-Selective Dilated Attention in Lightweight YOLO for Real-Time Edge Thermal Surveillance},
  journal   = {IEEE Sensors Journal},
  year      = {2026},
  note      = {Under Review}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
