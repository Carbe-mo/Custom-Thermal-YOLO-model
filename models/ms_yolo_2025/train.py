"""
models/ms_yolo_2025/train.py
============================
Trains and evaluates MS-YOLO (MobileNetV4-Small UIB Backbone + SlideLoss;
Zhang et al., IEEE IJCNN, pp. 1-8, 2025)
for 100 epochs on the Thermal Security Surveillance dataset.
"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_and_eval_competing_papers import main

if __name__ == "__main__":
    main()
