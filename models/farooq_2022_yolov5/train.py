"""
models/farooq_2022_yolov5/train.py
==================================
Trains and evaluates Thermal-YOLO (YOLOv5 C3-CSP + SGD; Farooq, Shariff, Corcoran,
IEEE Trans. Intelligent Vehicles, vol. 8, no. 2, pp. 1130-1144, 2022)
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
