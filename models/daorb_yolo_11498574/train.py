"""
models/daorb_yolo_11498574/train.py
===================================
Trains and evaluates DA-ORB-YOLO / TMCR Edge-YOLO (IEEE Sensors J., 2026, 11498574)
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
