"""
models/afb_yolo_11520574/train.py
=================================
Trains and evaluates AFB-YOLO (Lade, Bethi, Cenkeramaddi, IEEE Sensors J., 2026, 11520574)
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
