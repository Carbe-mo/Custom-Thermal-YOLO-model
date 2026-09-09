import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from common.modules.c2f_eca import register_c2f_eca
from common.modules.cbam_bottle import register_cbam_bottle
from common.modules.c2f_pconv import register_c2f_pconv
from common.modules.ghost_bottle import register_c2f_ghost
from common.modules.dual_eca import register_c2f_dual_eca
from common.modules.coord_att import register_c2f_coordatt
from common.modules.eca_mish import register_c2f_eca_mish

register_c2f_eca()
register_cbam_bottle()
register_c2f_pconv()
register_c2f_ghost()
register_c2f_dual_eca()
register_c2f_coordatt()
register_c2f_eca_mish()

runs = {
    "v0_baseline (Stock YOLOv8n)": "runs/v0_baseline/train/weights/best.pt",
    "v5b_c2f_eca (Leader)": "runs/v5b_c2f_eca/train/weights/best.pt",
    "v13_c2f_eca_distill (Tied Leader)": "runs/v13_c2f_eca_distill/train/weights/best.pt",
    "v7_c2f_cbam": "runs/v7_c2f_cbam/train/weights/best.pt",
    "v18_c2f_eca_mish": "runs/v18_c2f_eca_mish/train/weights/best.pt",
    "v15_c2f_dual_eca": "runs/v15_c2f_dual_eca/train/weights/best.pt",
    "v16_c2f_coordatt": "runs/v16_c2f_coordatt/train/weights/best.pt",
    "v14_c2f_ghost": "runs/v14_c2f_ghost/train/weights/best.pt",
    "v8_c2f_pconv": "runs/v8_c2f_pconv/train/weights/best.pt",
}

base_params = 3006428

print(f"{'Model Variant':<36} | {'Params':<11} | {'Delta vs Base':<18} | {'GFLOPs':<7} | {'Disk Size':<10}")
print("-" * 92)

for name, p in runs.items():
    pth = Path(p)
    if pth.exists():
        sz = os.path.getsize(pth) / (1024 * 1024)
        ckpt = torch.load(pth, map_location="cpu", weights_only=False)
        m = ckpt.get("model", None)
        params = sum(param.numel() for param in m.parameters()) if m is not None else 0
        diff = params - base_params
        diff_str = f"{diff:+d} ({diff/base_params*100:+.2f}%)" if diff != 0 else "0 (Baseline)"
        gflops = 8.1
        if "cbam" in name or "mish" in name or "dual" in name or "coord" in name:
            gflops = 8.2
        elif "ghost" in name:
            gflops = 7.9
        elif "pconv" in name:
            gflops = 7.0
        print(f"{name:<36} | {params:>9,} | {diff_str:<18} | {gflops:>5.1f} | {sz:>6.2f} MB")
