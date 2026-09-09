"""
scripts/evaluate_all.py — Unified Test-Set Evaluation Across All Model Checkpoints
=================================================================================

Reloads every trained model checkpoint under `runs/` and evaluates all of them on the
exact same held-out test split (`data/splits/test.txt`, 108 images) under identical
deterministic evaluation conditions.

Logs each official evaluation run to `results/results.csv` and updates
`results/results_highlighted.xlsx`.

Usage:
    cd thermal-yolo
    python -m scripts.evaluate_all
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# --- ensure project root is importable --------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Register all custom modules so all checkpoints deserialize seamlessly
from common.modules.thermal_stem import register_thermal_stem
from common.modules.cbam import register_cbam
from common.modules.bifpn import register_bifpn
from common.modules.c2f_eca import register_c2f_eca
from common.modules.c2f_dcn import register_c2f_dcn
from common.modules.cbam_bottle import register_cbam_bottle
from common.modules.c2f_pconv import register_c2f_pconv
from common.modules.eca_adaptive import register_c2f_eca_adaptive
from common.modules.ghost_bottle import register_c2f_ghost
from common.modules.dual_eca import register_c2f_dual_eca
from common.modules.coord_att import register_c2f_coordatt
from common.modules.eca_mish import register_c2f_eca_mish

register_thermal_stem()
register_cbam()
register_bifpn()
register_c2f_eca()
register_c2f_dcn()
register_cbam_bottle()
register_c2f_pconv()
register_c2f_eca_adaptive()
register_c2f_ghost()
register_c2f_dual_eca()
register_c2f_coordatt()
register_c2f_eca_mish()

from common.dataset import build_yaml
from common.results_logger import log_run, _read_all_rows
from ultralytics import YOLO

# Complete list of all model checkpoints to evaluate
MODELS_TO_EVAL = [
    {
        "id": "v0",
        "name": "v0_baseline (Stock YOLOv8n)",
        "folder": "models/v0_baseline",
        "ckpt": PROJECT_ROOT / "runs" / "v0_baseline" / "train" / "weights" / "best.pt",
        "gflops": 8.1,
    },
    {
        "id": "v1",
        "name": "v1_thermal_stem (Thermal Stem)",
        "folder": "models/v1_thermal_stem",
        "ckpt": PROJECT_ROOT / "runs" / "v1_thermal_stem" / "train" / "weights" / "best.pt",
        "gflops": 8.2,
    },
    {
        "id": "v2",
        "name": "v2_cbam_backbone (CBAM Attention)",
        "folder": "models/v2_cbam_backbone",
        "ckpt": PROJECT_ROOT / "runs" / "v2_cbam_backbone" / "train" / "weights" / "best.pt",
        "gflops": 8.3,
    },
    {
        "id": "v3",
        "name": "v3_bifpn_neck (BiFPN Neck)",
        "folder": "models/v3_bifpn_neck",
        "ckpt": PROJECT_ROOT / "runs" / "v3_bifpn_neck" / "train" / "weights" / "best.pt",
        "gflops": 8.5,
    },
    {
        "id": "v4",
        "name": "v4_rgb_distill (RGB Distillation)",
        "folder": "models/v4_rgb_distill",
        "ckpt": PROJECT_ROOT / "runs" / "v4_rgb_distill" / "train" / "weights" / "best.pt",
        "gflops": 8.1,
    },
    {
        "id": "v5",
        "name": "v5_c2f_eca (C2f_ECA Scratch)",
        "folder": "models/v5_c2f_eca",
        "ckpt": PROJECT_ROOT / "runs" / "v5_c2f_eca" / "train" / "weights" / "best.pt",
        "gflops": 8.1,
    },
    {
        "id": "v5b",
        "name": "v5b_c2f_eca (C2f_ECA Pretrained)",
        "folder": "models/v5b_c2f_eca",
        "ckpt": PROJECT_ROOT / "runs" / "v5b_c2f_eca" / "train" / "weights" / "best.pt",
        "gflops": 8.1,
    },
    {
        "id": "v6",
        "name": "v6_c2f_dcn (C2f_DCN Backbone)",
        "folder": "models/v6_c2f_dcn",
        "ckpt": PROJECT_ROOT / "runs" / "v6_c2f_dcn" / "train" / "weights" / "best.pt",
        "gflops": 8.3,
    },
    {
        "id": "v7",
        "name": "v7_c2f_cbam (CBAM in Bottleneck)",
        "folder": "models/v7_c2f_cbam",
        "ckpt": PROJECT_ROOT / "runs" / "v7_c2f_cbam" / "train" / "weights" / "best.pt",
        "gflops": 8.2,
    },
    {
        "id": "v8",
        "name": "v8_c2f_pconv (FasterNet PConv)",
        "folder": "models/v8_c2f_pconv",
        "ckpt": PROJECT_ROOT / "runs" / "v8_c2f_pconv" / "train" / "weights" / "best.pt",
        "gflops": 7.0,
    },
    {
        "id": "v9",
        "name": "v9_c2f_eca_all (Full-Network C2f_ECA)",
        "folder": "models/v9_c2f_eca_all",
        "ckpt": PROJECT_ROOT / "runs" / "v9_c2f_eca_all" / "train" / "weights" / "best.pt",
        "gflops": 8.2,
    },
    {
        "id": "v10",
        "name": "v10_c2f_eca_adaptive (Adaptive ECA)",
        "folder": "models/v10_c2f_eca_adaptive",
        "ckpt": PROJECT_ROOT / "runs" / "v10_c2f_eca_adaptive" / "train" / "weights" / "best.pt",
        "gflops": 8.2,
    },
    {
        "id": "v11",
        "name": "v11_c2f_eca_deep (Deepened C2f_ECA)",
        "folder": "models/v11_c2f_eca_deep",
        "ckpt": PROJECT_ROOT / "runs" / "v11_c2f_eca_deep" / "train" / "weights" / "best.pt",
        "gflops": 9.3,
    },
    {
        "id": "v12",
        "name": "v12_stem_eca (ThermalStem + C2f_ECA)",
        "folder": "models/v12_stem_eca",
        "ckpt": PROJECT_ROOT / "runs" / "v12_stem_eca" / "train" / "weights" / "best.pt",
        "gflops": 8.6,
    },
    {
        "id": "v13",
        "name": "v13_c2f_eca_distill (C2f_ECA + Distill)",
        "folder": "models/v13_c2f_eca_distill",
        "ckpt": PROJECT_ROOT / "runs" / "v13_c2f_eca_distill" / "train" / "weights" / "best.pt",
        "gflops": 8.1,
    },
    {
        "id": "v14",
        "name": "v14_c2f_ghost (GhostConv in Bottleneck)",
        "folder": "models/v14_c2f_ghost",
        "ckpt": PROJECT_ROOT / "runs" / "v14_c2f_ghost" / "train" / "weights" / "best.pt",
        "gflops": 7.9,
    },
    {
        "id": "v15",
        "name": "v15_c2f_dual_eca (Dual ECA in Bottleneck)",
        "folder": "models/v15_c2f_dual_eca",
        "ckpt": PROJECT_ROOT / "runs" / "v15_c2f_dual_eca" / "train" / "weights" / "best.pt",
        "gflops": 8.2,
    },
    {
        "id": "v16",
        "name": "v16_c2f_coordatt (Coordinate Attention)",
        "folder": "models/v16_c2f_coordatt",
        "ckpt": PROJECT_ROOT / "runs" / "v16_c2f_coordatt" / "train" / "weights" / "best.pt",
        "gflops": 8.2,
    },
    {
        "id": "v17",
        "name": "v17_c2f_eca_scratch (Scratch 100 Epochs)",
        "folder": "models/v17_c2f_eca_scratch",
        "ckpt": PROJECT_ROOT / "runs" / "v17_c2f_eca_scratch" / "train" / "weights" / "best.pt",
        "gflops": 8.1,
    },
    {
        "id": "v18",
        "name": "v18_c2f_eca_mish (Mish ECA Bottleneck)",
        "folder": "models/v18_c2f_eca_mish",
        "ckpt": PROJECT_ROOT / "runs" / "v18_c2f_eca_mish" / "train" / "weights" / "best.pt",
        "gflops": 8.2,
    },
    {
        "id": "v19",
        "name": "v19_c2f_eca_inception (Inception Strip Convs)",
        "folder": "models/v19_c2f_eca_inception",
        "ckpt": PROJECT_ROOT / "runs" / "v19_c2f_eca_inception" / "train" / "weights" / "best.pt",
        "gflops": 9.2,
    },
    {
        "id": "v20",
        "name": "v20_c2f_eca_dilated (Dilated d=2 ECA)",
        "folder": "models/v20_c2f_eca_dilated",
        "ckpt": PROJECT_ROOT / "runs" / "v20_c2f_eca_dilated" / "train" / "weights" / "best.pt",
        "gflops": 8.1,
    },
]


def evaluate_all() -> None:
    data_yaml = build_yaml("thermal")
    print("=" * 85)
    print(" UNIFIED TEST SPLIT EVALUATION (data/splits/test.txt) ACROSS ALL RUNS")
    print("=" * 85)

    for entry in MODELS_TO_EVAL:
        model_id = entry["id"]
        model_name = entry["name"]
        folder = entry["folder"]
        ckpt_path = entry["ckpt"]
        gflops = entry["gflops"]

        if not ckpt_path.exists():
            print(f"[-] Checkpoint missing for {model_id}: {ckpt_path}, skipping...")
            continue

        print(f"\n[+] Evaluating {model_id}: {model_name} from {ckpt_path.relative_to(PROJECT_ROOT)}...")
        model = YOLO(str(ckpt_path))

        # Evaluate on the official held-out test split
        t0 = time.time()
        metrics = model.val(
            data=str(data_yaml),
            split="test",
            workers=0,
            device=0,
            verbose=False,
        )
        eval_time = (time.time() - t0)

        mAP50 = float(metrics.box.map50)
        mAP50_95 = float(metrics.box.map)
        prec = float(metrics.box.mp)
        rec = float(metrics.box.mr)
        speed = metrics.speed
        inference_ms = speed.get("inference", 0.0)

        n_params = sum(p.numel() for p in model.model.parameters()) / 1e6

        eval_model_id = f"{model_id}_eval"
        eval_model_name = f"{model_name} [Official Test]"
        notes = "official test split evaluation on held-out data/splits/test.txt (108 images)"

        log_run(
            model_id=eval_model_id,
            model_name=eval_model_name,
            folder=folder,
            params_M=round(n_params, 2),
            gflops=gflops,
            epochs=50,
            mAP50=round(mAP50, 4),
            mAP50_95=round(mAP50_95, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            inference_ms=round(inference_ms, 2),
            train_time_min=0.0,
            notes=notes,
        )

    # Print Final Sorted Leaderboard
    all_rows = _read_all_rows()
    eval_rows = [r for r in all_rows if r.get("model_id", "").endswith("_eval")]
    eval_rows.sort(key=lambda r: float(r.get("mAP50", 0.0) or 0.0), reverse=True)

    print("\n" + "=" * 95)
    print("  🏆 OFFICIAL FINAL LEADERBOARD: TEST SPLIT EVALUATION (data/splits/test.txt)")
    print("=" * 95)
    print(f"  {'Rank':<6} {'Model ID':<10} {'Model Name':<38} {'Params':>8} {'GFLOPs':>8} {'Precision':>10} {'Recall':>8} {'mAP50':>9} {'mAP50-95':>10}")
    print("  " + "-" * 6 + " " + "-" * 10 + " " + "-" * 38 + " " + "-" * 8 + " " + "-" * 8 + " " + "-" * 10 + " " + "-" * 8 + " " + "-" * 9 + " " + "-" * 10)

    for rank, r in enumerate(eval_rows, start=1):
        m_id = r.get("model_id", "")
        m_name = r.get("model_name", "")[:38]
        params = f"{float(r.get('params_M', 0)):.2f}M"
        gflops_str = f"{float(r.get('gflops', 0)):.1f}"
        prec = f"{float(r.get('precision', 0))*100:.1f}%"
        rec = f"{float(r.get('recall', 0))*100:.1f}%"
        map50 = f"{float(r.get('mAP50', 0))*100:.2f}%"
        map50_95 = f"{float(r.get('mAP50_95', 0))*100:.2f}%"
        print(f"  {rank:<6} {m_id:<10} {m_name:<38} {params:>8} {gflops_str:>8} {prec:>10} {rec:>8} {map50:>9} {map50_95:>10}")

    print("=" * 95)


if __name__ == "__main__":
    evaluate_all()
