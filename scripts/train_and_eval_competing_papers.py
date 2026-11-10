"""
scripts/train_and_eval_competing_papers.py
==========================================
Trains and evaluates four competing literature thermal YOLO architectures on the
Thermal Security Surveillance dataset (746 train / 213 val / 108 test frames):

  1. Farooq et al. (2022) — Thermal-YOLO (YOLOv5 C3-CSP + SGD)
     M. A. Farooq, W. Shariff, P. Corcoran, "Evaluation of thermal imaging on embedded
     GPU platforms for application in vehicular assistance systems", IEEE Transactions
     on Intelligent Vehicles, vol. 8, no. 2, pp. 1130-1144, 2022.

  2. Zhang et al. (2025) — MS-YOLO (MobileNetV4-Small UIB Backbone + SlideLoss)
     J. Zhang et al., "MS-YOLO: Infrared Object Detection for Edge Deployment via
     MobileNetV4 and SlideLoss", IEEE IJCNN, pp. 1-8, 2025.

  3. Lade et al. (2026) — AFB-YOLO (IEEE Xplore 11520574, DOI 10.1109/JSEN.2026.3691670)
     S. Lade, Y. Bethi, L. R. Cenkeramaddi, "Thermal Imaging-based Perception for Automotive
     Edge AI Applications using a Lightweight Custom YOLO Model", IEEE Sensors J., 2026.
     (Attention Fusion Block: 5x5 DWConv inverted bottleneck + ECA + Coordinate Attention)

  4. TMCR Edge-YOLO (2026) — DA-ORB-YOLO (IEEE Xplore 11498574, DOI 10.1109/JSEN.2026.3686674)
     "TMCR Edge-YOLO: Thermal Sensing-based Mobility-Context Recognition on the Edge",
     IEEE Sensors J., 2026.
     (Dual-Attention Optimized Residual Block: 5x5 DWConv inverted bottleneck + ECA + Spatial Attention)

Compares all four models directly against our Proposed Hybrid Dilated ECA-YOLO (v30) on the official
held-out test split (108 thermal frames, 390 instances).
"""

from __future__ import annotations
import csv
import shutil
import sys
import time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.modules.c2f_eca import register_c2f_eca
from common.modules.c2f_eca_dilated import register_c2f_eca_dilated
from common.modules.afb_daorb import register_afb_daorb
from common.modules.ms_yolo import register_ms_yolo, set_slide_loss_enabled
from common.results_logger import log_run

register_c2f_eca()
register_c2f_eca_dilated()
register_afb_daorb()
register_ms_yolo()

from ultralytics import YOLO

DATA_YAML = REPO_ROOT / "data" / "thermal_data.yaml"
RUNS_DIR = REPO_ROOT / "runs"

COMPETING_MODELS = [
    {
        "key": "farooq_2022_yolov5",
        "display_name": "Thermal-YOLO (YOLOv5 + SGD) [Farooq et al., 2022]",
        "yaml_path": REPO_ROOT / "models" / "farooq_2022_yolov5" / "model.yaml",
        "params_m": 2.51,
        "gflops": 7.2,
        "optimizer": "SGD",
        "use_slide_loss": False,
    },
    {
        "key": "ms_yolo_2025",
        "display_name": "MS-YOLO (MobileNetV4 + SlideLoss) [Zhang et al., 2025]",
        "yaml_path": REPO_ROOT / "models" / "ms_yolo_2025" / "model.yaml",
        "params_m": 2.95,
        "gflops": 6.7,
        "optimizer": "auto",
        "use_slide_loss": True,
    },
    {
        "key": "afb_yolo_11520574",
        "display_name": "AFB-YOLO (ECA + CoordAtt) [Lade et al., 2026 / 11520574]",
        "yaml_path": REPO_ROOT / "models" / "afb_yolo_11520574" / "model.yaml",
        "params_m": 2.78,
        "gflops": 7.7,
        "optimizer": "auto",
        "use_slide_loss": False,
    },
    {
        "key": "daorb_yolo_11498574",
        "display_name": "DA-ORB-YOLO (ECA + SpatialAtt) [TMCR Edge-YOLO, 2026 / 11498574]",
        "yaml_path": REPO_ROOT / "models" / "daorb_yolo_11498574" / "model.yaml",
        "params_m": 2.77,
        "gflops": 7.7,
        "optimizer": "auto",
        "use_slide_loss": False,
    },
]


def measure_gpu_latency_ms(weights_path: Path, imgsz: int = 640, warmup: int = 20, runs: int = 100) -> float:
    from ultralytics.nn.tasks import load_checkpoint
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = load_checkpoint(str(weights_path))[0].to(dev).float().eval()
    x = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32, device=dev)
    with torch.no_grad():
        for _ in range(warmup):
            _ = m(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
        lats = []
        for _ in range(runs):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = m(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000.0)
    lats.sort()
    trim = max(1, int(runs * 0.1))
    return float(sum(lats[trim:-trim]) / len(lats[trim:-trim]))


def main():
    results_summary = []

    for item in COMPETING_MODELS:
        key = item["key"]
        name = item["display_name"]
        yaml_path = item["yaml_path"]
        best_pt = RUNS_DIR / key / "train" / "weights" / "best.pt"

        set_slide_loss_enabled(bool(item.get("use_slide_loss", False)))

        if not best_pt.exists():
            print(f"\n==================================================================")
            print(f"Training {name} for 100 epochs on Thermal Security Dataset...")
            print(f"==================================================================", flush=True)
            model = YOLO(str(yaml_path))
            model.train(
                data=str(DATA_YAML),
                pretrained=str(REPO_ROOT / "yolov8n.pt"),
                epochs=100,
                patience=100,
                batch=16,
                imgsz=640,
                device="0",
                workers=0,
                optimizer=item.get("optimizer", "auto"),
                project=str(RUNS_DIR / key),
                name="train",
                exist_ok=True,
                cos_lr=True,
                close_mosaic=20,
                seed=0,
                deterministic=True,
                verbose=True,
            )
        else:
            print(f"\nFound existing trained checkpoint for {name}: {best_pt}")

        set_slide_loss_enabled(False)

        # Copy trained best.pt into models/<key>/best.pt and weights/<key>_best.pt
        model_dir_pt = REPO_ROOT / "models" / key / "best.pt"
        weights_dir_pt = REPO_ROOT / "weights" / f"{key}_best.pt"
        weights_dir_pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, model_dir_pt)
        shutil.copy2(best_pt, weights_dir_pt)

        print(f"\n--> Evaluating {name} on held-out Test Split (108 images, 390 objects)...", flush=True)
        eval_model = YOLO(str(best_pt))
        metrics_val = eval_model.val(data=str(DATA_YAML), split="val", imgsz=640, batch=16, device="0", workers=0, verbose=False)
        metrics = eval_model.val(data=str(DATA_YAML), split="test", imgsz=640, batch=16, device="0", workers=0, verbose=False)

        mp = float(metrics.box.mp) * 100.0
        mr = float(metrics.box.mr) * 100.0
        map50 = float(metrics.box.map50) * 100.0
        map50_95 = float(metrics.box.map) * 100.0
        lat_ms = measure_gpu_latency_ms(best_pt)

        # Log both Val and Official Test to results/results.csv & results/results_highlighted.xlsx
        log_run(
            model_id=key,
            model_name=f"{name} [Val]",
            folder=f"models/{key}",
            params_M=item["params_m"],
            gflops=item["gflops"],
            epochs=100,
            mAP50=round(float(metrics_val.box.map50), 4),
            mAP50_95=round(float(metrics_val.box.map), 4),
            precision=round(float(metrics_val.box.mp), 4),
            recall=round(float(metrics_val.box.mr), 4),
            inference_ms=round(lat_ms, 2),
            train_time_min=0.0,
            notes=f"Competing literature model ({key}) evaluated on validation split (213 images)",
        )
        log_run(
            model_id=f"{key}_eval",
            model_name=f"{name} [Official Test]",
            folder=f"models/{key}",
            params_M=item["params_m"],
            gflops=item["gflops"],
            epochs=100,
            mAP50=round(float(metrics.box.map50), 4),
            mAP50_95=round(float(metrics.box.map), 4),
            precision=round(float(metrics.box.mp), 4),
            recall=round(float(metrics.box.mr), 4),
            inference_ms=round(lat_ms, 2),
            train_time_min=0.0,
            notes=f"Competing literature model ({key}) evaluated on official held-out test split (108 images)",
        )

        # Per-class metrics: 0=Person, 1=Trolley, 2=Bag, 3=Unattended
        p_cls = [float(x) * 100.0 for x in metrics.box.p]
        r_cls = [float(x) * 100.0 for x in metrics.box.r]
        ap50_cls = [float(x) * 100.0 for x in metrics.box.ap50]
        ap_cls = [float(x) * 100.0 for x in metrics.box.ap]

        row = {
            "Model": name,
            "Params_M": item["params_m"],
            "GFLOPs": item["gflops"],
            "Precision": round(mp, 2),
            "Recall": round(mr, 2),
            "mAP50": round(map50, 2),
            "mAP50_95": round(map50_95, 2),
            "GPU_Latency_ms": round(lat_ms, 2),
            "Person_P": round(p_cls[0], 1),
            "Person_R": round(r_cls[0], 1),
            "Person_mAP50": round(ap50_cls[0], 1),
            "Person_mAP50_95": round(ap_cls[0], 1),
            "Trolley_P": round(p_cls[1], 1),
            "Trolley_R": round(r_cls[1], 1),
            "Trolley_mAP50": round(ap50_cls[1], 1),
            "Trolley_mAP50_95": round(ap_cls[1], 1),
            "Bag_P": round(p_cls[2], 1),
            "Bag_R": round(r_cls[2], 1),
            "Bag_mAP50": round(ap50_cls[2], 1),
            "Bag_mAP50_95": round(ap_cls[2], 1),
            "Unattended_P": round(p_cls[3], 1),
            "Unattended_R": round(r_cls[3], 1),
            "Unattended_mAP50": round(ap50_cls[3], 1),
            "Unattended_mAP50_95": round(ap_cls[3], 1),
        }
        results_summary.append(row)
        print(
            f"  [Test Split Result] {name}:\n"
            f"    Precision: {mp:.2f}% | Recall: {mr:.2f}% | mAP@50: {map50:.2f}% | mAP@50-95: {map50_95:.2f}% | Latency: {lat_ms:.2f} ms\n"
            f"    Per-Class mAP@50: Person={ap50_cls[0]:.1f}%, Trolley={ap50_cls[1]:.1f}%, Bag={ap50_cls[2]:.1f}%, Unattended={ap50_cls[3]:.1f}%\n"
            f"    Per-Class mAP@50-95: Person={ap_cls[0]:.1f}%, Trolley={ap_cls[1]:.1f}%, Bag={ap_cls[2]:.1f}%, Unattended={ap_cls[3]:.1f}%",
            flush=True,
        )

    # Also append our Proposed Model (v30) for side-by-side reference
    v30_pt = RUNS_DIR / "v30_v27_cosine_100ep" / "train" / "weights" / "best.pt"
    v30_lat = measure_gpu_latency_ms(v30_pt)
    results_summary.append({
        "Model": "Proposed (Hybrid Dilated ECA-YOLO v30)",
        "Params_M": 3.01,
        "GFLOPs": 8.1,
        "Precision": 87.22,
        "Recall": 83.38,
        "mAP50": 87.89,
        "mAP50_95": 64.73,
        "GPU_Latency_ms": round(v30_lat, 2),
        "Person_P": 79.1,
        "Person_R": 66.4,
        "Person_mAP50": 76.9,
        "Person_mAP50_95": 51.1,
        "Trolley_P": 95.0,
        "Trolley_R": 94.0,
        "Trolley_mAP50": 97.9,
        "Trolley_mAP50_95": 82.5,
        "Bag_P": 76.0,
        "Bag_R": 89.7,
        "Bag_mAP50": 92.7,
        "Bag_mAP50_95": 66.0,
        "Unattended_P": 98.7,
        "Unattended_R": 83.3,
        "Unattended_mAP50": 84.1,
        "Unattended_mAP50_95": 59.3,
    })

    out_csv_results = REPO_ROOT / "results" / "competing_thermal_yolo_comparison.csv"
    out_csv_results.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_results, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_summary[0].keys()))
        writer.writeheader()
        writer.writerows(results_summary)

    out_csv = REPO_ROOT / "paper_assets" / "tables" / "competing_thermal_yolo_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_summary[0].keys()))
        writer.writeheader()
        writer.writerows(results_summary)

    print(f"\nSaved complete comparison table to: {out_csv_results} and {out_csv}")


if __name__ == "__main__":
    main()
