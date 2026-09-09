"""
scripts/benchmark_suite.py
===========================
Automated benchmark suite for all models matching the IEEE paper comparison:
  1. Ours (Hybrid Dilated ECA v30)
  2. YOLOv8n
  3. YOLOv8s
  4. YOLOv10n
  5. YOLOv10s
  6. YOLOv11n
  7. YOLOv11s

Usage:
  # On GPU (A100 / RTX / Jetson):
  python scripts/benchmark_suite.py --device 0 --half --runs 200

  # On CPU (Server / Pi 4 / Rock Pi):
  python scripts/benchmark_suite.py --device cpu --runs 50
"""

import argparse
import sys
import time
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.modules.c2f_eca import register_c2f_eca
from common.modules.c2f_eca_dilated import register_c2f_eca_dilated
register_c2f_eca()
register_c2f_eca_dilated()

from ultralytics import YOLO

MODELS = [
    ("Ours (Hybrid Dilated ECA)", "weights/v30_champion_best.pt"),
    ("YOLOv8n", "yolov8n.pt"),
    ("YOLOv8s", "yolov8s.pt"),
    ("YOLOv10n", "yolov10n.pt"),
    ("YOLOv10s", "yolov10s.pt"),
    ("YOLOv11n", "yolo11n.pt"),
    ("YOLOv11s", "yolo11s.pt"),
]

def benchmark_single(name, weights_path, device_str, imgsz=640, warmup=15, runs=100, half=False):
    if device_str.isdigit():
        torch_dev = torch.device(f"cuda:{device_str}")
        dev_arg = device_str
    elif device_str in ["cuda", "0"]:
        torch_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dev_arg = "0" if torch.cuda.is_available() else "cpu"
    else:
        torch_dev = torch.device(device_str)
        dev_arg = device_str

    is_cuda = torch_dev.type == "cuda"
    use_half = half and is_cuda

    print(f"\n--> Profiling {name} ({weights_path})...")
    model = YOLO(weights_path)
    model.model.to(torch_dev)
    model.model.eval()
    if use_half:
        model.model.half()

    # 1. Pure Model Forward Pass
    dtype = torch.float16 if use_half else torch.float32
    dummy_tensor = torch.zeros((1, 3, imgsz, imgsz), dtype=dtype, device=torch_dev)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model.model(dummy_tensor)
            if is_cuda: torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            if is_cuda: torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.model(dummy_tensor)
            if is_cuda: torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    latencies.sort()
    trimmed = latencies[max(1, int(runs * 0.02)): -max(1, int(runs * 0.02))] if runs >= 50 else latencies
    pure_mean = sum(trimmed) / len(trimmed)
    pure_fps = 1000.0 / pure_mean

    # 2. End-to-End Pipeline (Preprocess + Forward + NMS)
    dummy_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(min(5, warmup)):
        _ = model(dummy_img, device=dev_arg, half=use_half, verbose=False)

    e2e_runs = min(runs, 80)
    pre_l, inf_l, post_l = [], [], []
    for _ in range(e2e_runs):
        res = model(dummy_img, device=dev_arg, half=use_half, verbose=False)[0]
        pre_l.append(res.speed['preprocess'])
        inf_l.append(res.speed['inference'])
        post_l.append(res.speed['postprocess'])

    pre_m = sum(pre_l) / len(pre_l)
    inf_m = sum(inf_l) / len(inf_l)
    post_m = sum(post_l) / len(post_l)
    e2e_total = pre_m + inf_m + post_m
    e2e_fps = 1000.0 / e2e_total

    vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if is_cuda else 0.0

    print(f"    Pure Forward: {pure_mean:.2f} ms ({pure_fps:.1f} FPS) | E2E: {e2e_total:.2f} ms ({e2e_fps:.1f} FPS)")

    return {
        "Model": name,
        "Weights": weights_path,
        "Precision": "FP16" if use_half else "FP32",
        "Pure_Latency_ms": round(pure_mean, 2),
        "Pure_FPS": round(pure_fps, 1),
        "E2E_Preprocess_ms": round(pre_m, 2),
        "E2E_Inference_ms": round(inf_m, 2),
        "E2E_NMS_ms": round(post_m, 2),
        "E2E_Total_ms": round(e2e_total, 2),
        "E2E_FPS": round(e2e_fps, 1),
        "Peak_VRAM_MB": round(vram_mb, 2) if is_cuda else None
    }

def main():
    parser = argparse.ArgumentParser(description="Full Model Suite Benchmark")
    parser.add_argument("--device", type=str, default="0", help="'0', 'cuda', 'cpu'")
    parser.add_argument("--half", action="store_true", help="FP16 precision (GPU only)")
    parser.add_argument("--runs", type=int, default=150, help="Runs per model")
    parser.add_argument("--warmup", type=int, default=15, help="Warmup iterations")
    parser.add_argument("--out-csv", type=str, default="results/suite_benchmark.csv", help="Output CSV path")
    args = parser.parse_args()

    print("=" * 70)
    print("       FULL IEEE PAPER COMPARISON BENCHMARK SUITE       ")
    print("=" * 70)
    print(f"Device: {args.device} | Precision: {'FP16' if args.half else 'FP32'} | Runs: {args.runs}")
    print("=" * 70)

    records = []
    for name, path in MODELS:
        try:
            res = benchmark_single(name, path, args.device, warmup=args.warmup, runs=args.runs, half=args.half)
            records.append(res)
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")

    df = pd.DataFrame(records)
    out_path = REPO_ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("                   FINAL BENCHMARK TABLE                   ")
    print("=" * 70)
    print(df[["Model", "Precision", "Pure_Latency_ms", "Pure_FPS", "E2E_Total_ms", "E2E_FPS"]].to_string(index=False))
    print("=" * 70)
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    main()
