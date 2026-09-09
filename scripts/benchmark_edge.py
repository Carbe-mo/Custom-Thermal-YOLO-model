"""
scripts/benchmark_edge.py
==========================
Standalone cross-platform inference latency & throughput benchmark.
Measures BOTH:
  1. Pure Model Inference (Forward Pass) -> Architecture Efficiency
  2. Full End-to-End Pipeline (Preprocess + Inference + NMS Postprocess) -> Real-World Edge Deployment

Usage:
  python scripts/benchmark_edge.py --weights weights/v30_champion_best.pt --device cpu --runs 100
  python scripts/benchmark_edge.py --weights weights/v30_champion_best.pt --device 0 --half --runs 300
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def register_custom_modules():
    try:
        from common.modules.c2f_eca import register_c2f_eca
        register_c2f_eca()
    except Exception as e:
        print(f"[Warn] Could not register C2f_ECA: {e}")

    try:
        from common.modules.c2f_eca_dilated import register_c2f_eca_dilated
        register_c2f_eca_dilated()
    except Exception as e:
        print(f"[Warn] Could not register C2f_ECA_Dilated: {e}")

def get_system_info():
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info

def benchmark(weights_path: str, device: str = "cpu", imgsz: int = 640,
              warmup: int = 20, runs: int = 200, half: bool = False):
    register_custom_modules()
    from ultralytics import YOLO

    # Parse device
    if device.isdigit():
        torch_dev = torch.device(f"cuda:{device}")
        dev_str = device
    elif device in ["cuda", "0"]:
        torch_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dev_str = "0" if torch.cuda.is_available() else "cpu"
    else:
        torch_dev = torch.device(device)
        dev_str = device

    is_cuda = torch_dev.type == "cuda"

    print("=" * 65)
    print("      CROSS-PLATFORM EDGE INFERENCE BENCHMARK      ")
    print("=" * 65)
    sys_info = get_system_info()
    for k, v in sys_info.items():
        print(f"  {k:18s}: {v}")
    print(f"  weights           : {weights_path}")
    print(f"  target device     : {torch_dev}")
    print(f"  input resolution  : {imgsz}x{imgsz}")
    print(f"  precision         : {'FP16' if (half and is_cuda) else 'FP32'}")
    print(f"  warmup iterations : {warmup}")
    print(f"  timed iterations  : {runs}")
    print("-" * 65)

    # 1. Load Model
    model = YOLO(weights_path)
    model.model.to(torch_dev)
    model.model.eval()
    if half and is_cuda:
        model.model.half()

    # -------------------------------------------------------------
    # PHASE 1: Pure Model Forward Pass (Neural Network Core)
    # -------------------------------------------------------------
    print("[1/2] Profiling Pure Model Forward Pass (Tensor input)...")
    dtype = torch.float16 if (half and is_cuda) else torch.float32
    dummy_tensor = torch.zeros((1, 3, imgsz, imgsz), dtype=dtype, device=torch_dev)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.model(dummy_tensor)
            if is_cuda:
                torch.cuda.synchronize()

    # Timed runs
    pure_latencies = []
    with torch.no_grad():
        for _ in range(runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model.model(dummy_tensor)

            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            pure_latencies.append((t1 - t0) * 1000.0)

    pure_latencies.sort()
    if runs >= 50:
        trim = max(1, int(runs * 0.02))
        trimmed_pure = pure_latencies[trim:-trim]
    else:
        trimmed_pure = pure_latencies

    pure_mean = sum(trimmed_pure) / len(trimmed_pure)
    pure_median = trimmed_pure[len(trimmed_pure) // 2]
    pure_fps = 1000.0 / pure_mean

    # -------------------------------------------------------------
    # PHASE 2: Full End-to-End Pipeline (Preprocess + Forward + NMS)
    # -------------------------------------------------------------
    print("[2/2] Profiling End-to-End Pipeline (Preprocess + Forward + NMS)...")
    dummy_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(min(10, warmup)):
        _ = model(dummy_img, device=dev_str, half=(half and is_cuda), verbose=False)

    pre_times, inf_times, post_times = [], [], []
    e2e_runs = min(runs, 150)  # 100-150 runs is plenty for E2E
    for _ in range(e2e_runs):
        res = model(dummy_img, device=dev_str, half=(half and is_cuda), verbose=False)[0]
        pre_times.append(res.speed['preprocess'])
        inf_times.append(res.speed['inference'])
        post_times.append(res.speed['postprocess'])

    pre_mean = sum(pre_times) / len(pre_times)
    inf_mean = sum(inf_times) / len(inf_times)
    post_mean = sum(post_times) / len(post_times)
    total_e2e_mean = pre_mean + inf_mean + post_mean
    e2e_fps = 1000.0 / total_e2e_mean

    mem_mb = 0.0
    if is_cuda:
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # -------------------------------------------------------------
    # RESULTS REPORT
    # -------------------------------------------------------------
    print("=" * 65)
    print("                     BENCHMARK SUMMARY                     ")
    print("=" * 65)
    print("  [A] PURE MODEL FORWARD PASS (Network Architecture):")
    print(f"      - Mean Latency    : {pure_mean:.2f} ms")
    print(f"      - Median Latency  : {pure_median:.2f} ms")
    print(f"      - Throughput      : {pure_fps:.1f} FPS")
    print()
    print("  [B] FULL END-TO-END PIPELINE (Real-World Deployment):")
    print(f"      - Pre-processing  : {pre_mean:.2f} ms")
    print(f"      - Model Inference : {inf_mean:.2f} ms")
    print(f"      - NMS Post-process: {post_mean:.2f} ms")
    print(f"      - Total Latency   : {total_e2e_mean:.2f} ms")
    print(f"      - Real-World FPS  : {e2e_fps:.1f} FPS")
    if is_cuda:
        print(f"      - Peak VRAM Usage : {mem_mb:.2f} MB")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-platform Edge Inference Benchmark")
    parser.add_argument("--weights", type=str, default="weights/v30_champion_best.pt", help="Path to .pt weights")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu', '0', 'cuda'")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Timed iterations")
    parser.add_argument("--half", action="store_true", help="FP16 precision (GPU only)")
    args = parser.parse_args()

    benchmark(
        weights_path=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        warmup=args.warmup,
        runs=args.runs,
        half=args.half
    )
