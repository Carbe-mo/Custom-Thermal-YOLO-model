"""
scripts/benchmark_edge.py
==========================
Standalone cross-platform inference latency & throughput benchmark.
Works on:
  - Coder Servers (Intel Xeon / AMD EPYC / Tesla T4 / A100)
  - NVIDIA Jetson AGX Xavier (CUDA / TensorRT / FP16)
  - Raspberry Pi 4 / 5 (ARM Cortex-A72 / BCM2712 CPU)
  - Rock Pi 4C+ (Rockchip RK3399 CPU)

Usage:
  python scripts/benchmark_edge.py --weights weights/v30_champion_best.pt --device cpu --runs 200
  python scripts/benchmark_edge.py --weights weights/v30_champion_best.pt --device 0 --half --runs 300
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def register_custom_modules():
    """Register custom C2f_ECA and C2f_ECA_Dilated modules into ultralytics."""
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
              warmup: int = 30, runs: int = 200, half: bool = False):
    register_custom_modules()
    from ultralytics import YOLO

    # Parse torch device
    if device.isdigit():
        torch_dev = torch.device(f"cuda:{device}")
    elif device in ["cuda", "0"]:
        torch_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        torch_dev = torch.device(device)

    is_cuda = torch_dev.type == "cuda"

    print("=" * 60)
    print(" EDGE INFERENCE LATENCY BENCHMARK ")
    print("=" * 60)
    sys_info = get_system_info()
    for k, v in sys_info.items():
        print(f"  {k:16s}: {v}")
    print(f"  weights         : {weights_path}")
    print(f"  target device   : {torch_dev}")
    print(f"  input size      : {imgsz}x{imgsz}")
    print(f"  precision       : {'FP16' if (half and is_cuda) else 'FP32'}")
    print(f"  warmup iterations: {warmup}")
    print(f"  timed runs      : {runs}")
    print("-" * 60)

    # 1. Load Model
    model = YOLO(weights_path)
    model.model.to(torch_dev)
    model.model.eval()

    if half and is_cuda:
        model.model.half()

    # 2. Prepare Dummy Tensor [1, 3, imgsz, imgsz]
    dtype = torch.float16 if (half and is_cuda) else torch.float32
    dummy = torch.zeros((1, 3, imgsz, imgsz), dtype=dtype, device=torch_dev)

    # 3. Warmup
    print("[1/3] Warming up hardware cache & kernels...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.model(dummy)
            if is_cuda:
                torch.cuda.synchronize()

    # 4. Timed Runs
    print(f"[2/3] Profiling over {runs} consecutive runs...")
    latencies = []
    with torch.no_grad():
        for i in range(runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model.model(dummy)

            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)

    # 5. Metrics
    latencies.sort()
    if runs >= 50:
        trim = max(1, int(runs * 0.02))
        trimmed = latencies[trim:-trim]
    else:
        trimmed = latencies

    mean_lat = sum(trimmed) / len(trimmed)
    median_lat = trimmed[len(trimmed) // 2]
    min_lat = min(trimmed)
    max_lat = max(trimmed)
    fps = 1000.0 / mean_lat

    mem_mb = 0.0
    if is_cuda:
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print("-" * 60)
    print(" BENCHMARK RESULTS ")
    print("-" * 60)
    print(f"  Mean Latency  : {mean_lat:.2f} ms")
    print(f"  Median Latency: {median_lat:.2f} ms")
    print(f"  Min / Max     : {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"  Throughput    : {fps:.2f} FPS")
    if is_cuda:
        print(f"  Peak VRAM     : {mem_mb:.2f} MB")
    print("=" * 60)

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
