"""
scripts/benchmark_suite.py
===========================
Automated benchmark suite for all models matching the IEEE paper comparison:
  1. Ours (Hybrid Dilated ECA v30)
  2. YOLOv8n
  3. YOLOv8s
  4. YOLOv10n (Incompatible with ARMv8.0 Cortex-A72 on RPi 4B; requires ARMv8.2+)
  5. YOLOv10s
  6. YOLOv11n
  7. YOLOv11s

Features:
  - Isolated subprocess execution per model: If a model crashes (e.g. YOLOv10 SIGILL
    'Illegal instruction' on Raspberry Pi 4), the suite DOES NOT terminate; it marks
    that model as Incompatible and seamlessly completes all remaining models!
  - Auto-detection for Raspberry Pi 4 to automatically flag YOLOv10 incompatibility.
  - Generates unified CSV and printed terminal table.

Usage:
  python scripts/benchmark_suite.py --device cpu --runs 30
  python scripts/benchmark_suite.py --device cpu --skip-v10
  python scripts/benchmark_suite.py --device 0 --half --runs 100
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

ALL_MODELS = [
    ("Ours (Hybrid Dilated ECA)", "weights/v30_champion_best.pt"),
    ("YOLOv8n", "yolov8n.pt"),
    ("YOLOv8s", "yolov8s.pt"),
    ("YOLOv10n", "yolov10n.pt"),
    ("YOLOv10s", "yolov10s.pt"),
    ("YOLOv11n", "yolo11n.pt"),
    ("YOLOv11s", "yolo11s.pt"),
]


def is_raspberry_pi():
    """Detect if running on a Raspberry Pi (BCM2711 / Cortex-A72)."""
    try:
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", "r") as f:
                model_str = f.read().lower()
                if "raspberry pi" in model_str:
                    return True
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read().lower()
                if "bcm2835" in cpuinfo or "bcm2711" in cpuinfo or "raspberry" in cpuinfo:
                    return True
    except Exception:
        pass
    return False


# Worker code executed in an isolated child process
WORKER_SCRIPT = """
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(r"{repo_root}")
sys.path.insert(0, str(REPO_ROOT))

try:
    from common.modules.c2f_eca import register_c2f_eca
    register_c2f_eca()
    from common.modules.c2f_eca_dilated import register_c2f_eca_dilated
    register_c2f_eca_dilated()
except Exception as e:
    pass

from ultralytics import YOLO

weights_path = r"{weights_path}"
device_str = "{device_str}"
imgsz = {imgsz}
warmup = {warmup}
runs = {runs}
use_half = {use_half}

if device_str.isdigit():
    torch_dev = torch.device(f"cuda:{{device_str}}")
    dev_arg = device_str
elif device_str in ["cuda", "0"]:
    torch_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev_arg = "0" if torch.cuda.is_available() else "cpu"
else:
    torch_dev = torch.device(device_str)
    dev_arg = device_str

is_cuda = torch_dev.type == "cuda"
model = YOLO(weights_path)
model.model.to(torch_dev)
model.model.eval()
if use_half and is_cuda:
    model.model.half()

# 1. Pure Forward Pass
dtype = torch.float16 if (use_half and is_cuda) else torch.float32
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

# 2. End-to-End Pipeline
dummy_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
for _ in range(min(5, warmup)):
    _ = model(dummy_img, device=dev_arg, half=use_half, verbose=False)

e2e_runs = min(runs, 30)
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

result = {{
    "Pure_Latency_ms": round(pure_mean, 2),
    "Pure_FPS": round(pure_fps, 1),
    "E2E_Preprocess_ms": round(pre_m, 2),
    "E2E_Inference_ms": round(inf_m, 2),
    "E2E_NMS_ms": round(post_m, 2),
    "E2E_Total_ms": round(e2e_total, 2),
    "E2E_FPS": round(e2e_fps, 1),
}}
print("___JSON_RESULT___" + json.dumps(result))
"""


def benchmark_isolated(name, weights_path, device_str, imgsz=640, warmup=10, runs=30, half=False):
    print(f"\n--> Profiling {name} ({weights_path})...")

    # Generate child python code
    code = WORKER_SCRIPT.format(
        repo_root=str(REPO_ROOT),
        weights_path=weights_path,
        device_str=device_str,
        imgsz=imgsz,
        warmup=warmup,
        runs=runs,
        use_half=str(half),
    )

    cmd = [sys.executable, "-c", code]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(REPO_ROOT))
    except Exception as e:
        print(f"    [Error] Subprocess execution failed: {e}")
        return None

    # Check for crash (e.g. SIGILL Illegal instruction = returncode 132 or -4)
    if proc.returncode != 0:
        err_msg = proc.stderr.strip()
        if "illegal instruction" in err_msg.lower() or proc.returncode in [132, -4]:
            print(f"    [Incompatible] Crashed with 'Illegal instruction' (requires ARMv8.2+, board is ARMv8.0).")
            return {
                "Model": name,
                "Weights": weights_path,
                "Precision": "N/A",
                "Pure_Latency_ms": "Illegal Instruction (ARMv8.0)",
                "Pure_FPS": "N/A",
                "E2E_Preprocess_ms": None,
                "E2E_Inference_ms": None,
                "E2E_NMS_ms": None,
                "E2E_Total_ms": "Illegal Instruction",
                "E2E_FPS": "N/A",
            }
        else:
            print(f"    [Error] Model exited with code {proc.returncode}")
            if err_msg:
                print(f"    {err_msg[-300:]}")
            return None

    # Extract JSON result
    output = proc.stdout
    if "___JSON_RESULT___" not in output:
        print("    [Error] Could not parse benchmark output.")
        return None

    json_str = output.split("___JSON_RESULT___")[1].strip()
    data = json.loads(json_str)

    data["Model"] = name
    data["Weights"] = weights_path
    data["Precision"] = "FP16" if half else "FP32"

    print(f"    Pure Forward: {data['Pure_Latency_ms']} ms ({data['Pure_FPS']} FPS) | E2E: {data['E2E_Total_ms']} ms ({data['E2E_FPS']} FPS)")
    return data


def main():
    parser = argparse.ArgumentParser(description="Full Model Suite Benchmark (Crash-Resilient)")
    parser.add_argument("--device", type=str, default="cpu", help="'0', 'cuda', 'cpu'")
    parser.add_argument("--half", action="store_true", help="FP16 precision (GPU only)")
    parser.add_argument("--runs", type=int, default=25, help="Runs per model")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--skip-v10", action="store_true", help="Explicitly skip YOLOv10")
    parser.add_argument("--models", nargs="+", default=None, help="Specific models to test")
    parser.add_argument("--out-csv", type=str, default="results/suite_benchmark.csv", help="Output CSV path")
    args = parser.parse_args()

    on_rpi = is_raspberry_pi()
    if on_rpi:
        print("[Notice] Raspberry Pi hardware detected (ARM Cortex-A72 / ARMv8.0).")
        if not args.skip_v10:
            print("[Notice] YOLOv10 requires ARMv8.2 instructions and will be safely isolated if it throws SIGILL.")

    models_to_run = []
    for name, path in ALL_MODELS:
        if args.skip_v10 and "v10" in path.lower():
            print(f"[Skip] Skipping {name} due to --skip-v10 flag.")
            continue
        if args.models:
            if not any(m.lower() in path.lower() for m in args.models):
                continue
        models_to_run.append((name, path))

    print("=" * 70)
    print("       CRASH-RESILIENT EDGE BENCHMARK SUITE       ")
    print("=" * 70)
    print(f"Device: {args.device} | Runs: {args.runs} | Models: {len(models_to_run)}")
    print("=" * 70)

    records = []
    for name, path in models_to_run:
        res = benchmark_isolated(
            name, path, args.device,
            warmup=args.warmup,
            runs=args.runs,
            half=args.half
        )
        if res:
            records.append(res)

    if records:
        df = pd.DataFrame(records)
        out_path = REPO_ROOT / args.out_csv
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        print("\n" + "=" * 70)
        print("                   FINAL BENCHMARK TABLE                   ")
        print("=" * 70)
        display_cols = ["Model", "Precision", "Pure_Latency_ms", "Pure_FPS", "E2E_Total_ms", "E2E_FPS"]
        cols_present = [c for c in display_cols if c in df.columns]
        print(df[cols_present].to_string(index=False))
        print("=" * 70)
        print(f"Results saved to: {out_path}")
    else:
        print("[Error] No benchmark records collected.")


if __name__ == "__main__":
    main()
