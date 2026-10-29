"""
scripts/benchmark_rpi5_timing.py
================================
Runs 10 inferences per model across all 8 architectures on Raspberry Pi 5 (8GB, Cortex-A76).
Logs latency, FPS, and execution times to results/rpi5_timing_benchmark.csv.
"""

import csv
import re
import sys
import time
from pathlib import Path
import paramiko

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "results" / "rpi5_timing_benchmark.csv"

RPI_IP = "192.168.137.216"
USER = "rpi"
PASS = "rpi"
REMOTE_DIR = "/home/rpi/Custom-Thermal-YOLO-model"

MODELS = [
    ("Ours (Dilated ECA v30)", "weights/v30_champion_best.pt", 3),
    ("YOLOv8n", "weights/yolov8n.pt", 3),
    ("YOLOv8s", "weights/yolov8s.pt", 3),
    ("YOLOv10n", "weights/yolov10n.pt", 3),
    ("YOLOv10s", "weights/yolov10s.pt", 3),
    ("YOLOv11n", "weights/yolo11n.pt", 3),
    ("YOLOv11s", "weights/yolo11s.pt", 3),
    ("Ours Dual-Backbone (v35)", "weights/v35_dual_fusion_best.pt", 6),
]


def save_row(row_dict):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def main(runs=10):
    print("=" * 80)
    print(f"  RASPBERRY PI 5 TIMING BENCHMARK SUITE: {USER}@{RPI_IP}")
    print(f"  Inferences per model: {runs} | Target Architecture: Cortex-A76 @ 2.4 GHz")
    print("=" * 80)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(RPI_IP, username=USER, password=PASS, timeout=15)
    print("Connected to Raspberry Pi 5 over SSH!\n")

    results = []

    for name, weights, ch in MODELS:
        print("\n" + "-" * 75)
        print(f"  PROFILING: {name} (Weights: {weights}, Channels: {ch})")
        print("-" * 75)

        cmd = f"cd {REMOTE_DIR} && python3 -u scripts/measure_edge_power.py --weights {weights} --runs {runs} --channels {ch}"
        stdin, stdout, stderr = client.exec_command(cmd)

        # Handshake: wait for READY
        ready = False
        while True:
            line = stdout.readline()
            if not line:
                break
            stripped = line.strip()
            print(f"    [RPi 5] {stripped}")
            if "[SYNC_READY]" in stripped:
                ready = True
                break

        if not ready:
            err = stderr.read().decode()
            print(f"  [ERROR] Model failed to load:\n{err}")
            continue

        # Send trigger to run exactly `runs` inferences
        t0 = time.time()
        stdin.write("START\n")
        stdin.flush()

        # Wait for completion
        done_line = None
        while True:
            line = stdout.readline()
            if not line:
                break
            stripped = line.strip()
            print(f"    [RPi 5] {stripped}")
            if "[SYNC_DONE" in stripped:
                done_line = stripped
                break

        wall_dur = time.time() - t0

        if not done_line:
            err = stderr.read().decode()
            print(f"  [ERROR] Execution failed:\n{err}")
            continue

        m_dur = re.search(r"duration_s=([0-9\.]+)", done_line)
        m_ms = re.search(r"avg_ms=([0-9\.]+)", done_line)
        m_fps = re.search(r"fps=([0-9\.]+)", done_line)

        inf_duration_s = float(m_dur.group(1)) if m_dur else wall_dur
        avg_latency_ms = float(m_ms.group(1)) if m_ms else (inf_duration_s / runs * 1000.0)
        fps = float(m_fps.group(1)) if m_fps else (1000.0 / avg_latency_ms)

        print(f"\n  -> Completed 10 frames in {inf_duration_s:.3f} s")
        print(f"  -> Mean Latency per Frame: {avg_latency_ms:.2f} ms ({avg_latency_ms/1000:.3f} s)")
        print(f"  -> Throughput            : {fps:.2f} FPS")

        rec = {
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Device": "Raspberry Pi 5 (8GB)",
            "Processor": "Broadcom BCM2712 Quad Cortex-A76 @ 2.4GHz",
            "Model_Name": name,
            "Weights": weights,
            "Channels": ch,
            "Inferences": runs,
            "Total_Duration_s": round(inf_duration_s, 3),
            "Avg_Latency_ms": round(avg_latency_ms, 2),
            "FPS": round(fps, 2),
        }
        save_row(rec)
        results.append(rec)
        time.sleep(3)

    client.close()

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'MODEL':<28} | {'LATENCY (ms)':<14} | {'LATENCY (s)':<12} | {'FPS':<8}")
    print("-" * 80)
    for r in results:
        sec = r['Avg_Latency_ms'] / 1000.0
        print(f"{r['Model_Name']:<28} | {r['Avg_Latency_ms']:<14.1f} | {sec:<12.3f} | {r['FPS']:<8.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
