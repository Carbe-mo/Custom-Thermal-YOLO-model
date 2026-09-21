"""
scripts/joulescope_capture.py
==============================
Standalone Joulescope energy & power benchmarking tool for Edge AI hardware.
Measures dynamic current, voltage, power, and energy consumption for:
  - Raspberry Pi 4B (ARM Cortex-A72)
  - Rock Pi 4C+ (RK3399 ARM Cortex-A72 / A53)
  - NVIDIA Jetson AGX Xavier (Carmel ARMv8.2 + 512-core Volta GPU)

Usage (Run on Host PC with Joulescope USB connected):
  pip install pyjoulescope joulescope
  python scripts/joulescope_capture.py --device "Raspberry Pi 4B" --model "Dilated_ECA_v30" --latency-ms 970.76
  python scripts/joulescope_capture.py --device "Rock Pi 4C+" --model "Dilated_ECA_v30" --latency-ms 1238.64
  python scripts/joulescope_capture.py --device "Jetson AGX Xavier" --model "Dilated_ECA_v30" --latency-ms 15.2
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "results" / "power_benchmark_results.csv"


def check_joulescope_installed():
    try:
        import joulescope
        return joulescope
    except ImportError:
        print("\n[ERROR] 'joulescope' package not found.")
        print("Please install via:")
        print("    pip install pyjoulescope joulescope\n")
        sys.exit(1)


def capture_stream(device, duration_sec: float, desc: str = "Measurement"):
    """
    Capture high-resolution statistics from Joulescope over a specified window.
    Supports both JS220 and JS110 returning 2D NumPy array [current, voltage] or dict.
    """
    print(f"\n[Joulescope] Capturing {desc} for {duration_sec:.1f}s...")
    t0 = time.time()
    
    data = device.read(duration=duration_sec)
    actual_dur = time.time() - t0
    
    if isinstance(data, dict):
        current_a = np.asarray(data["signals"]["current"]["value"])
        voltage_v = np.asarray(data["signals"]["voltage"]["value"])
    else:
        # Standard Joulescope JS220 / v1 read returns Nx2 np.ndarray: [current (A), voltage (V)]
        current_a = data[:, 0]
        voltage_v = data[:, 1]
    
    power_w = current_a * voltage_v
    mean_power = float(np.mean(power_w))
    mean_voltage = float(np.mean(voltage_v))
    mean_current_ma = float(np.mean(current_a)) * 1000.0
    energy_j = mean_power * actual_dur
    
    return {
        "duration": actual_dur,
        "voltage_v": mean_voltage,
        "current_ma": mean_current_ma,
        "power_w": mean_power,
        "energy_j": energy_j,
    }


def save_power_result(row_dict):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = OUTPUT_CSV.exists()
    
    headers = [
        "Timestamp",
        "Target_Device",
        "Model_Name",
        "E2E_Latency_ms",
        "E2E_FPS",
        "Idle_Power_W",
        "Active_Power_W",
        "Dynamic_Power_W",
        "Energy_per_Frame_mJ",
        "Frames_per_Joule",
        "Mean_Voltage_V",
        "Mean_Current_mA",
    ]
    
    with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
    print(f"\n[Saved] Appended reading to {OUTPUT_CSV}")


def run_benchmark(device_name: str, model_name: str, latency_ms: float,
                  idle_duration: float = 15.0, active_duration: float = 30.0):
    js = check_joulescope_installed()
    
    print("\n" + "=" * 70)
    print(f" JOULESCOPE EDGE POWER BENCHMARK: {device_name} — {model_name}")
    print("=" * 70)
    
    # 1. Open Joulescope
    print("\nScanning for connected Joulescope...")
    try:
        device = js.scan_require_one(config="auto")
        device.open()
        print(f"Connected to: {device}")
    except Exception as e:
        print(f"\n[ERROR] Failed to connect to Joulescope: {e}")
        print("Please check USB cable connection and permissions.\n")
        return

    try:
        # 2. Measure Idle Baseline
        print("\n" + "-" * 70)
        print(" PHASE 1: IDLE BASELINE")
        print(f" Ensure {device_name} is turned ON and IDLE (no detection running).")
        input(" Press [ENTER] to record Idle Power baseline...")
        
        idle_data = capture_stream(device, idle_duration, desc=f"Idle ({device_name})")
        p_idle = idle_data["power_w"]
        print(f" -> Idle Power: {p_idle:.3f} W (Voltage: {idle_data['voltage_v']:.2f} V, Current: {idle_data['current_ma']:.1f} mA)")
        
        # 3. Measure Active Inference
        print("\n" + "-" * 70)
        print(" PHASE 2: ACTIVE INFERENCE")
        print(f" Start the continuous YOLO benchmark on {device_name}, e.g.:")
        print(f"    python scripts/benchmark_edge.py --weights weights/{model_name}.pt --runs 300")
        input("\n When the model starts running on the board, press [ENTER] to start recording...")
        
        active_data = capture_stream(device, active_duration, desc=f"Active Inference ({model_name})")
        p_active = active_data["power_w"]
        print(f" -> Active Power: {p_active:.3f} W (Voltage: {active_data['voltage_v']:.2f} V, Current: {active_data['current_ma']:.1f} mA)")
        
        # 4. Calculate Derived Green AI Metrics
        dynamic_power = max(0.0, p_active - p_idle)
        fps = (1000.0 / latency_ms) if latency_ms > 0 else 0.0
        
        # Energy per frame: Power (W = J/s) * Latency (s) = Joules / frame
        latency_sec = latency_ms / 1000.0
        energy_per_frame_j = p_active * latency_sec
        energy_per_frame_mj = energy_per_frame_j * 1000.0
        
        # Efficiency: Frames per Joule
        frames_per_joule = (1.0 / energy_per_frame_j) if energy_per_frame_j > 0 else 0.0

        # 5. Format & Display Results
        print("\n" + "=" * 70)
        print(" SUMMARY BENCHMARK METRICS (Ready for IEEE Sensors Table VII)")
        print("=" * 70)
        print(f" Target Hardware        : {device_name}")
        print(f" Evaluated Model        : {model_name}")
        print(f" Latency (E2E)          : {latency_ms:.2f} ms ({fps:.2f} FPS)")
        print(f" Baseline Idle Power    : {p_idle:.3f} W")
        print(f" Mean Active Power      : {p_active:.3f} W")
        print(f" Dynamic Power Delta    : {dynamic_power:.3f} W")
        print(f" Energy per Frame       : {energy_per_frame_mj:.2f} mJ / frame")
        print(f" Energy Efficiency      : {frames_per_joule:.2f} Frames / Joule")
        print("=" * 70)

        # 6. Save to CSV
        row = {
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Target_Device": device_name,
            "Model_Name": model_name,
            "E2E_Latency_ms": round(latency_ms, 2),
            "E2E_FPS": round(fps, 2),
            "Idle_Power_W": round(p_idle, 3),
            "Active_Power_W": round(p_active, 3),
            "Dynamic_Power_W": round(dynamic_power, 3),
            "Energy_per_Frame_mJ": round(energy_per_frame_mj, 2),
            "Frames_per_Joule": round(frames_per_joule, 2),
            "Mean_Voltage_V": round(active_data["voltage_v"], 2),
            "Mean_Current_mA": round(active_data["current_ma"], 1),
        }
        save_power_result(row)
        
    finally:
        device.close()
        print("[Joulescope] Disconnected successfully.")


def main():
    parser = argparse.ArgumentParser(description="Joulescope Energy Measurement for Edge AI")
    parser.add_argument("--device", type=str, default="Raspberry Pi 4B",
                        choices=["Raspberry Pi 4B", "Rock Pi 4C+", "Jetson AGX Xavier", "Generic Edge"],
                        help="Target edge hardware platform")
    parser.add_argument("--model", type=str, default="v30_champion_best",
                        help="Name of the model being evaluated")
    parser.add_argument("--latency-ms", type=float, default=970.76,
                        help="End-to-end inference latency in ms (used to compute mJ/frame)")
    parser.add_argument("--idle-sec", type=float, default=15.0,
                        help="Seconds to sample idle baseline")
    parser.add_argument("--active-sec", type=float, default=30.0,
                        help="Seconds to sample active inference")
    args = parser.parse_args()
    
    run_benchmark(
        device_name=args.device,
        model_name=args.model,
        latency_ms=args.latency_ms,
        idle_duration=args.idle_sec,
        active_duration=args.active_sec,
    )


if __name__ == "__main__":
    main()
