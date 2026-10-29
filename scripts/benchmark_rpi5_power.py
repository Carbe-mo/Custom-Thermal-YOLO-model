"""
scripts/benchmark_rpi5_power.py
===============================
Automated Joulescope JS220 Power & Energy Benchmark Suite for Raspberry Pi 5 (8GB).
Synchronizes inference execution on Raspberry Pi 5 (Cortex-A76 @ 2.4GHz) over SSH
with high-speed microjoule-level power measurement from the Joulescope JS220.

Logs results to:
  - project_flir/thermal-yolo/results/power_benchmark_results.csv
  - project_flir/thermal-yolo/results/rpi5_power_benchmark.csv
"""

import argparse
import csv
import re
import socket
import sys
import time
from pathlib import Path
import functools
import numpy as np
import paramiko
import joulescope

print = functools.partial(print, flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASTER_CSV = PROJECT_ROOT / "results" / "power_benchmark_results.csv"
RPI5_CSV = PROJECT_ROOT / "results" / "rpi5_power_benchmark.csv"

# Models to evaluate (Name, weights_path, channels)
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


def save_power_row(csv_path, row_dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    headers = list(row_dict.keys())

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
    print(f"  --> Logged record to {csv_path.name}")


def scan_network_for_ssh(subnet="192.168.137.", port=22, timeout=0.2):
    """Scans subnet for responsive SSH hosts."""
    found = []
    print(f"Scanning {subnet}0/24 for SSH devices...")
    for i in range(2, 255):
        ip = f"{subnet}{i}"
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect((ip, port))
            found.append(ip)
        except Exception:
            pass
        finally:
            s.close()
    return found


def preflight_check(target_ip, username, password):
    print("\n" + "=" * 80)
    print("  PRE-FLIGHT DIAGNOSTICS")
    print("=" * 80)

    # 1. Joulescope Check
    print("[1/2] Checking Joulescope JS220 connection...")
    devices = joulescope.scan()
    if not devices:
        print("  ❌ Joulescope JS220 NOT detected by software.")
        print("     Please ensure the Joulescope USB cable is firmly plugged into the laptop.")
        print("     If previously plugged in, unplug and reconnect the USB cable.")
        js_ok = False
    else:
        print(f"  ✅ Joulescope detected: {devices[0]}")
        js_ok = True

    # 2. SSH Check
    print(f"\n[2/2] Checking SSH connection to Raspberry Pi 5 ({username}@{target_ip})...")
    ssh_ok = False
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(target_ip, username=username, password=password, timeout=5)
        _, stdout, _ = ssh.exec_command("hostname && uname -m && cat /proc/cpuinfo | grep 'Model'")
        out = stdout.read().decode().strip()
        print(f"  ✅ Raspberry Pi 5 SSH Connected!\n     {out}")
        ssh.close()
        ssh_ok = True
    except Exception as e:
        print(f"  ❌ Failed to connect to {username}@{target_ip}: {e}")
        print("     Tips:")
        print("     - Ensure Windows Mobile Hotspot is turned ON.")
        print("     - Verify Raspberry Pi 5 is powered ON and connected to Wi-Fi/Hotspot.")

    return js_ok, ssh_ok


def run_rpi5_power_benchmark(target_ip="172.20.10.4", username="rpi", password="rpi",
                             runs=10, remote_dir="/home/rpi/Custom-Thermal-YOLO-model"):
    print("=" * 80)
    print(f"  RASPBERRY PI 5 JOULESCOPE JS220 POWER BENCHMARK: {username}@{target_ip}")
    print(f"  Processor: Broadcom BCM2712 Quad Cortex-A76 @ 2.4 GHz | Inferences/model: {runs}")
    print("=" * 80)

    # 1. Open Joulescope JS220 FIRST to lock power output ON
    print("\n[Step 1] Initializing Joulescope JS220 and locking power output ON...")
    device = joulescope.scan_require_one(config="auto")
    device.open()
    device.parameter_set("i_range", "auto")
    device.parameter_set("sensor_power", "on")
    print(f"  ✅ Device initialized: {device}")
    print("  ⚡ SENSOR POWER IS LOCKED ON! Raspberry Pi 5 is receiving continuous power.")

    results = []
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # 2. Wait for Raspberry Pi 5 to boot and connect over SSH
        print(f"\n[Step 2] Waiting for Raspberry Pi 5 to finish booting and establish SSH ({username}@{target_ip})...")
        connected = False
        start_wait = time.time()
        candidate_ips = [target_ip] + [f"172.20.10.{x}" for x in [3, 4, 2, 5, 6, 7, 8] if f"172.20.10.{x}" != target_ip]

        active_ip = target_ip
        while not connected and (time.time() - start_wait < 180):
            try:
                data = device.read(duration=0.5)
                v = float(np.mean(np.abs(data[:, 1])))
                i = float(np.mean(np.abs(data[:, 0]))) * 1000.0
                p = v * i / 1000.0
            except Exception:
                v, i, p = 0.0, 0.0, 0.0

            elapsed = int(time.time() - start_wait)
            print(f"\r  ⚡ Power Rail: {v:.2f} V | {i:.1f} mA ({p:.3f} W) — Booting ({elapsed}s)...   ", end="")

            # Try candidate IPs
            for cip in candidate_ips:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(0.4)
                    s.connect((cip, 22))
                    s.close()
                    # Port 22 open, try full SSH
                    ssh.connect(cip, username=username, password=password, timeout=3)
                    _, stdout, _ = ssh.exec_command("hostname && uname -m && cat /proc/cpuinfo | grep 'Model'")
                    out = stdout.read().decode().strip().replace("\n", " | ")
                    print(f"\n  ✅ Raspberry Pi 5 SSH Connected at {cip}!\n     Details: {out}")
                    active_ip = cip
                    connected = True
                    break
                except Exception:
                    pass

            if not connected:
                time.sleep(1.0)

        if not connected:
            print(f"\n[TIMEOUT] Raspberry Pi 5 did not respond on SSH within 180 seconds.")
            return

        print("\n  [Stabilization] Waiting 10s for boot services to settle to idle...")
        time.sleep(10)

        # 3. Measure Idle Baseline
        print("\n[Step 3] Measuring 15-second Idle Baseline on Raspberry Pi 5...")
        time.sleep(2)
        data_idle = device.read(duration=15.0)
        v_idle = np.abs(data_idle[:, 1])
        i_idle = np.abs(data_idle[:, 0])
        p_idle = float(np.mean(v_idle * i_idle))
        mean_v_idle = float(np.mean(v_idle))
        mean_i_idle_ma = float(np.mean(i_idle)) * 1000.0
        print(f"  -> RPi 5 Idle Power : {p_idle:.3f} W (Voltage: {mean_v_idle:.2f} V, Current: {mean_i_idle_ma:.1f} mA)")

        # 4. Profile Each Model
        print("\n[Step 4] Starting Synchronized Inference Profiling across 8 Models...")
        for name, weights, ch in MODELS:
            print("\n" + "=" * 80)
            print(f"  PROFILING: {name} (Weights: {weights}, Channels: {ch})")
            print("=" * 80)

            cmd = f"cd {remote_dir} && python3 -u scripts/measure_edge_power.py --weights {weights} --runs {runs} --channels {ch}"
            print(f"  [RPi 5] Executing: {cmd}")
            stdin, stdout, stderr = ssh.exec_command(cmd)

            # Wait for model load & warmup
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
                err_msg = stderr.read().decode()
                print(f"  [ERROR] Model failed to signal READY:\n{err_msg}")
                continue

            # Start Joulescope buffer recording
            device.parameter_set("buffer_duration", 180)
            t0 = time.time()
            device.start()
            print(f"  🔴 [Joulescope] Recording high-speed samples... Triggering {runs} inferences")

            # Send trigger to start exact N inferences
            stdin.write("START\n")
            stdin.flush()

            # Wait for SYNC_DONE
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

            device.stop()
            t_active = time.time() - t0
            print(f"  ⏹️ [Joulescope] Stopped. Elapsed: {t_active:.3f}s")

            if not done_line:
                err_msg = stderr.read().decode()
                print(f"  [ERROR] Execution failed:\n{err_msg}")
                continue

            # Parse stats
            m_dur = re.search(r"duration_s=([0-9\.]+)", done_line)
            m_ms = re.search(r"avg_ms=([0-9\.]+)", done_line)
            m_fps = re.search(r"fps=([0-9\.]+)", done_line)

            inf_duration_s = float(m_dur.group(1)) if m_dur else t_active
            avg_latency_ms = float(m_ms.group(1)) if m_ms else (inf_duration_s / runs * 1000.0)
            fps = float(m_fps.group(1)) if m_fps else (1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0)

            # Retrieve Joulescope samples
            start_id, end_id = device.stream_buffer.sample_id_range
            samples = device.stream_buffer.samples_get(start_id, end_id, fields=["current", "voltage"])
            i_raw = np.abs(samples["signals"]["current"]["value"])
            v_raw = np.abs(samples["signals"]["voltage"]["value"])

            power_w = i_raw * v_raw
            p_active = float(np.mean(power_w))
            total_energy_j = p_active * inf_duration_s
            dynamic_power_w = max(0.0, p_active - p_idle)

            energy_per_frame_j = total_energy_j / runs
            energy_per_frame_mj = energy_per_frame_j * 1000.0
            frames_per_joule = (1.0 / energy_per_frame_j) if energy_per_frame_j > 0 else 0.0

            mean_volts = float(np.mean(v_raw))
            mean_curr_ma = float(np.mean(i_raw)) * 1000.0

            print("\n  " + "-" * 55)
            print(f"  RESULTS: {name}")
            print("  " + "-" * 55)
            print(f"  -> Mean Latency        : {avg_latency_ms:.2f} ms ({fps:.2f} FPS)")
            print(f"  -> Active Power        : {p_active:.3f} W (Dynamic: {dynamic_power_w:.3f} W)")
            print(f"  -> Idle Power          : {p_idle:.3f} W")
            print(f"  -> Voltage / Current   : {mean_volts:.2f} V / {mean_curr_ma:.1f} mA")
            print(f"  -> Energy per Frame    : {energy_per_frame_mj:.2f} mJ / frame")
            print(f"  -> Energy Efficiency   : {frames_per_joule:.2f} FPS/W (Frames / Joule)")
            print("  " + "-" * 55)

            rec = {
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Target_Device": "Raspberry Pi 5 (8GB)",
                "Model_Name": name,
                "Weights": weights,
                "Inferences": runs,
                "Total_Duration_s": round(inf_duration_s, 3),
                "Avg_Latency_ms": round(avg_latency_ms, 2),
                "Avg_FPS": round(fps, 2),
                "Idle_Power_W": round(p_idle, 3),
                "Active_Power_W": round(p_active, 3),
                "Dynamic_Power_W": round(dynamic_power_w, 3),
                "Total_Energy_J": round(total_energy_j, 3),
                "Energy_per_Frame_mJ": round(energy_per_frame_mj, 2),
                "Frames_per_Joule": round(frames_per_joule, 2),
                "Mean_Voltage_V": round(mean_volts, 2),
                "Mean_Current_mA": round(mean_curr_ma, 1),
            }
            save_power_row(MASTER_CSV, rec)
            save_power_row(RPI5_CSV, rec)
            results.append(rec)

            print("  [Cool-down] Pausing 5s for thermal stabilization...")
            time.sleep(5)

        # Summary Table
        print("\n" + "=" * 90)
        print("  RASPBERRY PI 5 (8GB, CORTEX-A76) POWER & ENERGY BENCHMARK SUMMARY")
        print("=" * 90)
        print(f"{'MODEL':<26} | {'LATENCY (ms)':<13} | {'FPS':<6} | {'POWER (W)':<10} | {'ENERGY (mJ)':<12} | {'EFF (FPS/W)':<11}")
        print("-" * 90)
        for r in results:
            eff = r['Avg_FPS'] / r['Active_Power_W'] if r['Active_Power_W'] > 0 else 0
            print(f"{r['Model_Name']:<26} | {r['Avg_Latency_ms']:<13.1f} | {r['Avg_FPS']:<6.2f} | {r['Active_Power_W']:<10.3f} | {r['Energy_per_Frame_mJ']:<12.1f} | {eff:<11.2f}")
        print("=" * 90)

    finally:
        ssh.close()
        device.close()
        print("\nConnections closed cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Joulescope Power Benchmark")
    parser.add_argument("--ip", type=str, default="172.20.10.4", help="IP address of Raspberry Pi 5")
    parser.add_argument("--user", type=str, default="rpi", help="SSH username")
    parser.add_argument("--password", type=str, default="rpi", help="SSH password")
    parser.add_argument("--runs", type=int, default=10, help="Number of inferences per model")
    parser.add_argument("--check-only", action="store_true", help="Only run pre-flight diagnostics")
    parser.add_argument("--scan", action="store_true", help="Scan subnet for devices")
    args = parser.parse_args()

    if args.scan:
        devices = scan_network_for_ssh()
        print("Active SSH hosts found:", devices)
        sys.exit(0)

    if args.check_only:
        js, ssh = preflight_check(args.ip, args.user, args.password)
        sys.exit(0 if (js and ssh) else 1)

    run_rpi5_power_benchmark(target_ip=args.ip, username=args.user, password=args.password, runs=args.runs)
