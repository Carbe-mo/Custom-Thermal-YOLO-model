"""
scripts/measure_esp32_p4_joulescope.py
High-precision live Joulescope JS220 power measurement tool for Waveshare ESP32-P4 Nano.
"""

import time
import argparse
import numpy as np
import pyjoulescope_driver

def measure(duration_sec=30.0, output_csv="d:/Yolo/Data/project_flir/thermal-yolo/results/esp32_p4_benchmark.csv"):
    print("=" * 65)
    print("  JOULESCOPE JS220 PRECISION POWER BENCHMARK: ESP32-P4 NANO")
    print("=" * 65)

    d = pyjoulescope_driver.Driver()
    devices = d.device_paths()
    if not devices:
        raise RuntimeError("No Joulescope devices found!")
    
    dev = devices[0]
    print(f"  Connected to Joulescope: {dev}")
    d.open(dev)

    readings = []
    accum_energy_start = None
    accum_energy_end = None

    def on_stats(topic, value):
        nonlocal accum_energy_start, accum_energy_end
        try:
            i = value['signals']['current']['avg']['value'] * 1000.0  # mA
            v = value['signals']['voltage']['avg']['value']          # V
            p = value['signals']['power']['avg']['value'] * 1000.0    # mW
            e = value['accumulators']['energy']['value']             # J
            if accum_energy_start is None:
                accum_energy_start = e
            accum_energy_end = e
            readings.append((time.time(), v, i, p, e))
            print(f"  ⚡ Live: {p:6.1f} mW | {v:5.3f} V | {i:6.1f} mA | Energy: {e - accum_energy_start:6.2f} J", flush=True)
        except Exception as err:
            pass

    d.subscribe(f"{dev}/s/stats/value", pyjoulescope_driver.SubscribeFlags.PUB, on_stats)
    d.publish(f"{dev}/s/stats/ctrl", 1)

    print(f"\n[RECORDING STARTED] Capturing high-speed samples for {duration_sec:.1f} seconds...")
    print(">> If restarting the benchmark run, PRESS THE 'RESET' (RST/EN) BUTTON on ESP32-P4 NOW! <<\n")

    t_start = time.time()
    while time.time() - t_start < duration_sec:
        time.sleep(0.2)

    d.publish(f"{dev}/s/stats/ctrl", 0)
    d.close(dev)

    if not readings:
        print("No readings collected!")
        return

    vs = [r[1] for r in readings]
    is_ma = [r[2] for r in readings]
    ps_mw = [r[3] for r in readings]
    total_energy_j = (accum_energy_end - accum_energy_start) if accum_energy_end is not None and accum_energy_start is not None else 0.0

    mean_v = float(np.mean(vs))
    mean_i = float(np.mean(is_ma))
    mean_p_w = float(np.mean(ps_mw)) / 1000.0
    peak_p_w = float(np.max(ps_mw)) / 1000.0
    min_p_w = float(np.min(ps_mw)) / 1000.0

    print("\n" + "=" * 65)
    print("  ESP32-P4 NANO JOULESCOPE MEASUREMENT SUMMARY")
    print("=" * 65)
    print(f"  Capture Duration:      {len(readings)*0.5:.1f} s ({len(readings)} stat windows)")
    print(f"  Mean Voltage:          {mean_v:.3f} V")
    print(f"  Mean Current:          {mean_i:.2f} mA (Min: {np.min(is_ma):.2f} mA, Max: {np.max(is_ma):.2f} mA)")
    print(f"  Mean Power:            {mean_p_w*1000.0:.2f} mW ({mean_p_w:.4f} W)")
    print(f"  Active Peak Power:     {peak_p_w*1000.0:.2f} mW ({peak_p_w:.4f} W)")
    print(f"  Baseline Min Power:    {min_p_w*1000.0:.2f} mW ({min_p_w:.4f} W)")
    print(f"  Total Energy Expended: {total_energy_j:.3f} J ({total_energy_j*1000.0:.1f} mJ)")
    print("=" * 65)

    # Calculate per-frame energy for Ours Champion (2173.66 ms)
    champion_lat_s = 2.17366
    energy_per_frame_mj = mean_p_w * champion_lat_s * 1000.0
    fps = 1.0 / champion_lat_s
    frames_per_joule = 1.0 / (mean_p_w * champion_lat_s) if mean_p_w > 0 else 0

    print(f"  Energy per Frame (v30): {energy_per_frame_mj:.2f} mJ")
    print(f"  Frames per Joule (v30): {frames_per_joule:.2f} Frames/J")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESP32-P4 Joulescope Measurement")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    args = parser.parse_args()
    measure(duration_sec=args.duration)
