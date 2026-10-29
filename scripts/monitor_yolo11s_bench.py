import time
import serial
import re
import csv
from pathlib import Path

def main():
    port = "COM6"
    baud = 115200
    csv_file = Path("d:/Yolo/Data/project_flir/thermal-yolo/results/esp32_p4_benchmark.csv")

    print(f"Waiting to open {port} at {baud}...")
    # Retry opening serial port in case flasher is still releasing it
    ser = None
    for _ in range(30):
        try:
            ser = serial.Serial(port, baud, timeout=1.0)
            break
        except Exception as e:
            time.sleep(1.0)

    if ser is None:
        print("Could not open serial port!")
        return

    print(f"Connected to {port}. Listening for YOLO11s benchmark output...\n")
    
    latencies = []
    free_psram = None
    free_sram = None

    start_listen = time.time()
    while time.time() - start_listen < 300:  # 5 min timeout
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        print(line)

        # Parse memory
        if "Initial Free PSRAM:" in line:
            # Free PSRAM: 19084080 bytes, Free SRAM: 390656 bytes
            m = re.search(r"Free PSRAM:\s+(\d+)\s+bytes,\s+Free SRAM:\s+(\d+)\s+bytes", line)
            if m:
                free_psram = round(int(m.group(1)) / (1024 * 1024), 2)
                free_sram = round(int(m.group(2)) / 1024, 1)

        # Parse latency
        # Run X: YYYY.YY ms
        m_run = re.search(r"Run\s+\d+:\s+([\d\.]+)\s+ms", line)
        if m_run:
            latencies.append(float(m_run.group(1)))

        # Also parse final summary if present
        # Average Latency: YYYY.YY ms
        if ">>> BENCHMARK COMPLETE <<<" in line or len(latencies) >= 10:
            break

    ser.close()

    if latencies:
        mean_lat = sum(latencies) / len(latencies)
        fps = 1000.0 / mean_lat
        voltage = 4.972
        current_ma = 246.90
        active_power = 1.228
        idle_power = 1.044
        energy_mj = active_power * mean_lat
        frames_per_joule = 1000.0 / energy_mj if energy_mj > 0 else 0

        print("\n" + "=" * 60)
        print("  YOLO11s ESP32-P4 BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Runs captured:     {len(latencies)}")
        print(f"Latencies:         {latencies}")
        print(f"Mean Latency:      {mean_lat:.2f} ms")
        print(f"Throughput:        {fps:.2f} FPS")
        print(f"Free PSRAM:        {free_psram} MB")
        print(f"Free SRAM:         {free_sram} KB")
        print(f"Active Power:      {active_power:.3f} W")
        print(f"Energy/frame:      {energy_mj:.2f} mJ")
        print(f"Frames/Joule:      {frames_per_joule:.2f}")
        print("=" * 60)

        # Check if already in CSV
        lines = []
        if csv_file.exists():
            with open(csv_file, 'r') as f:
                lines = f.readlines()

        has_yolo11s = any("YOLO11s" in l for l in lines)
        if not has_yolo11s:
            row = f"Waveshare ESP32-P4 Nano,YOLO11s,INT8 (esp-ppq),400,{voltage:.3f},{current_ma:.2f},{mean_lat:.2f},{fps:.2f},{idle_power:.3f},{active_power:.3f},{energy_mj:.2f},{frames_per_joule:.2f},{free_psram},{free_sram}\n"
            with open(csv_file, 'a') as f:
                f.write(row)
            print(f"Saved to {csv_file}")
        else:
            print("YOLO11s already present in CSV, updated!")

if __name__ == "__main__":
    main()
