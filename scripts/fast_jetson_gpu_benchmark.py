import paramiko
import os
import re
import csv
import time

HOST = '192.168.55.1'
USER = 'acps'
PASS = 'ACPS@123'

remote_script = """
import os
import re
import csv
import time
import threading
import subprocess
import numpy as np
from jtop import jtop

models = [
    {"name": "Ours (Dilated ECA v30)", "engine": "/home/acps/thermal_yolo/weights/v30.engine"},
    {"name": "YOLOv8n", "engine": "/home/acps/thermal_yolo/weights/yolov8n.engine"},
    {"name": "YOLOv8s", "engine": "/home/acps/thermal_yolo/weights/yolov8s.engine"},
    {"name": "YOLO11n", "engine": "/home/acps/thermal_yolo/weights/yolo11n.engine"},
    {"name": "YOLO11s", "engine": "/home/acps/thermal_yolo/weights/yolo11s.engine"},
    {"name": "Ours Dual-Backbone Fusion (v35)", "engine": "/home/acps/thermal_yolo/weights/v35.engine"},
]

output_csv = "/home/acps/thermal_yolo/jetson_gpu_benchmark.csv"
results = []

# Idle baseline power measurement (5 seconds)
print("--- Measuring Idle Baseline Power (5 seconds) ---")
idle_gpu_powers = []
idle_tot_powers = []
with jtop() as jetson:
    for _ in range(25):
        if jetson.ok():
            p = jetson.power
            idle_gpu_powers.append(p.get('rail', {}).get('GPU', {}).get('power', 0))
            idle_tot_powers.append(p.get('tot', {}).get('power', 0))
        time.sleep(0.2)

idle_gpu_w = np.mean(idle_gpu_powers) / 1000.0 if idle_gpu_powers else 0.027
idle_tot_w = np.mean(idle_tot_powers) / 1000.0 if idle_tot_powers else 3.599
print(f"Idle GPU Rail: {idle_gpu_w:.3f} W | Idle Total Board: {idle_tot_w:.3f} W")

print("\\n=== Starting Jetson AGX Xavier GPU Benchmark (Volta TensorRT FP16) ===")

for item in models:
    name = item["name"]
    engine_path = item["engine"]

    if not os.path.exists(engine_path):
        print(f"Engine not found: {engine_path}")
        continue

    print(f"\\n--- Benchmarking {name} on Volta GPU ---")
    bench_cmd = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--loadEngine={engine_path}",
        "--iterations=10",
        "--warmUp=200",
        "--duration=3"
    ]

    gpu_powers = []
    tot_powers = []
    cpu_powers = []
    stop_event = threading.Event()

    def sample_power():
        try:
            with jtop() as jetson:
                while not stop_event.is_set():
                    if jetson.ok():
                        p = jetson.power
                        gpu_powers.append(p.get('rail', {}).get('GPU', {}).get('power', 0))
                        tot_powers.append(p.get('tot', {}).get('power', 0))
                        cpu_powers.append(p.get('rail', {}).get('CPU', {}).get('power', 0))
                    time.sleep(0.1)
        except Exception as e:
            print("jtop sampler error:", e)

    pwr_thread = threading.Thread(target=sample_power)
    pwr_thread.start()

    bench_res = subprocess.run(bench_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stop_event.set()
    pwr_thread.join(timeout=2.0)

    stdout_text = bench_res.stdout

    # Parse latency from trtexec stdout:
    # Look for mean latency under GPU Compute
    mean_lat = None
    min_lat = None
    max_lat = None
    in_gpu_compute = False

    for line in stdout_text.splitlines():
        if "GPU Compute" in line:
            in_gpu_compute = True
            continue
        if in_gpu_compute:
            m_mean = re.search(r"mean[:\s=]+([0-9\.]+)\s*ms", line)
            if m_mean:
                mean_lat = float(m_mean.group(1))
            m_min = re.search(r"min[:\s=]+([0-9\.]+)\s*ms", line)
            if m_min:
                min_lat = float(m_min.group(1))
            m_max = re.search(r"max[:\s=]+([0-9\.]+)\s*ms", line)
            if m_max:
                max_lat = float(m_max.group(1))
            if "total compute time" in line or "PASSED" in line:
                in_gpu_compute = False

    if mean_lat is None:
        # Fallback search across all lines
        for line in stdout_text.splitlines():
            m = re.search(r"mean[:\s=]+([0-9\.]+)\s*ms", line)
            if m:
                mean_lat = float(m.group(1))
                break

    if mean_lat is None:
        print(f"Failed to parse latency for {name}")
        continue

    fps = 1000.0 / mean_lat
    active_gpu_w = np.mean(gpu_powers) / 1000.0 if gpu_powers else 0.0
    active_tot_w = np.mean(tot_powers) / 1000.0 if tot_powers else 0.0
    active_cpu_w = np.mean(cpu_powers) / 1000.0 if cpu_powers else 0.0

    energy_mj_frame = active_tot_w * (mean_lat / 1000.0) * 1000.0
    active_energy_mj_frame = max(0.0, (active_tot_w - idle_tot_w)) * (mean_lat / 1000.0) * 1000.0
    frames_per_joule = fps / active_tot_w if active_tot_w > 0 else 0.0

    print(f"  Summary for {name}:")
    print(f"    GPU Latency: {mean_lat:.2f} ms (min: {min_lat}, max: {max_lat}) -> {fps:.2f} FPS")
    print(f"    Active Board Power: {active_tot_w:.3f} W (GPU Rail: {active_gpu_w:.3f} W, CPU Rail: {active_cpu_w:.3f} W)")
    print(f"    Total Energy/Frame: {energy_mj_frame:.2f} mJ/frame (Active: {active_energy_mj_frame:.2f} mJ/frame)")
    print(f"    Energy Efficiency: {frames_per_joule:.2f} Frames/Joule")

    results.append({
        "Device": "Jetson AGX Xavier (GPU - Volta FP16)",
        "Model": name,
        "Latency_ms": round(mean_lat, 2),
        "Min_Latency_ms": round(min_lat, 2) if min_lat else round(mean_lat, 2),
        "Max_Latency_ms": round(max_lat, 2) if max_lat else round(mean_lat, 2),
        "Throughput_FPS": round(fps, 2),
        "Idle_Board_Power_W": round(idle_tot_w, 3),
        "Active_Board_Power_W": round(active_tot_w, 3),
        "GPU_Rail_Power_W": round(active_gpu_w, 3),
        "Total_Energy_mJ_frame": round(energy_mj_frame, 2),
        "Active_Energy_mJ_frame": round(active_energy_mj_frame, 2),
        "Frames_per_Joule": round(frames_per_joule, 2),
    })

if results:
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\\nAll GPU benchmarks finished! Saved to {output_csv}")
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)

# Upload script
sftp = c.open_sftp()
with sftp.file('/home/acps/thermal_yolo/run_fast_gpu_benchmark.py', 'w') as f:
    f.write(remote_script)
sftp.close()

# Execute script with unbuffered output
stdin, stdout, stderr = c.exec_command('OPENBLAS_CORETYPE=ARMV8 /home/acps/Yolo-SARD/venv/bin/python3 -u /home/acps/thermal_yolo/run_fast_gpu_benchmark.py')
for line in stdout:
    print(line, end='', flush=True)

err = stderr.read().decode()
if err:
    print("STDERR:", err)

# Download CSV
sftp = c.open_sftp()
local_csv = os.path.join('D:', os.sep, 'Yolo', 'Data', 'project_flir', 'thermal-yolo', 'results', 'jetson_gpu_benchmark.csv')
os.makedirs(os.path.dirname(local_csv), exist_ok=True)
sftp.get('/home/acps/thermal_yolo/jetson_gpu_benchmark.csv', local_csv)
sftp.close()
c.close()

print(f"GPU CSV downloaded to {local_csv}")
