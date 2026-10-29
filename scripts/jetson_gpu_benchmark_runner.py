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
    {"name": "Ours (Dilated ECA v30)", "onnx": "/home/acps/thermal_yolo/weights/v30_champion_best.onnx", "engine": "/home/acps/thermal_yolo/weights/v30.engine"},
    {"name": "YOLOv8n", "onnx": "/home/acps/thermal_yolo/weights/yolov8n.onnx", "engine": "/home/acps/thermal_yolo/weights/yolov8n.engine"},
    {"name": "YOLOv8s", "onnx": "/home/acps/thermal_yolo/weights/yolov8s.onnx", "engine": "/home/acps/thermal_yolo/weights/yolov8s.engine"},
    {"name": "YOLOv10n", "onnx": "/home/acps/thermal_yolo/weights/yolov10n.onnx", "engine": "/home/acps/thermal_yolo/weights/yolov10n.engine"},
    {"name": "YOLOv10s", "onnx": "/home/acps/thermal_yolo/weights/yolov10s.onnx", "engine": "/home/acps/thermal_yolo/weights/yolov10s.engine"},
    {"name": "YOLO11n", "onnx": "/home/acps/thermal_yolo/weights/yolo11n.onnx", "engine": "/home/acps/thermal_yolo/weights/yolo11n.engine"},
    {"name": "YOLO11s", "onnx": "/home/acps/thermal_yolo/weights/yolo11s.onnx", "engine": "/home/acps/thermal_yolo/weights/yolo11s.engine"},
    {"name": "Ours Dual-Backbone Fusion (v35)", "onnx": "/home/acps/thermal_yolo/weights/v35_dual_fusion_best.onnx", "engine": "/home/acps/thermal_yolo/weights/v35.engine"},
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

idle_gpu_w = np.mean(idle_gpu_powers) / 1000.0 if idle_gpu_powers else 0.3
idle_tot_w = np.mean(idle_tot_powers) / 1000.0 if idle_tot_powers else 3.8
print(f"Idle GPU Rail: {idle_gpu_w:.3f} W | Idle Total Board: {idle_tot_w:.3f} W")

print("\\n=== Starting Jetson AGX Xavier GPU Benchmark (512-Core Volta GPU) ===")

for item in models:
    name = item["name"]
    onnx_path = item["onnx"]
    engine_path = item["engine"]

    print(f"\\n=======================================================")
    print(f"Processing {name}...")

    # Step 1: Build TensorRT Engine if not already built
    if not os.path.exists(engine_path):
        print(f"Building TensorRT FP16 Engine for {name}...")
        t_build_start = time.time()
        build_cmd = [
            "/usr/src/tensorrt/bin/trtexec",
            f"--onnx={onnx_path}",
            "--fp16",
            "--workspace=1024",
            "--minTiming=1",
            "--avgTiming=1",
            f"--saveEngine={engine_path}",
            "--buildOnly"
        ]
        res = subprocess.run(build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        t_build = time.time() - t_build_start
        if res.returncode != 0 or not os.path.exists(engine_path):
            print(f"FAILED to build engine for {name} (code {res.returncode}):")
            print(res.stderr[-500:])
            continue
        print(f"Engine successfully built and saved in {t_build:.1f}s: {engine_path}")
    else:
        print(f"Found existing engine: {engine_path}")

    # Step 2: Benchmark Engine with simultaneous jtop power measurement
    print(f"Benchmarking {name} on Volta GPU...")
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

    # Parse latency and FPS from trtexec stdout
    # Example lines:
    # [05/06/2026-20:00:00] [I] GPU Compute Time: min = 4.12 ms, max = 5.21 ms, mean = 4.35 ms, median = 4.31 ms, percentile(99%) = 5.12 ms
    # [05/06/2026-20:00:00] [I] Throughput: 229.885 qps
    mean_lat = None
    fps = None

    for line in stdout_text.splitlines():
        if "GPU Compute Time:" in line and "mean =" in line:
            m = re.search(r"mean\s*=\s*([0-9\.]+)\s*ms", line)
            if m:
                mean_lat = float(m.group(1))
        elif "Throughput:" in line:
            m = re.search(r"Throughput:\s*([0-9\.]+)\s*qps", line)
            if m:
                fps = float(m.group(1))

    if mean_lat is None:
        # Fallback to Host Latency or End-to-End
        for line in stdout_text.splitlines():
            if "mean =" in line and "ms" in line:
                m = re.search(r"mean\s*=\s*([0-9\.]+)\s*ms", line)
                if m:
                    mean_lat = float(m.group(1))
                    break

    if fps is None and mean_lat is not None and mean_lat > 0:
        fps = 1000.0 / mean_lat

    if mean_lat is None:
        print(f"Could not parse metrics for {name}. Output tail:")
        print(stdout_text[-500:])
        continue

    active_gpu_w = np.mean(gpu_powers) / 1000.0 if gpu_powers else 0.0
    active_tot_w = np.mean(tot_powers) / 1000.0 if tot_powers else 0.0
    active_cpu_w = np.mean(cpu_powers) / 1000.0 if cpu_powers else 0.0

    energy_mj_frame = active_tot_w * (mean_lat / 1000.0) * 1000.0
    active_energy_mj_frame = max(0.0, (active_tot_w - idle_tot_w)) * (mean_lat / 1000.0) * 1000.0
    frames_per_joule = fps / active_tot_w if active_tot_w > 0 else 0.0

    print(f"  Result for {name}:")
    print(f"    GPU Latency: {mean_lat:.2f} ms ({fps:.2f} FPS)")
    print(f"    Active Total Power: {active_tot_w:.3f} W (GPU rail: {active_gpu_w:.3f} W)")
    print(f"    Total Energy/Frame: {energy_mj_frame:.2f} mJ/frame (Active: {active_energy_mj_frame:.2f} mJ/frame)")
    print(f"    Energy Efficiency: {frames_per_joule:.2f} Frames/Joule")

    results.append({
        "Device": "Jetson AGX Xavier (GPU - Volta FP16)",
        "Model": name,
        "Latency_ms": round(mean_lat, 2),
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
with sftp.file('/home/acps/thermal_yolo/run_gpu_benchmark.py', 'w') as f:
    f.write(remote_script)
sftp.close()

# Execute script with unbuffered output
stdin, stdout, stderr = c.exec_command('OPENBLAS_CORETYPE=ARMV8 /home/acps/Yolo-SARD/venv/bin/python3 -u /home/acps/thermal_yolo/run_gpu_benchmark.py')
for line in stdout:
    print(line, end='', flush=True)

err = stderr.read().decode()
if err:
    print("STDERR:", err)

# Download CSV
sftp = c.open_sftp()
local_csv = 'D:/Yolo/Data/project_flir\thermal-yolo/results/jetson_gpu_benchmark.csv'
os.makedirs(os.path.dirname(local_csv), exist_ok=True)
sftp.get('/home/acps/thermal_yolo/jetson_gpu_benchmark.csv', local_csv)
sftp.close()
c.close()

print(f"GPU CSV downloaded to {local_csv}")
