"""
scripts/test_torchscript_jetson.py
Tests TorchScript model loading and execution on both CPU and GPU (CUDA) on Jetson AGX Xavier.
"""

import paramiko

JETSON_IP = "192.168.55.1"
USER = "acps"
PASS = "ACPS@123"

def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(JETSON_IP, username=USER, password=PASS, timeout=10)

    remote_code = """
import time
import torch

weights_path = '/home/acps/thermal_yolo/weights/v30_champion_best.torchscript'
print('Loading TorchScript model:', weights_path)
model = torch.jit.load(weights_path)
model.eval()

# --- 1. GPU (CUDA) TEST ---
print('\\n[1] Testing GPU (CUDA)...')
model_gpu = model.to('cuda')
x_gpu = torch.zeros(1, 3, 640, 640, device='cuda')
with torch.no_grad():
    for _ in range(2):
        _ = model_gpu(x_gpu)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = model_gpu(x_gpu)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
gpu_ms = (t1 - t0) / 10.0 * 1000.0
gpu_fps = 1000.0 / gpu_ms
print(f'  -> GPU (CUDA) Mean Latency: {gpu_ms:.2f} ms ({gpu_fps:.2f} FPS) 🚀')

# --- 2. CPU TEST ---
print('\\n[2] Testing CPU (Carmel 8-Core)...')
model_cpu = model.to('cpu')
x_cpu = torch.zeros(1, 3, 640, 640, device='cpu')
with torch.no_grad():
    for _ in range(2):
        _ = model_cpu(x_cpu)
    t0 = time.perf_counter()
    for _ in range(5):
        _ = model_cpu(x_cpu)
    t1 = time.perf_counter()
cpu_ms = (t1 - t0) / 5.0 * 1000.0
cpu_fps = 1000.0 / cpu_ms
print(f'  -> CPU Mean Latency: {cpu_ms:.2f} ms ({cpu_fps:.2f} FPS)')
"""

    cmd = f"OPENBLAS_CORETYPE=ARMV8 /home/acps/Yolo-SARD/venv/bin/python -c \"{remote_code}\""
    stdin, stdout, stderr = c.exec_command(cmd)
    print(stdout.read().decode().strip())
    err = stderr.read().decode().strip()
    if err:
        print("[ERR]:", err)

    c.close()

if __name__ == "__main__":
    main()
