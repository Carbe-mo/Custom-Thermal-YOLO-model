"""
scripts/test_onnx_jetson.py
Checks ONNX Runtime execution providers (CPU and CUDA/TensorRT) on Jetson AGX Xavier.
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
import onnxruntime as ort
import torch
print('ONNX Runtime Version:', ort.__version__)
print('Available Providers:', ort.get_available_providers())
print('PyTorch Version:', torch.__version__, '| CUDA:', torch.cuda.is_available())
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
