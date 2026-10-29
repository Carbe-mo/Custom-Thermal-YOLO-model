"""
scripts/test_jtop.py
Tests jtop power monitoring and CUDA inference on Jetson AGX Xavier.
"""

import paramiko

JETSON_IP = "192.168.55.1"
USER = "acps"
PASS = "ACPS@123"

def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(JETSON_IP, username=USER, password=PASS, timeout=10)
    print("Connected to Jetson AGX Xavier!\n")

    remote_code = """
import time
from jtop import jtop
import torch

print('Torch version:', torch.__version__, '| CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name(0))

with jtop() as jetson:
    if jetson.ok():
        print('jtop is active!')
        print('Hardware Board Info:', jetson.board.get('hardware', {}))
        for _ in range(3):
            p = jetson.power
            print('  ⚡ Total Power (mW):', p.get('tot', {}).get('power', 'N/A'), '| Rail breakdown:', {k: v.get('power') for k, v in p.get('rail', {}).items()})
            time.sleep(1.0)
    else:
        print('jtop service is not running. Try: sudo systemctl start jtop.service')
"""
    cmd = f"OPENBLAS_CORETYPE=ARMV8 /home/acps/Yolo-SARD/venv/bin/python -c \"{remote_code}\""
    stdin, stdout, stderr = c.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    if err:
        print(f"[ERR]: {err}")

    c.close()

if __name__ == "__main__":
    main()
