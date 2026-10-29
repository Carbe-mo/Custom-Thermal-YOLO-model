"""
scripts/setup_jetson.py
Searches for existing PyTorch wheels and environments on Jetson AGX Xavier.
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

    def run(title, cmd):
        print(f"=== {title} ===")
        stdin, stdout, stderr = c.exec_command(cmd)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if out:
            print(out)
        if err:
            print(f"[ERR]: {err}")
        print()

    run("Search for .whl files", "find /home/acps -name '*.whl' 2>/dev/null")
    run("Check .py3venv", "/home/acps/.py3venv/bin/python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\" 2>&1")
    run("Test CUDA tensor allocation on Xavier GPU", "OPENBLAS_CORETYPE=ARMV8 /home/acps/Yolo-SARD/venv/bin/python -c \"import torch; x = torch.randn(1, 3, 640, 640).cuda(); print('SUCCESS! GPU Tensor allocated:', x.device, x.shape)\" 2>&1")

    c.close()

if __name__ == "__main__":
    main()
