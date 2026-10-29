"""
scripts/setup_rpi5.py
Inspects and configures Raspberry Pi 5 over SSH:
- Installs or verifies PyTorch and Ultralytics
- Clones / uploads the thermal-yolo repository
- Uploads model weights and measure_edge_power.py
"""

import sys
import paramiko
from pathlib import Path

RPI_IP = "192.168.137.216"
USER = "rpi"
PASS = "rpi"

def main():
    print(f"Connecting to {USER}@{RPI_IP}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(RPI_IP, username=USER, password=PASS, timeout=10)
    print("SSH Connected successfully!")

    # 1. Verify PyTorch
    code = "import torch, torchvision; print('PyTorch Version:', torch.__version__, '| Torchvision:', torchvision.__version__)"
    stdin, stdout, stderr = client.exec_command(f"python3 -c \"{code}\"")
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print("PyTorch Status:", out if out else f"Error: {err}")

    # 2. Check Ultralytics
    stdin, stdout, stderr = client.exec_command("python3 -c \"import ultralytics; print('Ultralytics:', ultralytics.__version__)\"")
    out = stdout.read().decode().strip()
    print("Ultralytics Status:", out if out else "Not installed yet")

    if not out:
        print("\nInstalling Ultralytics (without torch reinstall)...")
        stdin, stdout, stderr = client.exec_command("pip3 install ultralytics --no-deps --break-system-packages")
        print(stdout.read().decode())
        # Install remaining lightweight dependencies
        stdin, stdout, stderr = client.exec_command("pip3 install pyyaml matplotlib psutil opencv-python-headless --break-system-packages")
        print(stdout.read().decode())

    # 3. Create repository directory on RPi 5
    remote_repo = "/home/rpi/Custom-Thermal-YOLO-model"
    stdin, stdout, stderr = client.exec_command(f"mkdir -p {remote_repo}/weights {remote_repo}/scripts {remote_repo}/common/modules {remote_repo}/results")
    print(f"Directory structure created at {remote_repo}")

    # 4. Upload common modules and measure_edge_power.py
    sftp = client.open_sftp()
    print("Uploading scripts and common modules...")
    sftp.put("scripts/measure_edge_power.py", f"{remote_repo}/scripts/measure_edge_power.py")
    for m in ["c2f_eca.py", "c2f_eca_dilated.py", "fusion.py"]:
        local_p = Path("common/modules") / m
        if local_p.exists():
            sftp.put(str(local_p), f"{remote_repo}/common/modules/{m}")
    print("Core modules uploaded.")

    # 5. Upload weights
    for w in ["v30_champion_best.pt", "v35_dual_fusion_best.pt", "yolov8n.pt", "yolov8s.pt", "yolov10n.pt", "yolov10s.pt", "yolo11n.pt", "yolo11s.pt"]:
        local_p = Path("weights") / w if (Path("weights") / w).exists() else Path(w)
        if local_p.exists():
            print(f"Uploading {w}...")
            sftp.put(str(local_p), f"{remote_repo}/weights/{w}")

    sftp.close()
    client.close()
    print("\nAll setup complete on Raspberry Pi 5!")

if __name__ == "__main__":
    main()
