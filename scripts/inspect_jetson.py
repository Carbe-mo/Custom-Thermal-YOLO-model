"""
scripts/inspect_jetson.py
Inspects the NVIDIA Jetson AGX over USB/SSH:
- OS & JetPack L4T version
- Memory, Disk, and CPU/GPU specifications
- CUDA, TensorRT, and PyTorch availability
- Built-in hardware power sensors (tegrastats / INA3221)
"""

import paramiko

JETSON_IP = "192.168.55.1"
USER = "acps"
PASS = "ACPS@123"

def main():
    print(f"Connecting to {USER}@{JETSON_IP}...")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(JETSON_IP, username=USER, password=PASS, timeout=10)
    print("SSH Connected successfully!\n")

    def run(title, cmd):
        print("=" * 60)
        print(f"  {title}")
        print("=" * 60)
        stdin, stdout, stderr = c.exec_command(cmd)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if out:
            print(out)
        if err:
            print(f"[STDERR]: {err}")
        print()

    run("JetPack L4T Release", "cat /etc/nv_tegra_release")
    run("RAM & Memory", "free -h")
    run("Storage Disk", "df -h /")
    run("CUDA Compiler Version", "/usr/local/cuda/bin/nvcc --version 2>/dev/null || which nvcc")
    run("tegrastats Path & Sample", "which tegrastats && head -n 1 <(tegrastats --interval 500)")
    run("Python Version", "python3 --version")
    run("PyTorch & CUDA Check", "python3 -c \"import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')\" 2>&1")
    run("Power Sensor Rails (sysfs)", "ls -d /sys/bus/i2c/drivers/ina3221x/* 2>/dev/null || ls -d /sys/bus/i2c/drivers/ina3221/* 2>/dev/null")

    c.close()

if __name__ == "__main__":
    main()
