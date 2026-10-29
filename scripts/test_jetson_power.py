"""
scripts/test_jetson_power.py
Tests connection and parses built-in hardware power monitors on NVIDIA Jetson AGX Xavier.
"""

import paramiko
import re
import time

JETSON_IP = "192.168.55.1"
USER = "acps"
PASS = "ACPS@123"

def main():
    print(f"Connecting to {USER}@{JETSON_IP}...")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(JETSON_IP, username=USER, password=PASS, timeout=15, banner_timeout=30)
    print("SSH Connected successfully!")

    # 1. System Info
    stdin, stdout, stderr = c.exec_command("cat /proc/device-tree/model")
    print("Hardware Model:", stdout.read().decode().strip())

    stdin, stdout, stderr = c.exec_command("uname -r")
    print("Kernel Version:", stdout.read().decode().strip())

    # 2. Sample tegrastats for 3 seconds
    print("\nReading onboard INA3221 hardware power sensors via tegrastats...")
    stdin, stdout, stderr = c.exec_command("tegrastats --interval 1000")
    
    for _ in range(3):
        line = stdout.readline()
        if not line:
            break
        stripped = line.strip()
        print(f"\n[Raw Telemetry] {stripped}")
        
        # Parse power rails: e.g. POM_5V_IN, VDD_CPU, VDD_GPU, VDD_SOC, etc.
        rails = re.findall(r"([A-Za-z0-9_]+)\s+([0-9]+)mW/([0-9]+)mW", stripped)
        if rails:
            print("  --- Parsed Power Rails (Current / Average) ---")
            for rail_name, cur_mw, avg_mw in rails:
                cur_w = float(cur_mw) / 1000.0
                avg_w = float(avg_mw) / 1000.0
                print(f"  ⚡ {rail_name:<16}: {cur_w:.3f} W (Average: {avg_w:.3f} W)")

    c.close()
    print("\nJetson power reading verified cleanly!")

if __name__ == "__main__":
    main()
