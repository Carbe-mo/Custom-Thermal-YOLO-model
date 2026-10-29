import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.55.1', username='acps', password='ACPS@123')

script = """
import time
import numpy as np
import onnxruntime as ort
from jtop import jtop

print('Testing ONNX Runtime CPU inference...')
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 8
session = ort.InferenceSession('/home/acps/thermal_yolo/weights/v30_champion_best.onnx', sess_options, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

# Warmup
print('Warming up...')
for _ in range(2):
    session.run(None, {input_name: dummy_input})

# Timed run with jtop
powers = []
times = []

with jtop() as jetson:
    for i in range(5):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
        if jetson.ok():
            p = jetson.power
            cpu_p = p.get('rail', {}).get('CPU', {}).get('power', 0)
            tot_p = p.get('tot', {}).get('power', 0)
            powers.append((cpu_p, tot_p))
        print(f"Run {i+1}: {dt:.2f} ms | CPU rail: {cpu_p} mW | Total: {tot_p} mW")

print(f"Mean Latency: {np.mean(times):.2f} ms ({1000.0/np.mean(times):.2f} FPS)")
"""

sftp = c.open_sftp()
with sftp.file('/tmp/test_cpu_jtop.py', 'w') as f:
    f.write(script)
sftp.close()

stdin, stdout, stderr = c.exec_command('OPENBLAS_CORETYPE=ARMV8 /home/acps/Yolo-SARD/venv/bin/python3 /tmp/test_cpu_jtop.py')
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    print('ERR:', err)

c.close()
