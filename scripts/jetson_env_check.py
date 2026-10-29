import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.55.1', username='acps', password='ACPS@123')

remote_script = """
import os
for p in ['onnx', 'onnxruntime', 'torch', 'cv2', 'jtop', 'numpy', 'tensorrt']:
    try:
        m = __import__(p)
        print(f"{p}: {getattr(m, '__version__', 'installed')}")
    except Exception as e:
        print(f"{p}: NOT INSTALLED ({e})")
"""

sftp = c.open_sftp()
with sftp.file('/tmp/check_env.py', 'w') as f:
    f.write(remote_script)
sftp.close()

stdin, stdout, stderr = c.exec_command('OPENBLAS_CORETYPE=ARMV8 /home/acps/Yolo-SARD/venv/bin/python3 /tmp/check_env.py')
print("VENV:")
print(stdout.read().decode())
print(stderr.read().decode())

stdin, stdout, stderr = c.exec_command('OPENBLAS_CORETYPE=ARMV8 python3 /tmp/check_env.py')
print("SYSTEM:")
print(stdout.read().decode())
print(stderr.read().decode())

c.close()
