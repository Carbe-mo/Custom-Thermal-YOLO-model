import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.55.1', username='acps', password='ACPS@123')

script = """
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    print("PyCUDA is available! GPU:", cuda.Device(0).name())
except Exception as e:
    print("PyCUDA not available:", e)
"""

sftp = c.open_sftp()
with sftp.file('/tmp/check_pycuda.py', 'w') as f:
    f.write(script)
sftp.close()

stdin, stdout, stderr = c.exec_command('OPENBLAS_CORETYPE=ARMV8 /home/acps/Yolo-SARD/venv/bin/python3 /tmp/check_pycuda.py')
print(stdout.read().decode())
print(stderr.read().decode())

c.close()
