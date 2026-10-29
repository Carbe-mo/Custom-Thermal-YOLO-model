import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.55.1', username='acps', password='ACPS@123')

script = """
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    x = torch.randn(1, 3, 640, 640, device='cuda', dtype=torch.float16)
    conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda().half()
    out = conv(x)
    print('CUDA Conv2d FP16 successful! Output shape:', out.shape)
"""

sftp = c.open_sftp()
with sftp.file('/tmp/test_torch_cuda.py', 'w') as f:
    f.write(script)
sftp.close()

stdin, stdout, stderr = c.exec_command('OPENBLAS_CORETYPE=ARMV8 /home/acps/Yolo-SARD/venv/bin/python3 /tmp/test_torch_cuda.py')
print(stdout.read().decode())
print(stderr.read().decode())

c.close()
