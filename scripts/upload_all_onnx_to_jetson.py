import paramiko
import os

HOST = '192.168.55.1'
USER = 'acps'
PASS = 'ACPS@123'
REMOTE_DIR = '/home/acps/thermal_yolo/weights'

models = {
    'yolov8n.onnx': 'D:/Yolo/Data/project_flir/thermal-yolo/yolov8n.onnx',
    'yolov8s.onnx': 'D:/Yolo/Data/project_flir/thermal-yolo/yolov8s.onnx',
    'yolov10n.onnx': 'D:/Yolo/Data/project_flir/thermal-yolo/yolov10n.onnx',
    'yolov10s.onnx': 'D:/Yolo/Data/project_flir/thermal-yolo/yolov10s.onnx',
    'yolo11n.onnx': 'D:/Yolo/Data/project_flir/thermal-yolo/yolo11n.onnx',
    'yolo11s.onnx': 'D:/Yolo/Data/project_flir/thermal-yolo/yolo11s.onnx',
    'v35_dual_fusion_best.onnx': 'D:/Yolo/Data/project_flir/thermal-yolo/runs/v35_dual_fusion/train/weights/best.onnx'
}

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS)
sftp = client.open_sftp()

for name, local_path in models.items():
    remote_path = f"{REMOTE_DIR}/{name}"
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"Uploading {name} ({size_mb:.1f} MB)...")
    sftp.put(local_path, remote_path)
    print(f"Uploaded {name}")

sftp.close()
client.close()
print("All 7 models uploaded successfully!")
