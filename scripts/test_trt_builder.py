import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.55.1', username='acps', password='ACPS@123')

script = """
import tensorrt as trt
import sys

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled.")

    print(f"Building TensorRT engine for {onnx_file_path}...")
    engine = builder.build_engine(network, config)
    if engine:
        print("Engine built successfully!")
    else:
        print("Failed to build engine.")
    return engine

build_engine('/home/acps/thermal_yolo/weights/yolov8n.onnx')
"""

sftp = c.open_sftp()
with sftp.file('/tmp/test_trt_py.py', 'w') as f:
    f.write(script)
sftp.close()

stdin, stdout, stderr = c.exec_command('OPENBLAS_CORETYPE=ARMV8 /home/acps/Yolo-SARD/venv/bin/python3 /tmp/test_trt_py.py')
for line in stdout:
    print(line.strip())
err = stderr.read().decode()
if err:
    print('ERR:', err)

c.close()
