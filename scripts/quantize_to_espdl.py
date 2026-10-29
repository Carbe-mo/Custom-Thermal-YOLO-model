import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq import QuantizationSettingFactory

class ThermalCalibDataset(Dataset):
    def __init__(self, img_dir, img_size=(640, 640), max_samples=32):
        self.img_paths = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    self.img_paths.append(os.path.join(root, f))
                    if len(self.img_paths) >= max_samples:
                        break
            if len(self.img_paths) >= max_samples:
                break
        self.img_size = img_size
        print(f"Loaded {len(self.img_paths)} calibration images from {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dim
        return torch.from_numpy(img)

def quantize_model(onnx_path, espdl_path, img_dir='data/thermal/images', img_size=640, target="esp32p4"):
    print(f"\n--- Quantizing {onnx_path} to {espdl_path} for target {target} ---")
    dataset = ThermalCalibDataset(img_dir, img_size=(img_size, img_size), max_samples=32)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    quant_setting = QuantizationSettingFactory.espdl_setting()

    quant_graph = espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=espdl_path,
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=[1, 3, img_size, img_size],
        target=target,
        quant_type="w8a8",
        setting=quant_setting,
        device="cpu"
    )
    print(f"Quantization complete! Saved to {espdl_path}")
    if os.path.exists(espdl_path):
        size_mb = os.path.getsize(espdl_path) / (1024 * 1024)
        print(f"Output ESP-DL model size: {size_mb:.2f} MB")

if __name__ == "__main__":
    onnx_in = "weights/v30_champion_best.onnx"
    espdl_out = "weights/v30_champion_p4.espdl"
    quantize_model(onnx_in, espdl_out)
