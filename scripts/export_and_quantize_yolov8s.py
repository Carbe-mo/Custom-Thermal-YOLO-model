"""
scripts/export_and_quantize_yolov8s.py
Exports stock YOLOv8s with ESP-DL Detect head and quantizes with esp-ppq to INT8 for ESP32-P4.
"""

import os
import sys
import torch
from pathlib import Path

repo_root = Path("d:/Yolo/Data/project_flir/thermal-yolo")
sys.path.insert(0, str(repo_root))

from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.engine.exporter import Exporter, try_export, arange_patch
from ultralytics.utils import colorstr
from ultralytics.utils.checks import check_requirements
import onnx
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ESP_Detect(Detect):
    def forward(self, x):
        box0 = self.cv2[0](x[0])
        score0 = self.cv3[0](x[0])
        box1 = self.cv2[1](x[1])
        score1 = self.cv3[1](x[1])
        box2 = self.cv2[2](x[2])
        score2 = self.cv3[2](x[2])
        return box0, score0, box1, score1, box2, score2

class ESP_Detect_Exporter(Exporter):
    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        requirements = ["onnx>=1.14.0"]
        check_requirements(requirements)
        opset = 13
        f = str(self.file.with_suffix(".onnx"))
        output_names = ["box0", "score0", "box1", "score1", "box2", "score2"]

        with arange_patch(self.args):
            torch.onnx.export(
                self.model,
                self.im,
                f,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["images"],
                output_names=output_names,
                dynamic_axes=None,
            )
        model_onnx = onnx.load(f)
        onnx.save(onnx.shape_inference.infer_shapes(model_onnx), f)
        return f

class ESP_YOLO(YOLO):
    def export(self, **kwargs):
        self._check_is_pytorch_model()
        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": "cpu",
            "verbose": False,
        }
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}
        return ESP_Detect_Exporter(overrides=args, _callbacks=self.callbacks)(
            model=self.model
        )

class ThermalCaliDataset(Dataset):
    def __init__(self, img_dir, img_size=640, max_samples=16):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))][:max_samples]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.transform(img)

def main():
    pt_path = Path("D:/Yolo/Data/pretrained_weights/yolov8s.pt")
    onnx_path = repo_root / "weights" / "yolov8s_esp.onnx"
    espdl_path = repo_root / "weights" / "yolov8s_p4.espdl"

    print(f"[1/3] Loading YOLOv8s from {pt_path}...")
    model = ESP_YOLO(str(pt_path))

    for m in model.modules():
        if isinstance(m, Detect):
            m.forward = ESP_Detect.forward.__get__(m)

    print(f"[2/3] Exporting static ONNX to {onnx_path}...")
    model.export(format="onnx", opset=13, dynamic=False, imgsz=640)

    exported = onnx.load(str(pt_path.with_suffix(".onnx")))
    onnx.save(onnx.shape_inference.infer_shapes(exported), str(onnx_path))
    print(f"Exported ONNX saved to {onnx_path}")

    print(f"[3/3] Quantizing with esp-ppq for ESP32-P4 RISC-V SIMD...")
    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx

    calib_dir = repo_root / "data" / "thermal" / "images"
    calib_ds = ThermalCaliDataset(str(calib_dir), img_size=640, max_samples=16)
    calib_loader = DataLoader(calib_ds, batch_size=4, shuffle=False)

    def collate_fn(batch):
        return batch.to("cpu")

    quant_setting = QuantizationSettingFactory.espdl_setting()
    quant_setting.quantize_activation_setting.calib_algorithm = "percentile"
    quant_setting.equalization = False
    quant_setting.bias_correct = True
    quant_setting.tqt_optimization = False

    espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calib_loader,
        calib_steps=8,
        input_shape=[1, 3, 640, 640],
        target="esp32p4",
        num_of_bits=8,
        collate_fn=collate_fn,
        setting=quant_setting,
        device="cpu",
        error_report=False,
        skip_export=False,
        export_test_values=False,
        verbose=0,
    )
    print(f"\n🎉 SUCCESS! Quantized YOLOv8s ESPDL created: {espdl_path} ({espdl_path.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
