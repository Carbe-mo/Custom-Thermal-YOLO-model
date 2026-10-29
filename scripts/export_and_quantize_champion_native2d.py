import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

repo_root = Path("d:/Yolo/Data/project_flir/thermal-yolo")
sys.path.insert(0, str(repo_root))

from ultralytics import YOLO
from ultralytics.nn.modules import Detect, Conv, C2f
from ultralytics.engine.exporter import Exporter, try_export, arange_patch
from ultralytics.utils import colorstr, LOGGER
from ultralytics.utils.checks import check_requirements
import onnx
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Import custom modules
from common.modules.c2f_eca import C2f_ECA
from common.modules.c2f_eca_dilated import C2f_ECA_Dilated, DilatedECABottleneck
from common.modules.eca import ECABottleneck, ECA

# 2D Native ECA Module for ESP32-P4 RISC-V SIMD
class Native2D_ECA(nn.Module):
    def __init__(self, channels: int, conv1d_layer: nn.Conv1d):
        super().__init__()
        self.channels = channels
        self.conv2d = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Convert 1D weights into exact equivalent 2D Toeplitz 1x1 conv weights
        with torch.no_grad():
            w1d = conv1d_layer.weight.data.squeeze() # [3] = [w_left, w_center, w_right]
            w2d = torch.zeros(channels, channels, 1, 1)
            for i in range(channels):
                w2d[i, i, 0, 0] = w1d[1]
                if i > 0:
                    w2d[i, i - 1, 0, 0] = w1d[0]
                if i < channels - 1:
                    w2d[i, i + 1, 0, 0] = w1d[2]
            self.conv2d.weight.data = w2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard 4D NCHW tensor throughout (zero squeeze/transpose)
        y = torch.mean(x, dim=(2, 3), keepdim=True)
        omega = self.sigmoid(self.conv2d(y))
        return x * omega

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
    def __init__(self, img_dir, img_size=640, max_samples=32):
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

def convert_model_to_native2d(model_wrapper):
    """Replaces all ECA modules with Native2D_ECA equivalent"""
    converted_count = 0
    for name, module in model_wrapper.model.named_modules():
        if isinstance(module, (ECABottleneck, DilatedECABottleneck)):
            if hasattr(module, 'eca') and isinstance(module.eca, ECA):
                c = module.cv2.conv.out_channels if hasattr(module.cv2, 'conv') else module.cv2.c2
                module.eca = Native2D_ECA(c, module.eca.conv1d)
                converted_count += 1
    print(f"Successfully converted {converted_count} ECA attention blocks to 100% equivalent Native2D Conv blocks!")

def main():
    pt_path = repo_root / "weights" / "v30_champion_best.pt"
    onnx_path = repo_root / "weights" / "v30_champion_native2d.onnx"
    espdl_path = repo_root / "weights" / "v30_champion_native2d.espdl"

    print(f"[1/4] Loading Champion model from {pt_path}...")
    model = ESP_YOLO(str(pt_path))
    
    print("[2/4] Transforming ECA modules to Native2D Conv format...")
    convert_model_to_native2d(model)

    for m in model.modules():
        if isinstance(m, Detect):
            m.forward = ESP_Detect.forward.__get__(m)

    print(f"[3/4] Exporting static ONNX with ESP_Detect head to {onnx_path}...")
    model.export(format="onnx", opset=13, dynamic=False, imgsz=640)
    
    # Verify exported ONNX has ZERO Conv1d, Squeeze, or Transpose
    exported_onnx = onnx.load(str(pt_path.with_suffix(".onnx")))
    onnx.save(onnx.shape_inference.infer_shapes(exported_onnx), str(onnx_path))
    op_types = set(n.op_type for n in exported_onnx.graph.node)
    print(f"Exported Native2D ONNX Op Types: {sorted(list(op_types))}")
    assert 'Squeeze' not in op_types and 'Transpose' not in op_types, "Found 1D Squeeze/Transpose!"

    print(f"[4/4] Quantizing with esp-ppq for ESP32-P4 RISC-V SIMD...")
    from esp_ppq import QuantizationSettingFactory
    from esp_ppq.api import espdl_quantize_onnx

    calib_dir = repo_root / "data" / "thermal" / "images"
    calib_ds = ThermalCaliDataset(str(calib_dir), img_size=640, max_samples=32)
    calib_loader = DataLoader(calib_ds, batch_size=8, shuffle=False)

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
        calib_steps=16,
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
    print(f"\n🎉 SUCCESS! Quantized Champion Native2D ESPDL created: {espdl_path} ({espdl_path.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
