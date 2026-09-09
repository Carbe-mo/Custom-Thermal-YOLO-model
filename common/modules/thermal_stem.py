"""
common/modules/thermal_stem.py — Thermal-Aware Input Stem
==========================================================

Architectural Rationale & Parameter Choices:
--------------------------------------------
1. Thermal IR Characteristics vs. RGB:
   - Thermal images capture heat radiation (long-wave infrared), which exhibits
     continuous thermal diffusion, lower high-frequency texture, and subtle
     contrast gradients between foreground objects (e.g. warm person / cold bag)
     and the surrounding background.
   - Standard YOLOv8 uses a single aggressive `Conv(3, 16, k=3, s=2)` as layer 0.
     Downsampling by 2 in a single step with a small 3x3 kernel discards ~75% of
     subtle thermal boundary pixels before any feature representation is formed.

2. Why Two-Stage Contrast-Preserving Architecture?
   - Stage 1 (Full Resolution Contrast Extraction, Stride=1):
     Uses a 3x3 convolution at stride=1 (`Conv(c1, c_mid, k=3, s=1)`) to perform
     local contrast enhancement and filter microbolometer sensor noise at full
     640x640 spatial resolution.
   - Stage 2 (Soft Gradient Downsampling, Stride=2):
     Follows with a second 3x3 convolution at stride=2 (`Conv(c_mid, c2, k=k, s=2)`)
     and a residual 1x1 stride=2 projection to gently downsample to 320x320
     while preserving thermal contours and object boundaries.

3. Channel & Stride Specifications:
   - Input Channels (c1): 3 (replicated thermal channel)
   - Intermediate Channels (c_mid): max(8, c2 // 2) = 8 for YOLOv8n (width=0.25)
   - Output Channels (c2): 16 for YOLOv8n (scaled by width_multiple)
   - Effective Receptive Field: 5x5 at stride 2 (vs 3x3 in stock YOLOv8), capturing
     larger context of thermal glow and object temperature silhouettes.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class ThermalStem(nn.Module):
    """
    Thermal-Aware Input Stem Module for YOLOv8.
    
    Replaces layer 0 `Conv(c1, c2, 3, 2)` with a 2-stage contrast-preserving stem:
      Stage 1: Conv(c1, c_mid, k=3, s=1, p=1) -> full-res thermal gradient extraction
      Stage 2: Conv(c_mid, c2, k=3, s=2, p=1) -> downsampling with thermal boundary preservation
      Residual: Conv(c1, c2, k=1, s=2, p=0)    -> direct skip connection
      Gate:     Squeeze-and-Excitation         -> channel contrast recalibration
    """
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2):
        super().__init__()
        assert s == 2, f"ThermalStem expected stride 2, got {s}"
        
        c_mid = max(8, c2 // 2)
        
        # Stage 1: Full-resolution feature extraction & local contrast filtering
        self.conv1 = Conv(c1, c_mid, k=3, s=1)
        
        # Stage 2: Downsampling convolution preserving thermal gradient context
        self.conv2 = Conv(c_mid, c2, k=k, s=2)
        
        # Shortcut branch: 1x1 stride-2 projection for residual thermal intensity preservation
        self.shortcut = Conv(c1, c2, k=1, s=2) if (c1 != c2 or s != 1) else nn.Identity()
        
        # Channel-wise contrast calibration gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, max(4, c2 // 4), kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(4, c2 // 4), c2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        out = out * self.gate(out)
        return out + self.shortcut(x)


_REGISTERED = False

def register_thermal_stem() -> None:
    """
    Register ThermalStem into Ultralytics nn modules and tasks
    so that model.yaml can reference 'ThermalStem' directly.
    """
    global _REGISTERED
    if _REGISTERED:
        return
        
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules
    
    tasks.__dict__["ThermalStem"] = ThermalStem
    modules.__dict__["ThermalStem"] = ThermalStem
    
    orig_parse_model = tasks.parse_model
    
    def custom_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)
        
        # Intercept ThermalStem in backbone
        if "backbone" in d_mod:
            for i, item in enumerate(d_mod["backbone"]):
                f, n, m_name, args = item
                if m_name == "ThermalStem":
                    custom_layers[i] = ("ThermalStem", args)
                    # Temporarily use stock Conv so channel arithmetic runs correctly
                    d_mod["backbone"][i] = [f, n, "Conv", args]
                    
        model, save = orig_parse_model(d_mod, ch, verbose=verbose)
        
        # Swap placeholder Conv with actual ThermalStem instance
        for idx, (m_type, args) in custom_layers.items():
            c1 = model[idx].conv.in_channels
            c2 = model[idx].conv.out_channels
            k = args[1] if len(args) > 1 else 3
            s = args[2] if len(args) > 2 else 2
            
            stem_layer = ThermalStem(c1, c2, k=k, s=s)
            stem_layer.i = idx
            stem_layer.f = -1
            stem_layer.type = "common.modules.thermal_stem.ThermalStem"
            stem_layer.np = sum(p.numel() for p in stem_layer.parameters())
            
            model[idx] = stem_layer
            
        return model, save

    tasks.parse_model = custom_parse_model
    _REGISTERED = True


if __name__ == "__main__":
    register_thermal_stem()
    stem = ThermalStem(c1=3, c2=16, k=3, s=2)
    x = torch.randn(2, 3, 640, 640)
    out = stem(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 16, 320, 320)
    print("ThermalStem unit test passed successfully!")
