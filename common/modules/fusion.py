"""
common/modules/fusion.py — RGB-Thermal Fusion Modules & Registration
=====================================================================

Implements fusion modules for dual-stream multi-modal detection:
- SliceChannels: Extracts channel slices (e.g. RGB 0:3, Thermal 3:6)
- AddFusion:     Element-wise feature addition
- GatedFusion:   Dynamic attention gating: g*F_rgb + (1-g)*F_thermal
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.nn.modules.conv import Conv, Index


class SliceChannels(nn.Module):
    """
    Extracts a slice of channels from the input tensor [B, C, H, W].
    """
    def __init__(self, start_ch: int, end_ch: int):
        super().__init__()
        self.start_ch = start_ch
        self.end_ch = end_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.start_ch:self.end_ch, :, :]


class AddFusion(nn.Module):
    """
    Element-wise addition of two feature maps.
    """
    def __init__(self, c: int = 0):
        super().__init__()

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return x[0] + x[1]


class GatedFusion(nn.Module):
    """
    Dynamic Gated Fusion (MBNet / GMFNet style).
    Learns spatial-channel gating map g in [0, 1] to dynamically balance modalities:
        F_fused = g * F_rgb + (1 - g) * F_thermal
    """
    def __init__(self, c: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.Sigmoid(),
        )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        f_rgb, f_th = x[0], x[1]
        cat = torch.cat([f_rgb, f_th], dim=1)
        g = self.gate_conv(cat)
        return g * f_rgb + (1.0 - g) * f_th


_REGISTERED = False


def register_fusion_modules() -> None:
    """
    Register SliceChannels, AddFusion, and GatedFusion with Ultralytics parser.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    # Register in tasks and modules dictionaries
    for cls in [SliceChannels, AddFusion, GatedFusion]:
        tasks.__dict__[cls.__name__] = cls
        modules.__dict__[cls.__name__] = cls

    orig_parse = tasks.parse_model

    def custom_parse(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)

        extra_save = []
        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m_name, args = item
                    if m_name == "SliceChannels":
                        custom_layers[(section, i)] = ("SliceChannels", f, n, args)
                        out_c = args[1] - args[0]
                        # Use Index placeholder to set exact output channels without width scaling
                        d_mod[section][i] = [f, 1, "Index", [out_c]]
                    elif m_name in ("GatedFusion", "AddFusion"):
                        custom_layers[(section, i)] = (m_name, f, n, args)
                        # Replace temporarily with Identity from first branch
                        f1, f2 = f[0], f[1]
                        extra_save.append(f2)
                        d_mod[section][i] = [f1, 1, "nn.Identity", []]

        model, save = orig_parse(d_mod, ch, verbose=verbose)

        # Merge extra save indices
        for s_idx in extra_save:
            if s_idx not in save:
                save.append(s_idx)
        save.sort()

        # Replace placeholders with actual module instances
        bb_len = len(d_mod.get("backbone", []))
        for (section, i), (m_name, f, n, args) in custom_layers.items():
            idx = i if section == "backbone" else bb_len + i

            if m_name == "SliceChannels":
                layer = SliceChannels(args[0], args[1])
                layer.i = idx
                layer.f = f
                layer.type = "common.modules.fusion.SliceChannels"
                layer.np = 0
                model[idx] = layer

            elif m_name == "GatedFusion":
                # Determine channel width of the branch from previous layer
                c = model[f[0]].conv.out_channels if hasattr(model[f[0]], "conv") else (
                    model[f[0]].cv2.conv.out_channels if hasattr(model[f[0]], "cv2") else args[0]
                )
                layer = GatedFusion(c)
                layer.i = idx
                layer.f = f
                layer.type = "common.modules.fusion.GatedFusion"
                layer.np = sum(p.numel() for p in layer.parameters())
                model[idx] = layer

            elif m_name == "AddFusion":
                c = args[0]
                layer = AddFusion(c)
                layer.i = idx
                layer.f = f
                layer.type = "common.modules.fusion.AddFusion"
                layer.np = 0
                model[idx] = layer

        return model, save

    tasks.parse_model = custom_parse
    _REGISTERED = True


if __name__ == "__main__":
    register_fusion_modules()
    print("Fusion modules registered successfully.")
