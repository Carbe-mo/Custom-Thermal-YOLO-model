"""
common/modules/sppf_eca.py — SPPF with Efficient Channel Attention (SPPF_ECA)
==============================================================================

Applies Spatial Pyramid Pooling - Fast followed by ECA (k=3) channel calibration:
  y = [self.cv1(x)]
  y.extend(self.m(y[-1]) for _ in range(self.n))
  y = self.cv2(torch.cat(y, 1))
  y = self.eca(y)
  return y + x if self.add else y

Maintains 100% pretrained weight transfer compatibility with stock SPPF from yolov8n.pt.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.nn.modules.conv import Conv
from common.modules.eca import ECA


class SPPF_ECA(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer with ECA channel attention calibration.
    """
    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False, eca_k: int = 3):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.eca = ECA(c2, k_size=eca_k)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        y = self.eca(y)
        return y + x if self.add else y


_REGISTERED = False


def register_sppf_eca() -> None:
    """Register SPPF_ECA into Ultralytics nn modules and parser."""
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.__dict__["SPPF_ECA"] = SPPF_ECA
    modules.__dict__["SPPF_ECA"] = SPPF_ECA

    orig_parse_model = tasks.parse_model

    def custom_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)

        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m_name, args = item
                    if m_name == "SPPF_ECA":
                        custom_layers[(section, i)] = ("SPPF_ECA", n, args)
                        # Substitute with standard SPPF for initial sizing
                        d_mod[section][i] = [f, n, "SPPF", args[:2]]

        model, save = orig_parse_model(d_mod, ch, verbose=verbose)

        bb_len = len(d_mod.get("backbone", []))
        for (section, i), (m_type, n_repeats, args) in custom_layers.items():
            idx = i if section == "backbone" else bb_len + i
            f = d[section][i][0]

            c1 = model[idx].cv1.conv.in_channels
            c2 = model[idx].cv2.conv.out_channels
            k = args[1] if len(args) > 1 else 5
            eca_k = args[2] if len(args) > 2 else 3

            layer = SPPF_ECA(c1, c2, k=k, eca_k=eca_k)
            layer.i = idx
            layer.f = f
            layer.type = "common.modules.sppf_eca.SPPF_ECA"
            layer.np = sum(p.numel() for p in layer.parameters())

            model[idx] = layer

        return model, save

    tasks.parse_model = custom_parse_model
    _REGISTERED = True
