"""
common/modules/coord_att.py — Coordinate Attention (CoordAtt) & C2f_CoordAtt
===========================================================================

Implements Coordinate Attention (Hou et al., CVPR 2021) inside Bottlenecks:
- Factorizes 2D attention into 1D horizontal and 1D vertical spatial pooling.
- Captures position-sensitive location information and channel interactions simultaneously.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.conv import Conv


class CoordAtt(nn.Module):
    """
    Coordinate Attention block:
    Captures cross-channel and direction-aware spatial dependencies.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mip = max(8, in_channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = self.pool_h(x)                       # [b, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)   # [b, c, 1, w] -> [b, c, w, 1]

        y = torch.cat([x_h, x_w], dim=2)           # [b, c, h+w, 1]
        y = self.act(self.bn1(self.conv1(y)))      # [b, mip, h+w, 1]

        x_h_out, x_w_out = torch.split(y, [h, w], dim=2)
        x_w_out = x_w_out.permute(0, 1, 3, 2)      # [b, mip, 1, w]

        a_h = self.sigmoid(self.conv_h(x_h_out))   # [b, c, h, 1]
        a_w = self.sigmoid(self.conv_w(x_w_out))   # [b, c, 1, w]

        return x * a_h * a_w


class CoordAttBottleneck(Bottleneck):
    """
    Bottleneck subclass incorporating Coordinate Attention in the residual path:
      res = self.ca(self.cv2(self.cv1(x)))
      return x + res if self.add else res
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
        reduction: int = 16,
    ):
        super().__init__(c1, c2, shortcut=shortcut, g=g, k=k, e=e)
        c_ = int(c2 * e)
        self.ca = CoordAtt(in_channels=c2, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.ca(self.cv2(self.cv1(x)))
        return x + res if self.add else res


class C2f_CoordAtt(nn.Module):
    """
    CSP Bottleneck with 2 convolutions and CoordAtt inside each bottleneck.
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            CoordAttBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


_REGISTERED = False

def register_c2f_coordatt() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.__dict__["C2f_CoordAtt"] = C2f_CoordAtt
    tasks.__dict__["CoordAttBottleneck"] = CoordAttBottleneck
    tasks.__dict__["CoordAtt"] = CoordAtt
    modules.__dict__["C2f_CoordAtt"] = C2f_CoordAtt
    modules.__dict__["CoordAttBottleneck"] = CoordAttBottleneck
    modules.__dict__["CoordAtt"] = CoordAtt

    orig_parse_model = tasks.parse_model

    def custom_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)

        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m_name, args = item
                    if m_name == "C2f_CoordAtt":
                        custom_layers[(section, i)] = ("C2f_CoordAtt", n, args)
                        d_mod[section][i] = [f, n, "C2f", args]

        model, save = orig_parse_model(d_mod, ch, verbose=verbose)

        bb_len = len(d_mod.get("backbone", []))
        for (section, i), (m_type, n_repeats, args) in custom_layers.items():
            idx = i if section == "backbone" else bb_len + i
            f = d[section][i][0]

            c1 = model[idx].cv1.conv.in_channels
            c2 = model[idx].cv2.conv.out_channels
            n = len(model[idx].m)
            shortcut = args[1] if len(args) > 1 else False
            g = args[2] if len(args) > 2 else 1
            e = args[3] if len(args) > 3 else 0.5

            layer = C2f_CoordAtt(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
            layer.i = idx
            layer.f = f
            layer.type = "common.modules.coord_att.C2f_CoordAtt"
            layer.np = sum(p.numel() for p in layer.parameters())

            model[idx] = layer

        return model, save

    tasks.parse_model = custom_parse_model
    _REGISTERED = True


if __name__ == "__main__":
    register_c2f_coordatt()
    c2f_ca = C2f_CoordAtt(c1=64, c2=64, n=2, shortcut=True)
    x = torch.randn(2, 64, 40, 40)
    out = c2f_ca(x)
    print(f"C2f_CoordAtt output: {out.shape}")
    assert out.shape == (2, 64, 40, 40)
    print("C2f_CoordAtt verification passed!")
