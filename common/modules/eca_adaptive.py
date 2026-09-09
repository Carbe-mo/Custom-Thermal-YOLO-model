"""
common/modules/eca_adaptive.py — Adaptive Kernel ECA Bottleneck & C2f_ECA_Adaptive
=================================================================================

Implements ECA with channel-dependent adaptive kernel size k(C):
  k = |log2(C)/gamma + b/gamma|_odd  (gamma=2, b=1)
  - C = 64  -> k = 3
  - C = 128 -> k = 3
  - C = 256 -> k = 5
  - C = 512 -> k = 5

Allows wider cross-channel interaction for deeper semantic thermal features in P4/P5.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.conv import Conv


class ECAAdaptive(nn.Module):
    """
    Efficient Channel Attention with Adaptive 1D Convolution Kernel Size.
    """
    def __init__(self, channels: int, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        # Calculate adaptive kernel size
        t = int(abs((math.log2(channels) + b) / gamma))
        k_size = t if t % 2 == 1 else t + 1
        k_size = max(3, k_size)

        self.k_size = k_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)
        # [B, C, 1, 1] -> [B, 1, C]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        # [B, 1, C] -> [B, C, 1, 1]
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


class ECAAdaptiveBottleneck(Bottleneck):
    """
    Bottleneck subclass incorporating Adaptive ECA in residual path:
      res = self.eca(self.cv2(self.cv1(x)))
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
    ):
        super().__init__(c1, c2, shortcut=shortcut, g=g, k=k, e=e)
        c_ = int(c2 * e)
        self.eca = ECAAdaptive(channels=c_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.eca(self.cv2(self.cv1(x)))
        return x + res if self.add else res


class C2f_ECA_Adaptive(nn.Module):
    """
    CSP Bottleneck with 2 convolutions and Adaptive ECA inside each bottleneck.
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
            ECAAdaptiveBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


_REGISTERED = False

def register_c2f_eca_adaptive() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.__dict__["C2f_ECA_Adaptive"] = C2f_ECA_Adaptive
    tasks.__dict__["ECAAdaptiveBottleneck"] = ECAAdaptiveBottleneck
    tasks.__dict__["ECAAdaptive"] = ECAAdaptive
    modules.__dict__["C2f_ECA_Adaptive"] = C2f_ECA_Adaptive
    modules.__dict__["ECAAdaptiveBottleneck"] = ECAAdaptiveBottleneck
    modules.__dict__["ECAAdaptive"] = ECAAdaptive

    orig_parse_model = tasks.parse_model

    def custom_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)

        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m_name, args = item
                    if m_name == "C2f_ECA_Adaptive":
                        custom_layers[(section, i)] = ("C2f_ECA_Adaptive", n, args)
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

            layer = C2f_ECA_Adaptive(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
            layer.i = idx
            layer.f = f
            layer.type = "common.modules.eca_adaptive.C2f_ECA_Adaptive"
            layer.np = sum(p.numel() for p in layer.parameters())

            model[idx] = layer

        return model, save

    tasks.parse_model = custom_parse_model
    _REGISTERED = True


if __name__ == "__main__":
    register_c2f_eca_adaptive()
    c2f = C2f_ECA_Adaptive(c1=64, c2=64, n=2, shortcut=True)
    x = torch.randn(2, 64, 40, 40)
    out = c2f(x)
    print(f"C2f_ECA_Adaptive output: {out.shape}")
    assert out.shape == (2, 64, 40, 40)
    print("C2f_ECA_Adaptive verified successfully!")
