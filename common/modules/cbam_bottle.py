"""
common/modules/cbam_bottle.py — CBAM inside Bottleneck (CBAMBottleneck & C2f_CBAM_Bottle)
=======================================================================================

Implements CBAM (Channel Attention + Spatial Attention) placed inside the
residual branch of each Bottleneck, right after the second convolution.
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
from common.modules.cbam import CBAM


class CBAMBottleneck(Bottleneck):
    """
    Bottleneck subclass incorporating full CBAM (Channel + Spatial Attention)
    inside the residual path before the shortcut addition:
      res = self.cbam(self.cv2(self.cv1(x)))
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
        self.cbam = CBAM(channels=c_, reduction=reduction, kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.cbam(self.cv2(self.cv1(x)))
        return x + res if self.add else res


class C2f_CBAM_Bottle(nn.Module):
    """
    CSP Bottleneck using CBAMBottleneck inside each split branch.
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
            CBAMBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


_REGISTERED = False

def register_cbam_bottle() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.__dict__["C2f_CBAM_Bottle"] = C2f_CBAM_Bottle
    tasks.__dict__["CBAMBottleneck"] = CBAMBottleneck
    modules.__dict__["C2f_CBAM_Bottle"] = C2f_CBAM_Bottle
    modules.__dict__["CBAMBottleneck"] = CBAMBottleneck

    orig_parse_model = tasks.parse_model

    def custom_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)

        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m_name, args = item
                    if m_name == "C2f_CBAM_Bottle":
                        custom_layers[(section, i)] = ("C2f_CBAM_Bottle", n, args)
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

            cbam_layer = C2f_CBAM_Bottle(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
            cbam_layer.i = idx
            cbam_layer.f = f
            cbam_layer.type = "common.modules.cbam_bottle.C2f_CBAM_Bottle"
            cbam_layer.np = sum(p.numel() for p in cbam_layer.parameters())

            model[idx] = cbam_layer

        return model, save

    tasks.parse_model = custom_parse_model
    _REGISTERED = True


if __name__ == "__main__":
    register_cbam_bottle()
    c2f_cbam = C2f_CBAM_Bottle(c1=64, c2=64, n=2, shortcut=True)
    x = torch.randn(2, 64, 40, 40)
    out = c2f_cbam(x)
    print(f"C2f_CBAM_Bottle output: {out.shape}")
    assert out.shape == (2, 64, 40, 40)
    print("C2f_CBAM_Bottle verification passed!")
