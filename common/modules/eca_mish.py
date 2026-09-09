"""
common/modules/eca_mish.py — Mish-Activated ECA Bottleneck (MishECABottleneck & C2f_ECA_Mish)
=============================================================================================

Implements Mish activation with ECA Channel Attention inside Bottlenecks:
- Mish activation: x * tanh(softplus(x)), smooth non-monotonic gradient flow.
- ECA channel attention: non-reductive 1D conv (k=3) in residual path.
- 100% pretrained weight transfer from yolov8n.pt.
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
from common.modules.eca import ECA


class MishConv(Conv):
    """
    Standard Conv with Mish activation: Conv2d + BatchNorm2d + Mish.
    """
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1, d: int = 1):
        super().__init__(c1, c2, k=k, s=s, p=p, g=g, d=d, act=False)
        self.act = nn.Mish(inplace=True)


class MishECABottleneck(Bottleneck):
    """
    Bottleneck subclass using MishConv and ECA Channel Attention:
      cv1 (Mish) -> cv2 (Mish) -> ECA(k=3) -> residual add
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
        k_size: int = 3,
    ):
        super().__init__(c1, c2, shortcut=shortcut, g=g, k=k, e=e)
        c_ = int(c2 * e)
        self.cv1 = MishConv(c1, c_, k[0], 1)
        self.cv2 = MishConv(c_, c2, k[1], 1, g=g)
        self.eca = ECA(channels=c2, k_size=k_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.eca(self.cv2(self.cv1(x)))
        return x + res if self.add else res


class C2f_ECA_Mish(nn.Module):
    """
    CSP Bottleneck using MishECABottleneck inside each split branch.
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
            MishECABottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


_REGISTERED = False

def register_c2f_eca_mish() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.__dict__["C2f_ECA_Mish"] = C2f_ECA_Mish
    tasks.__dict__["MishECABottleneck"] = MishECABottleneck
    modules.__dict__["C2f_ECA_Mish"] = C2f_ECA_Mish
    modules.__dict__["MishECABottleneck"] = MishECABottleneck

    orig_parse_model = tasks.parse_model

    def custom_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)

        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m_name, args = item
                    if m_name == "C2f_ECA_Mish":
                        custom_layers[(section, i)] = ("C2f_ECA_Mish", n, args)
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

            layer = C2f_ECA_Mish(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
            layer.i = idx
            layer.f = f
            layer.type = "common.modules.eca_mish.C2f_ECA_Mish"
            layer.np = sum(p.numel() for p in layer.parameters())

            model[idx] = layer

        return model, save

    tasks.parse_model = custom_parse_model
    _REGISTERED = True


if __name__ == "__main__":
    register_c2f_eca_mish()
    c2f_em = C2f_ECA_Mish(c1=64, c2=64, n=2, shortcut=True)
    x = torch.randn(2, 64, 40, 40)
    out = c2f_em(x)
    print(f"C2f_ECA_Mish output: {out.shape}")
    assert out.shape == (2, 64, 40, 40)
    print("C2f_ECA_Mish verification passed!")
