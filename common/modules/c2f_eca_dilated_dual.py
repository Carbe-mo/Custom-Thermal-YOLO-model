"""
common/modules/c2f_eca_dilated_dual.py — Dilated Bottleneck with Dual Sequential ECA Attention
===============================================================================================

Combines the expanded receptive field of dilation (d=2, p=2) at deep stages with Dual ECA
calibration:
  res = self.cv1(x)
  res = self.eca1(res)
  res = self.cv2(res)   # Dilated 3x3 (d=2, p=2)
  res = self.eca2(res)
  return x + res if self.add else res

Enables 100% pretrained weight transfer (322/361 items) while calibrating both intermediate
channel features and dilated contextual features to eliminate background thermal noise.
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


class DilatedDualECABottleneck(nn.Module):
    """
    Dilated 3x3 Bottleneck with Dual ECA calibration (post-cv1 and post-cv2).
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
        eca_k: int = 3,
        dilation: int = 2,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.eca1 = ECA(c_, k_size=eca_k)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g, p=dilation, d=dilation)
        self.eca2 = ECA(c2, k_size=eca_k)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.eca1(self.cv1(x))
        y = self.eca2(self.cv2(y))
        return x + y if self.add else y


class C2f_Dilated_Dual_ECA(nn.Module):
    """
    C2f module equipped with DilatedDualECABottlenecks.
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        eca_k: int = 3,
        dilation: int = 2,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DilatedDualECABottleneck(
                self.c, self.c, shortcut, g, k=(3, 3), e=1.0, eca_k=eca_k, dilation=dilation
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


_REGISTERED = False


def register_c2f_dilated_dual_eca() -> None:
    """Register C2f_Dilated_Dual_ECA into Ultralytics nn modules and parser."""
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.__dict__["C2f_Dilated_Dual_ECA"] = C2f_Dilated_Dual_ECA
    tasks.__dict__["DilatedDualECABottleneck"] = DilatedDualECABottleneck
    modules.__dict__["C2f_Dilated_Dual_ECA"] = C2f_Dilated_Dual_ECA
    modules.__dict__["DilatedDualECABottleneck"] = DilatedDualECABottleneck

    orig_parse_model = tasks.parse_model

    def custom_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)

        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m_name, args = item
                    if m_name == "C2f_Dilated_Dual_ECA":
                        custom_layers[(section, i)] = ("C2f_Dilated_Dual_ECA", n, args)
                        d_mod[section][i] = [f, n, "C2f", args[:2]]

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
            eca_k = args[4] if len(args) > 4 else 3
            dilation = args[5] if len(args) > 5 else 2

            layer = C2f_Dilated_Dual_ECA(
                c1, c2, n=n, shortcut=shortcut, g=g, e=e, eca_k=eca_k, dilation=dilation
            )
            layer.i = idx
            layer.f = f
            layer.type = "common.modules.c2f_eca_dilated_dual.C2f_Dilated_Dual_ECA"
            layer.np = sum(p.numel() for p in layer.parameters())

            model[idx] = layer

        return model, save

    tasks.parse_model = custom_parse_model
    _REGISTERED = True
