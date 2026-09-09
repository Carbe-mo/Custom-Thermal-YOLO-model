"""
common/modules/c2f_eca_graduated.py — Graduated Stage-Selective Dilated ECA Bottleneck
========================================================================================

Implements `GraduatedDilatedECABottleneck` and `C2f_ECA_Graduated`.

The dilation rate is passed as a constructor argument (default d=1 = standard 3×3).
This allows the model.yaml to specify different dilation rates per stage:
  - P2/P3: dilation=1  (standard 3×3, fine edge preservation)
  - P4:    dilation=2  (5×5 effective RF, medium context)
  - P5:    dilation=3  (7×7 effective RF, large semantic context)

All weight shapes (3×3 kernel, same padding formula) remain identical to the
stock Bottleneck, preserving yolov8n.pt transfer for all stages.
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


class GraduatedDilatedECABottleneck(nn.Module):
    """
    Bottleneck with configurable dilation rate + ECA attention.
    dilation=1  → standard 3×3 (same as ECABottleneck)
    dilation=2  → 5×5 effective RF (d=2, p=2)
    dilation=3  → 7×7 effective RF (d=3, p=3)
    All configurations keep identical tensor shapes for pretrained weight transfer.
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
        dilation: int = 1,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        # dilation=1 → p=1 (standard), dilation=d → p=d (same-size output)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g, p=dilation, d=dilation)
        self.eca = ECA(c2, k_size=eca_k)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        y = self.eca(y)
        return x + y if self.add else y


class C2f_ECA_Graduated(nn.Module):
    """
    C2f module with GraduatedDilatedECABottlenecks.
    Pass dilation= as a yaml arg to set per-stage dilation rate.
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
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            GraduatedDilatedECABottleneck(
                self.c, self.c, shortcut, g, k=(3, 3), e=1.0, eca_k=eca_k, dilation=dilation
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


_REGISTERED = False


def register_c2f_eca_graduated() -> None:
    """Register C2f_ECA_Graduated into Ultralytics nn modules and parser."""
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.__dict__["C2f_ECA_Graduated"] = C2f_ECA_Graduated
    tasks.__dict__["GraduatedDilatedECABottleneck"] = GraduatedDilatedECABottleneck
    modules.__dict__["C2f_ECA_Graduated"] = C2f_ECA_Graduated
    modules.__dict__["GraduatedDilatedECABottleneck"] = GraduatedDilatedECABottleneck

    orig_parse_model = tasks.parse_model

    def custom_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)

        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m_name, args = item
                    if m_name == "C2f_ECA_Graduated":
                        custom_layers[(section, i)] = ("C2f_ECA_Graduated", n, args)
                        d_mod[section][i] = [f, n, "C2f", args]

        model, save = orig_parse_model(d_mod, ch, verbose=verbose)

        bb_len = len(d_mod.get("backbone", []))
        for (section, i), (m_type, n_repeats, args) in custom_layers.items():
            idx = i if section == "backbone" else bb_len + i
            f = d[section][i][0]

            c1 = model[idx].cv1.conv.in_channels
            c2 = model[idx].cv2.conv.out_channels
            n  = len(model[idx].m)
            shortcut = args[1] if len(args) > 1 else False
            g        = args[2] if len(args) > 2 else 1
            e        = args[3] if len(args) > 3 else 0.5
            eca_k    = args[4] if len(args) > 4 else 3
            dilation = args[5] if len(args) > 5 else 1  # default standard

            layer = C2f_ECA_Graduated(
                c1, c2, n=n, shortcut=shortcut, g=g, e=e, eca_k=eca_k, dilation=dilation
            )
            layer.i  = idx
            layer.f  = f
            layer.type = "common.modules.c2f_eca_graduated.C2f_ECA_Graduated"
            layer.np = sum(p.numel() for p in layer.parameters())

            model[idx] = layer

        return model, save

    tasks.parse_model = custom_parse_model
    _REGISTERED = True


if __name__ == "__main__":
    register_c2f_eca_graduated()
    for d in [1, 2, 3]:
        m = C2f_ECA_Graduated(64, 64, n=2, shortcut=True, dilation=d)
        x = torch.randn(2, 64, 40, 40)
        out = m(x)
        print(f"GraduatedDilated d={d} output shape: {out.shape}")
