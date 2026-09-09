"""
common/modules/c2f_eca_inception.py — Multi-Scale Inception Strip Convolutions with ECA
========================================================================================

Implements `InceptionECABottleneck` and `C2f_ECA_Inception`.
In thermal imagery, objects have distinct aspect ratios (tall pedestrians vs wide luggage/carts).
This module replaces standard cv2 with 3 multi-scale branches:
  1. Standard 3x3 Conv (local features)
  2. Asymmetric strip convolutions (1x5 followed by 5x1) for anisotropic thermal silhouettes
  3. 1x1 Conv (high-frequency thermal edge preservation)
All branches are fused and calibrated via non-reductive ECA (k=3) channel attention.
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


class InceptionECABottleneck(nn.Module):
    """
    Multi-Scale Inception Bottleneck with Asymmetric Strip Convolutions and ECA Attention.
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
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        
        # Branch 1: Standard 3x3 Conv
        self.branch_3x3 = Conv(c_, c_, 3, 1, g=g)
        
        # Branch 2: Asymmetric 1x5 + 5x1 Strip Convolutions (tall pedestrians & wide luggage)
        self.strip_v = Conv(c_, c_, (5, 1), 1, p=(2, 0), g=g)
        self.strip_h = Conv(c_, c_, (1, 5), 1, p=(0, 2), g=g)
        
        # Branch 3: 1x1 Projection
        self.branch_1x1 = Conv(c_, c_, 1, 1)
        
        # Channel fusion projection back to c2
        self.fuse = Conv(c_, c2, 1, 1)
        
        # ECA channel attention
        self.eca = ECA(c2, k_size=eca_k)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cv1(x)
        
        # Multi-scale anisotropic feature extraction
        out1 = self.branch_3x3(h)
        out2 = self.strip_h(self.strip_v(h))
        out3 = self.branch_1x1(h)
        
        # Fuse branches
        fused = self.fuse(out1 + out2 + out3)
        
        # Apply ECA channel calibration
        calibrated = self.eca(fused)
        
        return x + calibrated if self.add else calibrated


class C2f_ECA_Inception(nn.Module):
    """
    C2f module equipped with InceptionECABottlenecks.
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
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            InceptionECABottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, eca_k=eca_k)
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

def register_c2f_eca_inception() -> None:
    """
    Register C2f_ECA_Inception into Ultralytics nn modules and parser.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.__dict__["C2f_ECA_Inception"] = C2f_ECA_Inception
    tasks.__dict__["InceptionECABottleneck"] = InceptionECABottleneck
    modules.__dict__["C2f_ECA_Inception"] = C2f_ECA_Inception
    modules.__dict__["InceptionECABottleneck"] = InceptionECABottleneck

    orig_parse_model = tasks.parse_model

    def custom_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)

        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m_name, args = item
                    if m_name == "C2f_ECA_Inception":
                        custom_layers[(section, i)] = ("C2f_ECA_Inception", n, args)
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
            eca_k = args[4] if len(args) > 4 else 3

            layer = C2f_ECA_Inception(c1, c2, n=n, shortcut=shortcut, g=g, e=e, eca_k=eca_k)
            layer.i = idx
            layer.f = f
            layer.type = "common.modules.c2f_eca_inception.C2f_ECA_Inception"
            layer.np = sum(p.numel() for p in layer.parameters())

            model[idx] = layer

        return model, save

    tasks.parse_model = custom_parse_model
    _REGISTERED = True


if __name__ == "__main__":
    register_c2f_eca_inception()
    m = C2f_ECA_Inception(64, 64, n=2, shortcut=True)
    x = torch.randn(2, 64, 40, 40)
    out = m(x)
    print("Inception C2f output shape:", out.shape)
