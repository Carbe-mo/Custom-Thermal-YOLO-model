"""
common/modules/afb_daorb.py — Re-implementations of two competing thermal YOLO blocks
=====================================================================================

1) AFB  — Attention Fusion Block (ECA + Coordinate Attention)
   Lade, Bethi, Cenkeramaddi, "Thermal Imaging-based Perception for Automotive Edge AI
   Applications using a Lightweight Custom YOLO Model", IEEE Sensors J., 2026,
   DOI 10.1109/JSEN.2026.3691670 (IEEE Xplore 11520574).

2) DAORB — Dual Attention Optimized Residual Block (ECA + Spatial Attention)
   "TMCR Edge-YOLO: Thermal Sensing-based Mobility-Context Recognition on the Edge",
   IEEE Sensors J., 2026, DOI 10.1109/JSEN.2026.3686674 (IEEE Xplore 11498574).
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn


class _ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., CVPR'20)."""

    def __init__(self, c: int, k: int | None = 3, gamma: int = 2, b: int = 1):
        super().__init__()
        if k is None:
            t = int(abs((math.log2(max(c, 2)) + b) / gamma))
            k = t if t % 2 else t + 1
            k = max(3, k)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)                                   # [B,C,1,1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))     # [B,1,C]
        y = self.act(y.transpose(-1, -2).unsqueeze(-1))    # [B,C,1,1]
        return x * y


class _CoordAtt(nn.Module):
    """Coordinate Attention (Hou et al., CVPR'21) with SiLU, as in Fig. 1(e)."""

    def __init__(self, c: int, reduction: int = 32):
        super().__init__()
        mip = max(8, c // reduction)
        self.conv1 = nn.Conv2d(c, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, c, 1)
        self.conv_w = nn.Conv2d(mip, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)                      # [B,C,H,1]
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # [B,C,W,1]
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))                  # [B,C,H,1]
        a_w = torch.sigmoid(self.conv_w(y_w.permute(0, 1, 3, 2)))  # [B,C,1,W]
        return x * a_h * a_w


class _SpatialAtt(nn.Module):
    """CBAM-style Spatial Attention: sigma(Conv7x7([avg_c; max_c]))."""

    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class _InvertedCore(nn.Module):
    """Shared expansion -> DW5x5 -> projection core (Alg. 1, lines 1-3 of both papers)."""

    def __init__(self, c: int, r: int = 2, dw_k: int = 5):
        super().__init__()
        ce = c * r
        self.expand = nn.Conv2d(c, ce, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ce)
        self.dw = nn.Conv2d(ce, ce, dw_k, padding=dw_k // 2, groups=ce, bias=False)
        self.bn2 = nn.BatchNorm2d(ce)
        self.project = nn.Conv2d(ce, c, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.bn1(self.act(self.expand(x)))   # BN(SiLU(Conv1x1))
        u = self.bn2(self.act(self.dw(u)))       # BN(SiLU(DWConv5x5))
        return self.bn3(self.project(u))         # BN(Conv1x1)


class AFB(nn.Module):
    """Attention Fusion Block (ECA k=3 + Coordinate Attention), residual. C -> C."""

    def __init__(self, c1: int, c2: int | None = None, n: int = 1, r: int = 2):
        super().__init__()
        c = c1
        self.blocks = nn.Sequential(*(self._make_one(c, r) for _ in range(max(1, n))))

    @staticmethod
    def _make_one(c: int, r: int) -> nn.Module:
        return _SingleAFB(c, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _SingleAFB(nn.Module):
    def __init__(self, c: int, r: int = 2):
        super().__init__()
        self.core = _InvertedCore(c, r)
        self.eca = _ECA(c, k=3)
        self.ca = _CoordAtt(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ca(self.eca(self.core(x)))


class DAORB(nn.Module):
    """Dual Attention Optimized Residual Block (adaptive ECA + Spatial Attention). C -> C."""

    def __init__(self, c1: int, c2: int | None = None, n: int = 1, r: int = 2):
        super().__init__()
        c = c1
        self.blocks = nn.Sequential(*(self._make_one(c, r) for _ in range(max(1, n))))

    @staticmethod
    def _make_one(c: int, r: int) -> nn.Module:
        return _SingleDAORB(c, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _SingleDAORB(nn.Module):
    def __init__(self, c: int, r: int = 2):
        super().__init__()
        self.core = _InvertedCore(c, r)
        self.eca = _ECA(c, k=None)
        self.sa = _SpatialAtt(7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sa(self.eca(self.core(x)))


_REGISTERED = False


def register_afb_daorb() -> None:
    """Expose AFB / DAORB to Ultralytics YAML parsing using C2f placeholder replacement."""
    global _REGISTERED
    if _REGISTERED:
        return
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    for name, cls in (
        ("AFB", AFB),
        ("_SingleAFB", _SingleAFB),
        ("DAORB", DAORB),
        ("_SingleDAORB", _SingleDAORB),
    ):
        tasks.__dict__[name] = cls
        modules.__dict__[name] = cls

    orig_parse_model = tasks.parse_model

    def custom_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)

        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m, args = item
                    if m in ("AFB", "DAORB"):
                        custom_layers[(section, i)] = (m, n, args)
                        d_mod[section][i] = [f, n, "C2f", [args[0], True]]

        model, save = orig_parse_model(d_mod, ch, verbose=verbose)

        bb_len = len(d_mod.get("backbone", []))
        for (section, i), (m_type, n_repeats, args) in custom_layers.items():
            idx = i if section == "backbone" else bb_len + i
            f = d[section][i][0]
            c1 = model[idx].cv1.conv.in_channels
            c2 = model[idx].cv2.conv.out_channels
            n = len(model[idx].m)

            cls = AFB if m_type == "AFB" else DAORB
            new_layer = cls(c1, c2, n=n)
            new_layer.i = idx
            new_layer.f = f
            new_layer.type = f"common.modules.afb_daorb.{m_type}"
            new_layer.np = sum(p.numel() for p in new_layer.parameters())
            model[idx] = new_layer

        return model, save

    tasks.parse_model = custom_parse_model
    _REGISTERED = True
