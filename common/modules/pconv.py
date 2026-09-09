"""
common/modules/pconv.py — Partial Convolution (PConv) & PConvBottleneck
======================================================================

Implements FasterNet Partial Convolution (PConv) and PConvBottleneck:
- Only convolves the first 1/4 of channels (dim // 4) with 3x3 Conv
- Leaves the remaining 3/4 of channels untouched (identity pass-through)
- Reduces redundant feature computation and preserves raw infrared spatial cues.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.conv import Conv


class PConv(nn.Module):
    """
    Partial Convolution (PConv from FasterNet):
    Applies 3x3 Conv only on the first 1/n_div of input channels.
    """
    def __init__(self, dim: int, n_div: int = 4):
        super().__init__()
        self.dim_conv = max(1, dim // n_div)
        self.dim_untouched = dim - self.dim_conv
        self.conv = nn.Conv2d(
            self.dim_conv,
            self.dim_conv,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(self.dim_conv)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.act(self.bn(self.conv(x1)))
        return torch.cat((x1, x2), dim=1)


class PConvBottleneck(Bottleneck):
    """
    Bottleneck using PConv as the intermediate spatial transformation:
      cv1 (1x1) -> PConv (3x3 on 1/4 channels) -> cv2 (1x1) -> residual add
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
        n_div: int = 4,
    ):
        super().__init__(c1, c2, shortcut=shortcut, g=g, k=k, e=e)
        c_ = int(c2 * e)
        # cv1: 1x1 conv from c1 to c_
        self.cv1 = Conv(c1, c_, 1, 1)
        # pconv: 3x3 partial conv on c_
        self.pconv = PConv(c_, n_div=n_div)
        # cv2: 1x1 conv from c_ to c2
        self.cv2 = Conv(c_, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.cv2(self.pconv(self.cv1(x)))
        return x + res if self.add else res
