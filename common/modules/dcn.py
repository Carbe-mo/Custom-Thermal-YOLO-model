"""
common/modules/dcn.py — Deformable Convolution (DCNv2) & DCNBottleneck
======================================================================

Architectural Design & Choice Rationale:
----------------------------------------
Why Deformable Convolution (DCNv2) for Thermal Object Detection?
1. Thermal IR Object Geometry:
   - Thermal radiation emitted by persons, luggage, and trolleys creates diffuse,
     non-rigid heat silhouettes rather than crisp rectangular RGB boundaries.
   - Standard 3x3 convolutions sample on a rigid fixed grid ([-1, 0, 1] x [-1, 0, 1]),
     which forces the network to capture substantial cold background noise around
     irregular object boundaries.

2. Modulated Deformable Convolution (DCNv2):
   - Learns dynamic 2D offsets (delta_p_k) and modulation masks (delta_m_k) for each
     sampling point:
       y(p) = sum_{k} w_k * delta_m_k(p) * x(p + p_k + delta_p_k(p))
   - Warps the receptive field to dynamically hug the irregular heat contours of
     unattended bags, persons in non-standard postures, and trolleys.

3. DCNBottleneck:
   - Subclass of Ultralytics `Bottleneck`.
   - Replaces the second standard 3x3 convolution (`self.cv2`) with a Modulated
     `DCNv2` block inside the residual path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.ops as ops
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.conv import Conv


class DCNv2(nn.Module):
    """
    Modulated Deformable Convolution v2 (DCNv2).
    Generates spatial sampling offsets and feature modulation masks dynamically from input features.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

        # Offset (2 * k * k) and mask (k * k) generator
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        # Initialize offset to 0 and mask to 0.5
        nn.init.constant_(self.conv_offset_mask.weight, 0.0)
        nn.init.constant_(self.conv_offset_mask.bias, 0.0)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=1)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        out = ops.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=(1, 1),
            mask=mask,
        )
        return self.act(self.bn(out))


class DCNBottleneck(Bottleneck):
    """
    Bottleneck subclass replacing the second 3x3 Conv with Modulated DCNv2.
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
        # Replace standard cv2 with DCNv2
        self.cv2 = DCNv2(c_, c2, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual path: cv1 (3x3) -> cv2 (DCNv2) -> add shortcut
        res = self.cv2(self.cv1(x))
        return x + res if self.add else res


if __name__ == "__main__":
    dcn_b = DCNBottleneck(c1=64, c2=64, shortcut=True)
    x = torch.randn(2, 64, 40, 40)
    out = dcn_b(x)
    print(f"DCNBottleneck input:  {x.shape}")
    print(f"DCNBottleneck output: {out.shape}")
    assert out.shape == (2, 64, 40, 40)
    n_params = sum(p.numel() for p in dcn_b.parameters())
    print(f"DCNBottleneck parameters: {n_params}")
    print("DCNBottleneck verification passed successfully!")
