"""
common/modules/fusion.py — RGB-Thermal Fusion Modules
=====================================================

Implements fusion modules for multi-modal detection:
- SliceChannels: Splits 6-channel input into RGB/Thermal streams
- AddFusion:     Element-wise addition of two feature maps
- GatedFusion:   Learned gated combination (MBNet / GMFNet style)

Usage in model.yaml:
  - [-1, 1, SliceChannels, [0, 3]]          # Extract RGB channels
  - [[5, 16], 1, GatedFusion, [64]]         # Fuse P3 features
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


class SliceChannels(nn.Module):
    """
    Slices a contiguous range of channels from the input tensor.
    
    Args:
        start_ch: Start channel index (inclusive).
        end_ch:   End channel index (exclusive).
    
    Example:
        SliceChannels(0, 3)  → extracts channels 0,1,2 (RGB)
        SliceChannels(3, 6)  → extracts channels 3,4,5 (Thermal replicated)
    """
    def __init__(self, start_ch: int, end_ch: int):
        super().__init__()
        self.start_ch = start_ch
        self.end_ch = end_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.start_ch:self.end_ch, :, :]


class AddFusion(nn.Module):
    """
    Element-wise addition of two feature maps.
    
    Input: list of two tensors [F_rgb, F_thermal] with identical shapes.
    Output: F_rgb + F_thermal (same shape).
    """
    def __init__(self, c1: int = 0):
        super().__init__()
        # c1 is accepted but unused — needed for YAML parse_model compatibility

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return x[0] + x[1]


class GatedFusion(nn.Module):
    """
    Dynamic Gated Fusion (inspired by MBNet / GMFNet).
    
    Learns a gating map g ∈ [0,1]^C that dynamically combines two modalities:
        F_fused = g * F_rgb + (1 - g) * F_thermal
    
    The gate is computed from the concatenation of both feature maps
    via a 1×1 convolution + sigmoid.
    
    Args:
        c: Number of channels in each input feature map.
    """
    def __init__(self, c: int):
        super().__init__()
        # Gate: concat(F_rgb, F_th) → 1x1 conv → sigmoid → [0,1]^C
        self.gate_conv = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.Sigmoid(),
        )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        f_rgb, f_th = x[0], x[1]
        cat = torch.cat([f_rgb, f_th], dim=1)
        g = self.gate_conv(cat)
        return g * f_rgb + (1.0 - g) * f_th


# ---------------------------------------------------------------------------
# Registration with Ultralytics
# ---------------------------------------------------------------------------
_REGISTERED = False


def register_fusion_modules() -> None:
    """
    Register SliceChannels, AddFusion, and GatedFusion into Ultralytics
    nn modules and parse_model so model.yaml can reference them directly.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    for cls in [SliceChannels, AddFusion, GatedFusion]:
        tasks.__dict__[cls.__name__] = cls
        modules.__dict__[cls.__name__] = cls

    _REGISTERED = True


if __name__ == "__main__":
    register_fusion_modules()

    # Unit test: SliceChannels
    x = torch.randn(2, 6, 40, 40)
    rgb = SliceChannels(0, 3)(x)
    th = SliceChannels(3, 6)(x)
    assert rgb.shape == (2, 3, 40, 40)
    assert th.shape == (2, 3, 40, 40)
    print(f"SliceChannels: {x.shape} → RGB {rgb.shape}, TH {th.shape}")

    # Unit test: AddFusion
    f1 = torch.randn(2, 64, 20, 20)
    f2 = torch.randn(2, 64, 20, 20)
    fused_add = AddFusion(64)([f1, f2])
    assert fused_add.shape == f1.shape
    print(f"AddFusion: 2×{f1.shape} → {fused_add.shape}")

    # Unit test: GatedFusion
    gf = GatedFusion(64)
    fused_gate = gf([f1, f2])
    assert fused_gate.shape == f1.shape
    n_params = sum(p.numel() for p in gf.parameters())
    print(f"GatedFusion: 2×{f1.shape} → {fused_gate.shape}  ({n_params} params)")

    print("\nAll fusion module tests passed!")
