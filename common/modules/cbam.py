"""
common/modules/cbam.py — Convolutional Block Attention Module (CBAM)
=====================================================================

Implements the standard CBAM (Woo et al., ECCV 2018):
1. Channel Attention Module (CAM):
   - Computes both Global AvgPool and Global MaxPool across spatial dims (H, W).
   - Passes both through a shared 2-layer MLP with reduction ratio r=16.
   - Sums outputs and applies Sigmoid to produce channel weights M_c.
   - Refined features: F_c = M_c(F) * F

2. Spatial Attention Module (SAM):
   - Computes Channel AvgPool and Channel MaxPool across channels.
   - Concatenates into a 2-channel spatial map: [Avg(F_c), Max(F_c)].
   - Convolves with a 7x7 filter and applies Sigmoid to produce spatial weights M_s.
   - Refined features: F_s = M_s(F_c) * F_c

3. Residual Connection:
   - F_out = F_s + F (maintains gradient flow and preserves baseline representations).
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM).
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool, _ = torch.max(torch.max(x, dim=2, keepdim=True)[0], dim=3, keepdim=True)
        
        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)
        
        weights = self.sigmoid(avg_out + max_out)
        return x * weights


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM).
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale))
        return x * scale


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Channel + Spatial Attention)
    with residual connection.
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channels = channels
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ca(x)
        out = self.sa(out)
        return out + x


_REGISTERED = False

def register_cbam() -> None:
    """
    Register CBAM into Ultralytics nn modules and parse_model
    so that model.yaml can reference 'CBAM' directly.
    """
    global _REGISTERED
    if _REGISTERED:
        return
        
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules
    
    tasks.__dict__["CBAM"] = CBAM
    modules.__dict__["CBAM"] = CBAM
    
    orig_parse_model = tasks.parse_model
    
    def custom_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)
        
        # Intercept CBAM in backbone or head
        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m_name, args = item
                    if m_name == "CBAM":
                        custom_layers[(section, i)] = ("CBAM", args)
                        # Replace temporarily with nn.Identity so parse_model computes channel propagation correctly
                        d_mod[section][i] = [f, n, "nn.Identity", []]
                        
        model, save = orig_parse_model(d_mod, ch, verbose=verbose)
        
        # Trace feature shapes through layers to determine channel counts
        dummy = torch.zeros(1, 3 if isinstance(ch, int) else ch[0], 64, 64)
        features = {}
        bb_len = len(d_mod.get("backbone", []))
        
        for idx, layer in enumerate(model):
            is_cbam = idx in [i if s == "backbone" else bb_len + i for (s, i) in custom_layers.keys()]
            if is_cbam:
                in_ch = dummy.shape[1]
                cbam_layer = CBAM(in_ch)
                cbam_layer.i = idx
                cbam_layer.f = -1
                cbam_layer.type = "common.modules.cbam.CBAM"
                cbam_layer.np = sum(p.numel() for p in cbam_layer.parameters())
                model[idx] = cbam_layer
                dummy = cbam_layer(dummy)
            else:
                section = "backbone" if idx < bb_len else "head"
                sec_idx = idx if idx < bb_len else idx - bb_len
                f = d_mod[section][sec_idx][0]
                if isinstance(f, int) and f == -1:
                    dummy = layer(dummy)
                elif isinstance(f, list):
                    dummy = layer([features[x if x >= 0 else idx + x] for x in f])
                else:
                    dummy = layer(features[f if f >= 0 else idx + f])
            features[idx] = dummy
            
        return model, save

    tasks.parse_model = custom_parse_model
    _REGISTERED = True


if __name__ == "__main__":
    register_cbam()
    cbam = CBAM(channels=64)
    x = torch.randn(2, 64, 40, 40)
    out = cbam(x)
    print(f"CBAM Input:  {x.shape}")
    print(f"CBAM Output: {out.shape}")
    assert out.shape == (2, 64, 40, 40)
    n_params = sum(p.numel() for p in cbam.parameters())
    print(f"CBAM Parameters: {n_params}")
    print("CBAM unit test passed successfully!")
