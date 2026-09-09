"""
common/modules/bifpn.py — Bidirectional Feature Pyramid Network (BiFPN)
========================================================================

Implements Weighted Bi-directional Feature Pyramid Network (BiFPN) modules
(Tan et al., EfficientDet, CVPR 2020):

1. Fast Normalized Fusion:
   - Learnable positive weights w_i >= 0 per input stream:
     w_i = ReLU(w_i) / (sum_j ReLU(w_j) + eps)
   - Allows the network to dynamically learn which feature level is more
     informative at each scale.

2. BiFPN_Concat2:
   - 2-input weighted concatenation module with channel-preserving learnable gating.

3. BiFPN_Concat3:
   - 3-input weighted concatenation module incorporating the distinctive BiFPN
     extra residual skip connection directly from the original backbone feature node.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN_Concat2(nn.Module):
    """
    2-Input Fast Normalized Weighted Feature Fusion.
    Output: Concatenated tensor with channels weighted by learnable importance weights.
    """
    def __init__(self, dimension: int = 1, eps: float = 1e-4):
        super().__init__()
        self.d = dimension
        self.eps = eps
        # 2 learnable positive weights initialized to 1.0
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.w, inplace=False)
        weight = w / (torch.sum(w, dim=0) + self.eps)
        # Apply fast normalized weights
        x0 = x[0] * weight[0]
        x1 = x[1] * weight[1]
        return torch.cat([x0, x1], dim=self.d)


class BiFPN_Concat3(nn.Module):
    """
    3-Input Fast Normalized Weighted Feature Fusion.
    Incorporates the original backbone skip connection alongside top-down and bottom-up features.
    """
    def __init__(self, dimension: int = 1, eps: float = 1e-4):
        super().__init__()
        self.d = dimension
        self.eps = eps
        # 3 learnable positive weights initialized to 1.0
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.w, inplace=False)
        weight = w / (torch.sum(w, dim=0) + self.eps)
        # Apply fast normalized weights
        x0 = x[0] * weight[0]
        x1 = x[1] * weight[1]
        x2 = x[2] * weight[2]
        return torch.cat([x0, x1, x2], dim=self.d)


_REGISTERED = False

def register_bifpn() -> None:
    """
    Register BiFPN modules into Ultralytics nn modules and parse_model
    so that model.yaml can reference 'BiFPN_Concat2' and 'BiFPN_Concat3' directly.
    """
    global _REGISTERED
    if _REGISTERED:
        return
        
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules
    
    tasks.__dict__["BiFPN_Concat2"] = BiFPN_Concat2
    tasks.__dict__["BiFPN_Concat3"] = BiFPN_Concat3
    modules.__dict__["BiFPN_Concat2"] = BiFPN_Concat2
    modules.__dict__["BiFPN_Concat3"] = BiFPN_Concat3
    
    orig_parse_model = tasks.parse_model
    
    def custom_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)
        
        # Intercept BiFPN in backbone or head
        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m_name, args = item
                    if m_name in ("BiFPN_Concat2", "BiFPN_Concat3"):
                        custom_layers[(section, i)] = (m_name, args)
                        # Temporarily replace with stock Concat so parse_model computes channel sums correctly
                        d_mod[section][i] = [f, n, "Concat", args]
                        
        model, save = orig_parse_model(d_mod, ch, verbose=verbose)
        
        # Replace Concat placeholders with actual BiFPN_Concat instances
        bb_len = len(d_mod.get("backbone", []))
        for (section, i), (m_type, args) in custom_layers.items():
            idx = i if section == "backbone" else bb_len + i
            f = d[section][i][0]
            dim = args[0] if len(args) > 0 else 1
            
            if m_type == "BiFPN_Concat2":
                layer = BiFPN_Concat2(dimension=dim)
            else:
                layer = BiFPN_Concat3(dimension=dim)
                
            layer.i = idx
            layer.f = f
            layer.type = f"common.modules.bifpn.{m_type}"
            layer.np = sum(p.numel() for p in layer.parameters())
            
            model[idx] = layer
            
        return model, save

    tasks.parse_model = custom_parse_model
    _REGISTERED = True


if __name__ == "__main__":
    register_bifpn()
    # Test Concat2
    bifpn2 = BiFPN_Concat2(dimension=1)
    t1 = torch.randn(2, 64, 40, 40)
    t2 = torch.randn(2, 64, 40, 40)
    out2 = bifpn2([t1, t2])
    print(f"BiFPN_Concat2 Output Shape: {out2.shape}")
    assert out2.shape == (2, 128, 40, 40)
    
    # Test Concat3
    bifpn3 = BiFPN_Concat3(dimension=1)
    t3 = torch.randn(2, 64, 40, 40)
    out3 = bifpn3([t1, t2, t3])
    print(f"BiFPN_Concat3 Output Shape: {out3.shape}")
    assert out3.shape == (2, 192, 40, 40)
    print("BiFPN modules verified successfully!")
