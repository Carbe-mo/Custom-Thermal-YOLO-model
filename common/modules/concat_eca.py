"""
common/modules/concat_eca.py — Cross-Scale Concatenation with Efficient Channel Attention (Concat_ECA)
======================================================================================================

In standard PANet, multi-scale feature maps from different depths (e.g. shallow high-res spatial features
and deep low-res semantic features) are simply stacked along the channel dimension with equal weighting.

Concat_ECA concatenates feature maps and immediately applies an Efficient Channel Attention (ECA) block:
  1. out = torch.cat(x, dimension)
  2. out = self.eca(out)

This dynamically learns an adaptive channel-weighting gate between:
  - Dense local spatial features (from the backbone skip connections)
  - Broad semantic context (from upsampled deep layers)

Preserves 100% pretrained weight transfer compatibility with yolov8n.pt (adds only 3 params per Concat).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.modules.eca import ECA


class Concat_ECA(nn.Module):
    """
    Concatenate a list of tensors along dimension (default 1) followed by
    Efficient Channel Attention (ECA) to adaptively reweight channels across scales.
    """
    def __init__(self, dimension: int = 1, eca_k: int = 3) -> None:
        super().__init__()
        self.d = dimension
        self.eca = ECA(k_size=eca_k)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        out = torch.cat(x, self.d)
        return self.eca(out)


_REGISTERED = False


def register_concat_eca() -> None:
    """Register Concat_ECA into Ultralytics nn modules and parser."""
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.__dict__["Concat_ECA"] = Concat_ECA
    modules.__dict__["Concat_ECA"] = Concat_ECA

    orig_parse_model = tasks.parse_model

    def custom_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)

        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m_name, args = item
                    if m_name == "Concat_ECA":
                        custom_layers[(section, i)] = ("Concat_ECA", n, args)
                        # Substitute with standard Concat for channel propagation
                        dim = args[0] if len(args) > 0 else 1
                        d_mod[section][i] = [f, n, "Concat", [dim]]

        model, save = orig_parse_model(d_mod, ch, verbose=verbose)

        bb_len = len(d_mod.get("backbone", []))
        for (section, i), (m_type, n_repeats, args) in custom_layers.items():
            idx = i if section == "backbone" else bb_len + i
            f = d[section][i][0]

            dim = args[0] if len(args) > 0 else 1
            eca_k = args[1] if len(args) > 1 else 3

            layer = Concat_ECA(dimension=dim, eca_k=eca_k)
            layer.i = idx
            layer.f = f
            layer.type = "common.modules.concat_eca.Concat_ECA"
            layer.np = sum(p.numel() for p in layer.parameters())

            model[idx] = layer

        return model, save

    tasks.parse_model = custom_parse_model
    _REGISTERED = True
