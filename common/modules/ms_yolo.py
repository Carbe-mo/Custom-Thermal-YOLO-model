"""
common/modules/ms_yolo.py — Re-implementation of MS-YOLO (Zhang et al., IJCNN 2025)
===================================================================================
Paper:
  Jiali Zhang, Thomas S. White, Haoliang Zhang, Wenqing Hu, Donald C. Wunsch II, Jian Liu,
  "MS-YOLO: Infrared Object Detection for Edge Deployment via MobileNetV4 and SlideLoss",
  2025 International Joint Conference on Neural Networks (IJCNN), pp. 1-8, 2025 (arXiv:2509.21696).

Architecture (Fig. 1 & Fig. 2 of the paper):
  1. Backbone: MobileNetV4-Small (Conv2D stem + Universal Inverted Bottleneck stacks UIBs-1 & UIBs-2
     using ReLU6 activations as specified in Fig. 1 caption & Section III-B).
  2. Neck & Head: YOLOv8n PANet Neck + Detect Head.
  3. Loss: SlideLoss (Eq. 1 of Section III-C) weighting hard samples near the mean IoU threshold mu.
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn


class ConvBNReLU6(nn.Module):
    """Conv2D + BatchNorm2d + ReLU6 (Fig. 1 caption: 'Conv2D represents a convolutional operation
    followed by batch normalization and a ReLU6 activation function')."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class UIBBlock(nn.Module):
    """Single Universal Inverted Bottleneck (UIB) block (Section III-B, Fig. 2):
      (a) optional first depthwise conv with kernel size k1 (if k1 > 0)
      (b) 1x1 expansion conv with expansion ratio r
      (c) optional second depthwise conv with kernel size k2 and stride s (if k2 > 0)
      (d) 1x1 projection conv to c2
    """

    def __init__(self, c1: int, c2: int, k1: int, k2: int, s: int, r: int):
        super().__init__()
        cmid = c1 * r
        self.use_res = (s == 1 and c1 == c2)

        # (a) First depthwise convolution (if k1 > 0). If k2 == 0 and s > 1, apply stride here.
        s1 = s if (k2 == 0 and s > 1) else 1
        if k1 > 0:
            self.dw1 = nn.Sequential(
                nn.Conv2d(c1, c1, kernel_size=k1, stride=s1, padding=k1 // 2, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
            )
        else:
            self.dw1 = nn.Identity()

        # (b) 1x1 expansion convolution
        self.expand = nn.Sequential(
            nn.Conv2d(c1, cmid, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(cmid),
            nn.ReLU6(inplace=True),
        )

        # (c) Second (middle) depthwise convolution with stride s
        if k2 > 0:
            self.dw2 = nn.Sequential(
                nn.Conv2d(cmid, cmid, kernel_size=k2, stride=s, padding=k2 // 2, groups=cmid, bias=False),
                nn.BatchNorm2d(cmid),
                nn.ReLU6(inplace=True),
            )
        else:
            self.dw2 = nn.Identity()

        # (d) 1x1 projection convolution
        self.project = nn.Sequential(
            nn.Conv2d(cmid, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.project(self.dw2(self.expand(self.dw1(x))))
        if self.use_res:
            return x + out
        return out


class UIBs1(nn.Module):
    """UIBs-1 stack from Fig. 2 of MS-YOLO (6 UIB blocks, stride 2 on first block):
      1. k1=5, k2=5, s=2, r=3
      2. k1=0, k2=3, s=1, r=2
      3. k1=0, k2=3, s=1, r=2
      4. k1=0, k2=3, s=1, r=2
      5. k1=0, k2=3, s=1, r=2
      6. k1=3, k2=0, s=1, r=4
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        specs = [
            (c1, c2, 5, 5, 2, 3),
            (c2, c2, 0, 3, 1, 2),
            (c2, c2, 0, 3, 1, 2),
            (c2, c2, 0, 3, 1, 2),
            (c2, c2, 0, 3, 1, 2),
            (c2, c2, 3, 0, 1, 4),
        ]
        self.blocks = nn.Sequential(*(UIBBlock(*sp) for sp in specs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class UIBs2(nn.Module):
    """UIBs-2 stack from Fig. 2 of MS-YOLO (6 UIB blocks, stride 2 on first block):
      1. k1=3, k2=3, s=2, r=6
      2. k1=5, k2=5, s=1, r=4
      3. k1=0, k2=5, s=1, r=4
      4. k1=0, k2=5, s=1, r=3
      5. k1=0, k2=3, s=1, r=4
      6. k1=0, k2=3, s=1, r=4
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        specs = [
            (c1, c2, 3, 3, 2, 6),
            (c2, c2, 5, 5, 1, 4),
            (c2, c2, 0, 5, 1, 4),
            (c2, c2, 0, 5, 1, 3),
            (c2, c2, 0, 3, 1, 4),
            (c2, c2, 0, 3, 1, 4),
        ]
        self.blocks = nn.Sequential(*(UIBBlock(*sp) for sp in specs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class SlideBCEWithLogitsLoss(nn.Module):
    """SlideLoss (Yu et al., 2022; Zhang et al., IJCNN 2025, Section III-C, Eq. 1):
    Dynamically weights classification BCE loss based on prediction-target IoU (or target score
    in TaskAlignedAssigner, which equals IoU^beta * prob^alpha) relative to mean threshold mu:
      f(x) = 1          if x <= mu - 0.1
             e^(1 - mu) if mu - 0.1 < x < mu
             e^(1 - x)  if x >= mu
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.bce(pred, target)
        with torch.no_grad():
            t_f = target.float()
            pos_mask = t_f > 0.0
            if pos_mask.any():
                iou_proxy = t_f[pos_mask]
                mu = iou_proxy.mean().clamp(min=0.1, max=0.95)
                w = torch.ones_like(t_f)
                mid_mask = pos_mask & (t_f > (mu - 0.1)) & (t_f < mu)
                high_mask = pos_mask & (t_f >= mu)
                w[mid_mask] = torch.exp(1.0 - mu)
                w[high_mask] = torch.exp(1.0 - t_f[high_mask])
                w = w.to(loss.dtype)
            else:
                w = torch.ones_like(loss)
        return loss * w


_REGISTERED_MS = False
_SLIDE_LOSS_ENABLED = False
_LOSS_PATCHED = False


def set_slide_loss_enabled(enabled: bool) -> None:
    """Enable or disable SlideLoss (Eq. 1 of Zhang et al., IJCNN 2025) inside v8DetectionLoss."""
    global _SLIDE_LOSS_ENABLED, _LOSS_PATCHED
    _SLIDE_LOSS_ENABLED = enabled
    if not _LOSS_PATCHED:
        from ultralytics.utils.loss import v8DetectionLoss

        orig_init = v8DetectionLoss.__init__

        def patched_init(self, model, *args, **kwargs):
            orig_init(self, model, *args, **kwargs)
            if _SLIDE_LOSS_ENABLED:
                self.bce = SlideBCEWithLogitsLoss()

        v8DetectionLoss.__init__ = patched_init
        _LOSS_PATCHED = True


def register_ms_yolo() -> None:
    """Register ConvBNReLU6, UIBs1, UIBs2 into Ultralytics YAML parser."""
    global _REGISTERED_MS
    if _REGISTERED_MS:
        return
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    for name, cls in (
        ("ConvBNReLU6", ConvBNReLU6),
        ("UIBBlock", UIBBlock),
        ("UIBs1", UIBs1),
        ("UIBs2", UIBs2),
    ):
        tasks.__dict__[name] = cls
        modules.__dict__[name] = cls

    orig_parse_model = tasks.parse_model

    def ms_parse_model(d, ch, verbose=True):
        custom_layers = {}
        d_mod = copy.deepcopy(d)

        for section in ("backbone", "head"):
            if section in d_mod:
                for i, item in enumerate(d_mod[section]):
                    f, n, m, args = item
                    if m in ("ConvBNReLU6", "UIBs1", "UIBs2"):
                        custom_layers[(section, i)] = (m, n, args)
                        # Replace with standard Conv placeholder so channel tracking works identically
                        if m == "ConvBNReLU6":
                            d_mod[section][i] = [f, 1, "Conv", args]
                        else:
                            # UIBs1 and UIBs2 have stride 2 and output args[0] channels
                            d_mod[section][i] = [f, 1, "Conv", [args[0], 3, 2]]

        model, save = orig_parse_model(d_mod, ch, verbose=verbose)

        bb_len = len(d_mod.get("backbone", []))
        for (section, i), (m_type, n_repeats, args) in custom_layers.items():
            idx = i if section == "backbone" else bb_len + i
            f = d[section][i][0]
            c1 = model[idx].conv.in_channels
            c2 = model[idx].conv.out_channels

            if m_type == "ConvBNReLU6":
                k = args[1] if len(args) > 1 else 1
                s = args[2] if len(args) > 2 else 1
                new_layer = ConvBNReLU6(c1, c2, k=k, s=s)
            elif m_type == "UIBs1":
                new_layer = UIBs1(c1, c2)
            elif m_type == "UIBs2":
                new_layer = UIBs2(c1, c2)
            else:
                continue

            new_layer.i = idx
            new_layer.f = f
            new_layer.type = f"common.modules.ms_yolo.{m_type}"
            new_layer.np = sum(p.numel() for p in new_layer.parameters())
            model[idx] = new_layer

        return model, save

    tasks.parse_model = ms_parse_model
    _REGISTERED_MS = True
