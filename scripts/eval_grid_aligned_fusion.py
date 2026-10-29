"""
Exact 2D-Grid Anchor-Aligned Confidence-Weighted Fusion + Flip TTA across existing checkpoints.
At each spatial anchor (stride 8, 16, 32), un-flips horizontal TTA on the 2D feature grid
so every model and flipped view aligns 1-to-1 per anchor cell, and computes confidence-weighted
bounding box regression + weighted class probability fusion.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import load_checkpoint
import models.v36_dual_stream_registered.model  # noqa: F401
import models.v37_cs_cmaf_fusion.model  # noqa: F401
import common.modules  # noqa: F401

DATA = ROOT / "data" / "fusion6_data.yaml"


def unflip_anchor_grid(pred: torch.Tensor, img_h: int, img_w: int, strides=(8, 16, 32)) -> torch.Tensor:
    """
    pred: [B, 4+nc, N] from horizontally flipped input image.
    1) Flips cx -> img_w - cx
    2) Reverses the W dimension of each stride level (P3, P4, P5) so anchor indices match unflipped image 1-to-1!
    """
    B, C, _ = pred.shape
    out = pred.clone()
    out[:, 0, :] = img_w - out[:, 0, :]

    chunks = []
    offset = 0
    for s in strides:
        hs, ws = img_h // s, img_w // s
        n = hs * ws
        lvl = out[:, :, offset:offset + n].view(B, C, hs, ws)
        lvl = torch.flip(lvl, dims=[3]).reshape(B, C, n)
        chunks.append(lvl)
        offset += n
    return torch.cat(chunks, dim=2)


class GridAlignedFusionEnsemble(nn.Module):
    def __init__(self, models_cfg: list[tuple[nn.Module, float, bool]], use_flip: bool = False,
                 flip_rel_weight: float = 0.5, conf_power: float = 1.0, HybridConcatFlip: bool = False):
        super().__init__()
        self.submodels = nn.ModuleList([m for m, _, _ in models_cfg])
        self.weights = [w for _, w, _ in models_cfg]
        self.is_3ch = [is3 for _, _, is3 in models_cfg]
        self.use_flip = use_flip
        self.flip_rel_weight = flip_rel_weight
        self.conf_power = conf_power
        self.hybrid_concat_flip = HybridConcatFlip

        base = self.submodels[0]
        self.stride = base.stride
        self.names = base.names
        self.yaml = getattr(base, "yaml", {"nc": 4, "channels": 6})
        self.end2end = False

    def _forward_one(self, m: nn.Module, x: torch.Tensor, is3: bool) -> torch.Tensor:
        inp = x[:, 3:6] if is3 else x
        y = m(inp)
        return y[0] if isinstance(y, (tuple, list)) else y

    def _fuse_aligned(self, preds: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
        # preds: list of [B, 4+nc, N]
        w_sum = sum(weights)
        cls_acc = 0
        box_num = 0
        box_den = 0
        for p, w in zip(preds, weights):
            boxes = p[:, :4, :]      # [B, 4, N]
            scores = p[:, 4:, :]     # [B, nc, N]
            cls_acc = cls_acc + (w / w_sum) * scores
            if self.conf_power > 0:
                max_sc = scores.max(dim=1, keepdim=True).values.clamp(min=1e-4) ** self.conf_power
                bw = w * max_sc
            else:
                bw = w
            box_num = box_num + bw * boxes
            box_den = box_den + bw
        fused_boxes = box_num / box_den
        return torch.cat([fused_boxes, cls_acc], dim=1)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        img_h, img_w = x.shape[2], x.shape[3]
        orig_preds, orig_w = [], []
        flip_preds, flip_w = [], []

        for m, w, is3 in zip(self.submodels, self.weights, self.is_3ch):
            p = self._forward_one(m, x, is3)
            orig_preds.append(p)
            orig_w.append(w)

        if self.use_flip:
            x_flip = torch.flip(x, dims=[3])
            for m, w, is3 in zip(self.submodels, self.weights, self.is_3ch):
                pf = self._forward_one(m, x_flip, is3)
                pf_aligned = unflip_anchor_grid(pf, img_h, img_w)
                flip_preds.append(pf_aligned)
                flip_w.append(w * self.flip_rel_weight)

        if not self.use_flip:
            return self._fuse_aligned(orig_preds, orig_w), []

        if self.hybrid_concat_flip:
            # Fuse all original views into 8400 anchors, fuse all flipped views into 8400 anchors, then concat -> 16800
            f_orig = self._fuse_aligned(orig_preds, orig_w)
            f_flip = self._fuse_aligned(flip_preds, orig_w)
            f_flip_scaled = f_flip.clone()
            f_flip_scaled[:, 4:, :] *= self.flip_rel_weight
            return torch.cat([f_orig, f_flip_scaled], dim=-1), []
        else:
            return self._fuse_aligned(orig_preds + flip_preds, orig_w + flip_w), []


def evaluate(wrapper: nn.Module, split: str, rect: bool, iou: float = 0.65):
    wrapper.eval().cuda()
    v = DetectionValidator(args=dict(
        data=str(DATA), split=split, imgsz=640, batch=8, rect=rect,
        conf=0.001, iou=iou, workers=0, device=0,
        project=str(ROOT / "runs" / "phase1"), name="grid_tmp",
        exist_ok=True, plots=False, verbose=False,
    ))
    v(model=wrapper)
    m = v.metrics
    per_cls = {v.names[i]: round(float(m.box.ap50[j]) * 100, 2)
               for j, i in enumerate(m.box.ap_class_index)}
    return dict(
        mAP50=round(float(m.box.map50) * 100, 2),
        mAP50_95=round(float(m.box.map) * 100, 2),
        P=round(float(m.box.mp) * 100, 2),
        R=round(float(m.box.mr) * 100, 2),
        per_cls=per_cls,
    )


def main():
    m_main = load_checkpoint(str(ROOT / "runs/v36_dual_stream_registered/v36_main/weights/best.pt"))[0].float().cuda().eval()
    m_eca = load_checkpoint(str(ROOT / "runs/v36_dual_stream_registered/v36_eca_rgb/weights/best.pt"))[0].float().cuda().eval()
    m_thm = load_checkpoint(str(ROOT / "runs/v36_dual_stream_registered/v36_thermal/weights/best.pt"))[0].float().cuda().eval()
    m_v37 = load_checkpoint(str(ROOT / "runs/v37_cs_cmaf_fusion/v37a_warm_ca_e100/weights/best.pt"))[0].float().cuda().eval()
    m_v30 = load_checkpoint(str(ROOT / "runs/v30_v27_cosine_100ep/train/weights/best.pt"))[0].float().cuda().eval()

    experiments = {
        "A_main_eca_v30_cp1": (
            [(m_main, 0.42, False), (m_eca, 0.38, False), (m_v30, 0.20, True)],
            False, 0.5, 1.0, False, 0.65
        ),
        "B_main_eca_v30_gridflip_cp1": (
            [(m_main, 0.42, False), (m_eca, 0.38, False), (m_v30, 0.20, True)],
            True, 0.45, 1.0, False, 0.65
        ),
        "C_main_eca_v30_hybridflip_cp1": (
            [(m_main, 0.42, False), (m_eca, 0.38, False), (m_v30, 0.20, True)],
            True, 0.85, 1.0, True, 0.60
        ),
        "D_main_eca_v30_thm_v37_cp1": (
            [(m_main, 0.35, False), (m_eca, 0.32, False), (m_v30, 0.15, True), (m_thm, 0.10, False), (m_v37, 0.08, False)],
            False, 0.5, 1.0, False, 0.65
        ),
        "E_main_eca_v30_thm_v37_hybridflip": (
            [(m_main, 0.35, False), (m_eca, 0.32, False), (m_v30, 0.15, True), (m_thm, 0.10, False), (m_v37, 0.08, False)],
            True, 0.85, 1.0, True, 0.60
        ),
        "F_main_eca_v30_thm_v37_gridflip": (
            [(m_main, 0.35, False), (m_eca, 0.32, False), (m_v30, 0.15, True), (m_thm, 0.10, False), (m_v37, 0.08, False)],
            True, 0.45, 1.5, False, 0.65
        ),
    }

    out = {}
    for name, (mcfg, uflip, fw, cp, hcf, iou) in experiments.items():
        ens = GridAlignedFusionEnsemble(mcfg, use_flip=uflip, flip_rel_weight=fw, conf_power=cp, HybridConcatFlip=hcf)
        for split in ("val", "test"):
            for rect in (True, False):
                k = f"{name}|{split}|rect={rect}"
                r = evaluate(ens, split=split, rect=rect, iou=iou)
                out[k] = r
                print(f"{k:46s} -> mAP50={r['mAP50']:6.2f}%  mAP50-95={r['mAP50_95']:6.2f}%  P={r['P']:5.2f}%  R={r['R']:5.2f}%  {r['per_cls']}", flush=True)

    p = ROOT / "results" / "phase1_grid_aligned_eval.json"
    p.write_text(json.dumps(out, indent=2))
    print("Saved to", p)


if __name__ == "__main__":
    main()
