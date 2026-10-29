"""
Evaluate Modality-Aware Class-Specific Anchor-Aligned Fusion (MACAF) across existing checkpoints.
Physical rationale:
  - Thermal-only stream (v30, v36_thermal) has high emissivity contrast on warm 'person' and 'trolley'
    targets, but lower contrast on thermally-equilibrated 'bag' / 'unattended' objects.
  - Registered RGB+Thermal streams (v36_main with CA+ECA, v36_eca_rgb with ECA) excel on 'bag' and
    'unattended' anomalies via high-frequency RGB texture + spatial coordinate attention.
  - Assigning class-aware modality weights W[m, c] at each spatial anchor routes each class to its
    physically optimal sensor stream.
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


class ClassAwareGridFusion(nn.Module):
    """
    models_cfg: list of (model, [w_person, w_trolley, w_bag, w_unattended], box_w, is_3ch)
    """
    def __init__(self, models_cfg, use_flip: bool = True, flip_rel_w: float = 0.5,
                 conf_power: float = 1.25, hybrid_concat_flip: bool = False):
        super().__init__()
        self.submodels = nn.ModuleList([m for m, _, _, _ in models_cfg])
        # class weights shape [M, 4, 1]
        cw = torch.tensor([cw_list for _, cw_list, _, _ in models_cfg], dtype=torch.float32)
        cw = cw / cw.sum(dim=0, keepdim=True).clamp(min=1e-6)
        self.register_buffer("cls_weights", cw.unsqueeze(-1))  # [M, 4, 1]
        self.box_weights = [bw for _, _, bw, _ in models_cfg]
        self.is_3ch = [is3 for _, _, _, is3 in models_cfg]
        self.use_flip = use_flip
        self.flip_rel_w = flip_rel_w
        self.conf_power = conf_power
        self.hybrid_concat_flip = hybrid_concat_flip

        base = self.submodels[0]
        self.stride = base.stride
        self.names = base.names
        self.yaml = getattr(base, "yaml", {"nc": 4, "channels": 6})
        self.end2end = False

    def _forward_one(self, m: nn.Module, x: torch.Tensor, is3: bool) -> torch.Tensor:
        inp = x[:, 3:6] if is3 else x
        y = m(inp)
        return y[0] if isinstance(y, (tuple, list)) else y

    def _fuse_set(self, preds: list[torch.Tensor]) -> torch.Tensor:
        cls_acc = 0
        box_num = 0
        box_den = 0
        for idx, (p, bw) in enumerate(zip(preds, self.box_weights)):
            boxes = p[:, :4, :]    # [B, 4, N]
            scores = p[:, 4:, :]   # [B, 4, N]
            w_c = self.cls_weights[idx].unsqueeze(0)  # [1, 4, 1]
            cls_acc = cls_acc + w_c * scores

            max_sc = (w_c * scores).max(dim=1, keepdim=True).values.clamp(min=1e-4) ** self.conf_power
            w_b = bw * max_sc
            box_num = box_num + w_b * boxes
            box_den = box_den + w_b
        return torch.cat([box_num / box_den, cls_acc], dim=1)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        img_h, img_w = x.shape[2], x.shape[3]
        orig_preds = [self._forward_one(m, x, is3) for m, is3 in zip(self.submodels, self.is_3ch)]
        f_orig = self._fuse_set(orig_preds)

        if not self.use_flip:
            return f_orig, []

        x_flip = torch.flip(x, dims=[3])
        flip_preds = [unflip_anchor_grid(self._forward_one(m, x_flip, is3), img_h, img_w)
                      for m, is3 in zip(self.submodels, self.is_3ch)]
        f_flip = self._fuse_set(flip_preds)

        if self.hybrid_concat_flip:
            f_flip_s = f_flip.clone()
            f_flip_s[:, 4:, :] *= self.flip_rel_w
            return torch.cat([f_orig, f_flip_s], dim=-1), []
        else:
            # Anchor-aligned blend of f_orig and f_flip
            w0, wf = 1.0, self.flip_rel_w
            cls_out = (w0 * f_orig[:, 4:, :] + wf * f_flip[:, 4:, :]) / (w0 + wf)
            c0 = f_orig[:, 4:, :].max(dim=1, keepdim=True).values.clamp(min=1e-4) ** self.conf_power
            cf = (wf * f_flip[:, 4:, :]).max(dim=1, keepdim=True).values.clamp(min=1e-4) ** self.conf_power
            box_out = (c0 * f_orig[:, :4, :] + cf * f_flip[:, :4, :]) / (c0 + cf)
            return torch.cat([box_out, cls_out], dim=1), []


def evaluate(wrapper: nn.Module, split: str, rect: bool, iou: float = 0.65):
    wrapper.eval().cuda()
    v = DetectionValidator(args=dict(
        data=str(DATA), split=split, imgsz=640, batch=8, rect=rect,
        conf=0.001, iou=iou, workers=0, device=0,
        project=str(ROOT / "runs" / "phase1"), name="macaf_tmp",
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

    # First check individual baselines for v30, v36_thm, v37a
    singles = {
        "single_v30_thm": [(m_v30, [1, 1, 1, 1], 1.0, True)],
        "single_v36_thm": [(m_thm, [1, 1, 1, 1], 1.0, False)],
        "single_v37a":    [(m_v37, [1, 1, 1, 1], 1.0, False)],
    }
    res_all = {}
    for sname, cfg in singles.items():
        w = ClassAwareGridFusion(cfg, use_flip=False)
        for split in ("val", "test"):
            for rect in (True, False):
                k = f"{sname}|{split}|rect={rect}"
                r = evaluate(w, split, rect, iou=0.65)
                res_all[k] = r
                print(f"{k:44s} -> mAP50={r['mAP50']:6.2f}%  mAP50-95={r['mAP50_95']:6.2f}%  {r['per_cls']}", flush=True)

    # Class order: [person, trolley, bag, unattended]
    macaf_configs = {
        "MACAF_3m_grid": (
            [
                (m_main, [0.40, 0.40, 0.42, 0.55], 0.45, False),
                (m_eca,  [0.35, 0.38, 0.46, 0.35], 0.40, False),
                (m_v30,  [0.25, 0.22, 0.12, 0.10], 0.15, True),
            ],
            True, 0.45, 1.25, False, 0.65
        ),
        "MACAF_3m_hybrid": (
            [
                (m_main, [0.40, 0.40, 0.42, 0.55], 0.45, False),
                (m_eca,  [0.35, 0.38, 0.46, 0.35], 0.40, False),
                (m_v30,  [0.25, 0.22, 0.12, 0.10], 0.15, True),
            ],
            True, 0.85, 1.25, True, 0.60
        ),
        "MACAF_4m_grid": (
            [
                (m_main, [0.36, 0.35, 0.40, 0.52], 0.40, False),
                (m_eca,  [0.32, 0.35, 0.44, 0.32], 0.36, False),
                (m_v30,  [0.20, 0.18, 0.10, 0.08], 0.14, True),
                (m_thm,  [0.12, 0.12, 0.06, 0.08], 0.10, False),
            ],
            True, 0.45, 1.25, False, 0.65
        ),
        "MACAF_4m_hybrid": (
            [
                (m_main, [0.36, 0.35, 0.40, 0.52], 0.40, False),
                (m_eca,  [0.32, 0.35, 0.44, 0.32], 0.36, False),
                (m_v30,  [0.20, 0.18, 0.10, 0.08], 0.14, True),
                (m_thm,  [0.12, 0.12, 0.06, 0.08], 0.10, False),
            ],
            True, 0.85, 1.25, True, 0.60
        ),
    }

    for cname, (cfg, uflip, fw, cp, hcf, iou) in macaf_configs.items():
        w = ClassAwareGridFusion(cfg, use_flip=uflip, flip_rel_w=fw, conf_power=cp, hybrid_concat_flip=hcf)
        for split in ("val", "test"):
            for rect in (True, False):
                k = f"{cname}|{split}|rect={rect}"
                r = evaluate(w, split, rect, iou=iou)
                res_all[k] = r
                print(f"{k:44s} -> mAP50={r['mAP50']:6.2f}%  mAP50-95={r['mAP50_95']:6.2f}%  P={r['P']:5.2f}%  R={r['R']:5.2f}%  {r['per_cls']}", flush=True)

    out_file = ROOT / "results" / "phase1_macaf_eval.json"
    out_file.write_text(json.dumps(res_all, indent=2))
    print("Saved to", out_file)


if __name__ == "__main__":
    main()
