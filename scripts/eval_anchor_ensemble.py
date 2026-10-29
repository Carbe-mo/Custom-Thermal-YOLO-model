"""
Evaluate Anchor-Aligned Multi-Model & Flip-TTA Ensembles on Val (213) and Test (108) splits
using the exact Ultralytics DetectionValidator pipeline.
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


def flip_decoded_preds(pred: torch.Tensor, img_w: int) -> torch.Tensor:
    """pred: [B, 4+nc, N] where box is cx, cy, w, h in pixel coords."""
    out = pred.clone()
    out[:, 0, :] = img_w - out[:, 0, :]
    return out


class MultiModelTTAWrapper(nn.Module):
    """
    Wraps multiple 6-ch (or 3-ch thermal) YOLO models and optional horizontal flip TTA.
    Supports:
      mode="concat": concatenates predictions along anchor dim [B, 4+nc, N*K] before NMS
      mode="avg":    averages anchor-aligned predictions (only for non-flipped same-grid models)
    """
    def __init__(self, models_list: list[tuple[nn.Module, float, bool]], use_flip: bool = False,
                 flip_weight: float = 0.5, mode: str = "concat"):
        super().__init__()
        self.submodels = nn.ModuleList([m for m, _, _ in models_list])
        self.weights = [w for _, w, _ in models_list]
        self.is_3ch_thm = [is3 for _, _, is3 in models_list]
        self.use_flip = use_flip
        self.flip_weight = flip_weight
        self.mode = mode

        base = self.submodels[0]
        self.stride = base.stride
        self.names = base.names
        self.yaml = getattr(base, "yaml", {"nc": 4, "channels": 6})
        self.end2end = False

    def _run_single(self, m: nn.Module, x: torch.Tensor, is_3ch: bool) -> torch.Tensor:
        inp = x[:, 3:6] if is_3ch else x
        out = m(inp)
        return out[0] if isinstance(out, (tuple, list)) else out

    def forward(self, x: torch.Tensor, *args, **kwargs):
        img_w = x.shape[-1]
        preds = []
        total_w = sum(self.weights)
        for m, w, is3 in zip(self.submodels, self.weights, self.is_3ch_thm):
            p = self._run_single(m, x, is3)
            if self.mode == "avg":
                preds.append(p * (w / total_w))
            else:
                p_scaled = p.clone()
                p_scaled[:, 4:, :] *= w
                preds.append(p_scaled)

            if self.use_flip:
                x_flip = torch.flip(x, dims=[3])
                pf = self._run_single(m, x_flip, is3)
                pf = flip_decoded_preds(pf, img_w)
                if self.mode == "concat":
                    pf_scaled = pf.clone()
                    pf_scaled[:, 4:, :] *= (w * self.flip_weight)
                    preds.append(pf_scaled)

        if self.mode == "avg" and not self.use_flip:
            return sum(preds), []
        return torch.cat(preds, dim=-1), []


def run_eval(wrapper: nn.Module, split: str, rect: bool, imgsz: int = 640, iou: float = 0.6, conf: float = 0.001):
    wrapper.eval().cuda()
    v = DetectionValidator(args=dict(
        data=str(DATA), split=split, imgsz=imgsz, batch=8, rect=rect,
        conf=conf, iou=iou, workers=0, device=0,
        project=str(ROOT / "runs" / "phase1"), name="ens_tmp",
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

    configs = {
        "1_v36_main_single": ([(m_main, 1.0, False)], False, 1.0, "concat", 0.65),
        "2_v36_eca_single": ([(m_eca, 1.0, False)], False, 1.0, "concat", 0.65),
        "3_v36_main_flipTTA": ([(m_main, 1.0, False)], True, 0.9, "concat", 0.60),
        "4_v36_eca_flipTTA": ([(m_eca, 1.0, False)], True, 0.9, "concat", 0.60),
        "5_avg_main_eca": ([(m_main, 0.5, False), (m_eca, 0.5, False)], False, 1.0, "avg", 0.65),
        "6_concat_main_eca": ([(m_main, 1.0, False), (m_eca, 1.0, False)], False, 1.0, "concat", 0.60),
        "7_concat_main_eca_flip": ([(m_main, 1.0, False), (m_eca, 1.0, False)], True, 0.85, "concat", 0.55),
        "8_avg_main_eca_v30": ([(m_main, 0.4, False), (m_eca, 0.4, False), (m_v30, 0.2, True)], False, 1.0, "avg", 0.65),
        "9_concat_main_eca_v30_v37": ([(m_main, 1.0, False), (m_eca, 1.0, False), (m_v30, 0.7, True), (m_v37, 0.6, False)], False, 1.0, "concat", 0.55),
        "10_concat_all_flip": ([(m_main, 1.0, False), (m_eca, 1.0, False), (m_v30, 0.7, True), (m_thm, 0.6, False)], True, 0.85, "concat", 0.55),
    }

    results = {}
    for cname, (mlist, uflip, fw, mode, iou) in configs.items():
        w = MultiModelTTAWrapper(mlist, use_flip=uflip, flip_weight=fw, mode=mode)
        for split in ("val", "test"):
            for rect in (True, False):
                k = f"{cname}|{split}|rect={rect}"
                res = run_eval(w, split=split, rect=rect, iou=iou)
                results[k] = res
                print(f"{k:45s} -> mAP50={res['mAP50']:6.2f}%  mAP50-95={res['mAP50_95']:6.2f}%  P={res['P']:5.2f}%  R={res['R']:5.2f}%  {res['per_cls']}", flush=True)

    out_path = ROOT / "results" / "phase1_ensemble_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    print("Saved to", out_path)


if __name__ == "__main__":
    main()
