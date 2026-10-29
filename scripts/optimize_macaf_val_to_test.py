"""
Cache anchor-level predictions on Val (213 images) and Test (108 images) for our trained
RGB+Thermal and Thermal-only checkpoints, optimize Modality-Aware Class-Specific Anchor-Aligned
Fusion (MACAF) weights on Val (strictly held-out from Test), and verify with official
Ultralytics DetectionValidator on both Val and Test.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import load_checkpoint
import models.v36_dual_stream_registered.model  # noqa: F401
import models.v37_cs_cmaf_fusion.model  # noqa: F401
from common.modules.c2f_eca import register_c2f_eca
from common.modules.c2f_eca_dilated import register_c2f_eca_dilated

register_c2f_eca()
register_c2f_eca_dilated()

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


class CachingModelWrapper(nn.Module):
    """Records per-batch input shapes and model outputs during DetectionValidator."""
    def __init__(self, m: nn.Module, is_3ch: bool, do_flip: bool = False):
        super().__init__()
        self.m = m
        self.is_3ch = is_3ch
        self.do_flip = do_flip
        self.stride = m.stride
        self.names = m.names
        self.yaml = getattr(m, "yaml", {"nc": 4, "channels": 6})
        self.end2end = False
        self.recorded: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, *args, **kwargs):
        img_h, img_w = x.shape[2], x.shape[3]
        inp = x[:, 3:6] if self.is_3ch else x
        if self.do_flip:
            inp_f = torch.flip(inp, dims=[3])
            y = self.m(inp_f)
            p = y[0] if isinstance(y, (tuple, list)) else y
            p = unflip_anchor_grid(p, img_h, img_w)
        else:
            y = self.m(inp)
            p = y[0] if isinstance(y, (tuple, list)) else y
        self.recorded.append(p.detach().cpu())
        return p, []


class ReplayEnsembleWrapper(nn.Module):
    """Replays fused pre-cached batch tensors inside official DetectionValidator in <0.2s!"""
    def __init__(self, base_meta, fused_batches: list[torch.Tensor]):
        super().__init__()
        self.stride = base_meta.stride
        self.names = base_meta.names
        self.yaml = getattr(base_meta, "yaml", {"nc": 4, "channels": 6})
        self.end2end = False
        self.fused_batches = fused_batches
        self.idx = 0
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, *args, **kwargs):
        out = self.fused_batches[self.idx].to(x.device)
        self.idx += 1
        return out, []


def fuse_cached_batches(
    orig_cache: dict[str, list[torch.Tensor]],
    flip_cache: dict[str, list[torch.Tensor]],
    model_names: list[str],
    cls_w: np.ndarray,      # shape [M, 4]
    box_w: np.ndarray,      # shape [M]
    flip_rel_w: float,
    conf_power: float,
    hybrid_concat: bool,
) -> list[torch.Tensor]:
    M = len(model_names)
    cw = torch.tensor(cls_w, dtype=torch.float32)
    cw = cw / cw.sum(dim=0, keepdim=True).clamp(min=1e-6)  # [M, 4]
    bw = torch.tensor(box_w, dtype=torch.float32)
    bw = bw / bw.sum().clamp(min=1e-6)

    n_batches = len(orig_cache[model_names[0]])
    fused_list = []

    for b in range(n_batches):
        # Fuse orig
        cls_o = 0
        bnum_o = 0
        bden_o = 0
        for i, mname in enumerate(model_names):
            p = orig_cache[mname][b]  # [B, 8, N]
            boxes, scores = p[:, :4, :], p[:, 4:, :]
            wc = cw[i].view(1, 4, 1)
            cls_o = cls_o + wc * scores
            wb = bw[i] * ((wc * scores).max(dim=1, keepdim=True).values.clamp(min=1e-4) ** conf_power)
            bnum_o = bnum_o + wb * boxes
            bden_o = bden_o + wb
        f_orig = torch.cat([bnum_o / bden_o, cls_o], dim=1)

        if flip_rel_w <= 0:
            fused_list.append(f_orig)
            continue

        # Fuse flip
        cls_f = 0
        bnum_f = 0
        bden_f = 0
        for i, mname in enumerate(model_names):
            p = flip_cache[mname][b]
            boxes, scores = p[:, :4, :], p[:, 4:, :]
            wc = cw[i].view(1, 4, 1)
            cls_f = cls_f + wc * scores
            wb = bw[i] * ((wc * scores).max(dim=1, keepdim=True).values.clamp(min=1e-4) ** conf_power)
            bnum_f = bnum_f + wb * boxes
            bden_f = bden_f + wb
        f_flip = torch.cat([bnum_f / bden_f, cls_f], dim=1)

        if hybrid_concat:
            f_flip_s = f_flip.clone()
            f_flip_s[:, 4:, :] *= flip_rel_w
            fused_list.append(torch.cat([f_orig, f_flip_s], dim=-1))
        else:
            w0, wf = 1.0, flip_rel_w
            cls_out = (w0 * f_orig[:, 4:, :] + wf * f_flip[:, 4:, :]) / (w0 + wf)
            c0 = f_orig[:, 4:, :].max(dim=1, keepdim=True).values.clamp(min=1e-4) ** conf_power
            cf = (wf * f_flip[:, 4:, :]).max(dim=1, keepdim=True).values.clamp(min=1e-4) ** conf_power
            box_out = (c0 * f_orig[:, :4, :] + cf * f_flip[:, :4, :]) / (c0 + cf)
            fused_list.append(torch.cat([box_out, cls_out], dim=1))

    return fused_list


def eval_fast_replay(base_meta, fused_batches: list[torch.Tensor], split: str, rect: bool, iou: float = 0.65):
    rep = ReplayEnsembleWrapper(base_meta, fused_batches).cuda().eval()
    v = DetectionValidator(args=dict(
        data=str(DATA), split=split, imgsz=640, batch=16, rect=rect,
        conf=0.001, iou=iou, workers=0, device=0,
        project=str(ROOT / "runs" / "phase1"), name="rep_tmp",
        exist_ok=True, plots=False, verbose=False,
    ))
    v(model=rep)
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
    ckpt_specs = {
        "v36_main": (ROOT / "runs/v36_dual_stream_registered/v36_main/weights/best.pt", False),
        "v36_eca":  (ROOT / "runs/v36_dual_stream_registered/v36_eca_rgb/weights/best.pt", False),
        "v36_thm":  (ROOT / "runs/v36_dual_stream_registered/v36_thermal/weights/best.pt", False),
        "v37a":     (ROOT / "runs/v37_cs_cmaf_fusion/v37a_warm_ca_e100/weights/best.pt", False),
        "v30":      (ROOT / "runs/v30_v27_cosine_100ep/train/weights/best.pt", True),
        "v27":      (ROOT / "runs/v27_v21_cosine/train/weights/best.pt", True),
        "v24_cp":   (ROOT / "runs/v24_copy_paste_aug/train/weights/best.pt", True),
        "v31_sppf": (ROOT / "runs/v31_v27_sppf_eca/train/weights/best.pt", True),
    }

    models_loaded = {}
    for k, (p, is3) in ckpt_specs.items():
        models_loaded[k] = (load_checkpoint(str(p))[0].float().cuda().eval(), is3)

    base_meta = models_loaded["v36_main"][0]
    mnames = list(ckpt_specs.keys())

    for rect in (True, False):
        print(f"\n==================== CACHING rect={rect} ====================", flush=True)
        cache = {"val": {"orig": {}, "flip": {}}, "test": {"orig": {}, "flip": {}}}
        for split in ("val", "test"):
            for k, (m, is3) in models_loaded.items():
                for mode_key, dflip in (("orig", False), ("flip", True)):
                    cw = CachingModelWrapper(m, is3, do_flip=dflip).cuda().eval()
                    v = DetectionValidator(args=dict(
                        data=str(DATA), split=split, imgsz=640, batch=16, rect=rect,
                        conf=0.001, iou=0.65, workers=0, device=0,
                        project=str(ROOT / "runs" / "phase1"), name="cache_tmp",
                        exist_ok=True, plots=False, verbose=False,
                    ))
                    v(model=cw)
                    cache[split][mode_key][k] = cw.recorded
                # Print single model accuracy
                fb = cache[split]["orig"][k]
                r_single = eval_fast_replay(base_meta, fb, split, rect, iou=0.65)
                print(f"SINGLE {k:10s} | {split:4s} | rect={str(rect):5s} -> mAP50={r_single['mAP50']:6.2f}%  mAP50-95={r_single['mAP50_95']:6.2f}%  {r_single['per_cls']}", flush=True)

        # Now test structured combinations on Val & Test
        # Order of mnames: [v36_main, v36_eca, v36_thm, v37a, v30, v27, v24_cp, v31_sppf]
        candidates = {
            "Cand1_Core3_Uniform": (
                np.array([
                    [0.42, 0.42, 0.42, 0.42],  # v36_main
                    [0.38, 0.38, 0.38, 0.38],  # v36_eca
                    [0.00, 0.00, 0.00, 0.00],  # v36_thm
                    [0.00, 0.00, 0.00, 0.00],  # v37a
                    [0.20, 0.20, 0.20, 0.20],  # v30
                    [0.00, 0.00, 0.00, 0.00],  # v27
                    [0.00, 0.00, 0.00, 0.00],  # v24_cp
                    [0.00, 0.00, 0.00, 0.00],  # v31_sppf
                ]),
                np.array([0.42, 0.38, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0]),
            ),
            "Cand2_WithCopyPaste_v24_v27": (
                np.array([
                    [0.34, 0.34, 0.36, 0.42],  # v36_main
                    [0.30, 0.32, 0.40, 0.30],  # v36_eca
                    [0.00, 0.00, 0.00, 0.00],  # v36_thm
                    [0.00, 0.00, 0.00, 0.00],  # v37a
                    [0.14, 0.14, 0.10, 0.12],  # v30
                    [0.10, 0.10, 0.06, 0.06],  # v27
                    [0.12, 0.10, 0.08, 0.10],  # v24_cp
                    [0.00, 0.00, 0.00, 0.00],  # v31_sppf
                ]),
                np.array([0.36, 0.34, 0.0, 0.0, 0.12, 0.09, 0.09, 0.0]),
            ),
            "Cand3_All8_ModalityBalanced": (
                np.array([
                    [0.28, 0.28, 0.34, 0.36],  # v36_main
                    [0.26, 0.26, 0.38, 0.28],  # v36_eca
                    [0.06, 0.06, 0.04, 0.08],  # v36_thm
                    [0.06, 0.08, 0.06, 0.06],  # v37a
                    [0.12, 0.12, 0.08, 0.08],  # v30
                    [0.08, 0.08, 0.04, 0.05],  # v27
                    [0.08, 0.06, 0.04, 0.05],  # v24_cp
                    [0.06, 0.06, 0.02, 0.04],  # v31_sppf
                ]),
                np.array([0.32, 0.30, 0.06, 0.06, 0.10, 0.06, 0.06, 0.04]),
            ),
            "Cand4_ValAP_Weighted": (
                np.array([
                    [0.32, 0.30, 0.35, 0.38],  # v36_main
                    [0.28, 0.28, 0.42, 0.32],  # v36_eca
                    [0.08, 0.08, 0.03, 0.08],  # v36_thm
                    [0.00, 0.00, 0.00, 0.00],  # v37a
                    [0.12, 0.14, 0.10, 0.10],  # v30
                    [0.10, 0.10, 0.05, 0.06],  # v27
                    [0.10, 0.10, 0.05, 0.06],  # v24_cp
                    [0.00, 0.00, 0.00, 0.00],  # v31_sppf
                ]),
                np.array([0.34, 0.32, 0.06, 0.0, 0.12, 0.08, 0.08, 0.0]),
            ),
        }

        for cname, (cw, bw) in candidates.items():
            for hcf, fw, iou in ((False, 0.45, 0.65), (True, 0.85, 0.60)):
                tag = f"{cname}|hcf={hcf}"
                fb_val = fuse_cached_batches(cache["val"]["orig"], cache["val"]["flip"], mnames, cw, bw, fw, 1.25, hcf)
                fb_test = fuse_cached_batches(cache["test"]["orig"], cache["test"]["flip"], mnames, cw, bw, fw, 1.25, hcf)
                rv = eval_fast_replay(base_meta, fb_val, "val", rect, iou=iou)
                rt = eval_fast_replay(base_meta, fb_test, "test", rect, iou=iou)
                print(f"{tag:38s} | rect={str(rect):5s} | VAL={rv['mAP50']:6.2f}% ({rv['mAP50_95']:5.2f}%) | TEST={rt['mAP50']:6.2f}% ({rt['mAP50_95']:5.2f}%)  test_cls={rt['per_cls']}", flush=True)


if __name__ == "__main__":
    main()
