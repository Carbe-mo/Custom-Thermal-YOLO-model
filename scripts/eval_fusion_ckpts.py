"""Phase-1: re-evaluate existing fusion checkpoints on val/test under several protocols (no training)."""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import load_checkpoint
import models.v36_dual_stream_registered.model  # noqa: F401  (register classes for unpickling)
import models.v37_cs_cmaf_fusion.model  # noqa: F401

DATA = ROOT / "data" / "fusion6_data.yaml"
CKPTS = {
    "v36_main": ROOT / "runs/v36_dual_stream_registered/v36_main/weights/best.pt",
    "v36_eca_rgb": ROOT / "runs/v36_dual_stream_registered/v36_eca_rgb/weights/best.pt",
    "v37a": ROOT / "runs/v37_cs_cmaf_fusion/v37a_warm_ca_e100/weights/best.pt",
}
OUT = ROOT / "results" / "phase1_fusion_eval.json"


def ev(w, split, rect, imgsz, conf=0.001, iou=0.7):
    model, _ = load_checkpoint(str(w))
    model = model.float()
    v = DetectionValidator(args=dict(data=str(DATA), split=split, imgsz=imgsz, batch=8, rect=rect, conf=conf, iou=iou,
                                     workers=0, device=0, project=str(ROOT / "runs" / "phase1"), name="tmp",
                                     exist_ok=True, plots=False, verbose=False))
    v(model=model)
    m = v.metrics
    return dict(mAP50=round(float(m.box.map50) * 100, 2), mAP50_95=round(float(m.box.map) * 100, 2),
                P=round(float(m.box.mp) * 100, 2), R=round(float(m.box.mr) * 100, 2))


res = {}
for name, w in CKPTS.items():
    for split in ("val", "test"):
        for rect, imgsz in ((True, 640), (False, 640), (True, 768), (True, 832)):
            for iou in (0.7, 0.6):
                k = f"{name}|{split}|rect={rect}|imgsz={imgsz}|iou={iou}"
                try:
                    res[k] = ev(w, split, rect, imgsz, iou=iou)
                except Exception as e:  # noqa: BLE001
                    res[k] = {"error": str(e)[:200]}
                print(k, res[k], flush=True)
                OUT.write_text(json.dumps(res, indent=2))
print("DONE")
