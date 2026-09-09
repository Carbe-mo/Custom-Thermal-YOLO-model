import sys, os, traceback
sys.path.insert(0, ".")
from common.modules.c2f_dcn import register_c2f_dcn
register_c2f_dcn()
from ultralytics.models.yolo.detect import DetectionTrainer

try:
    print("Initializing trainer...")
    trainer = DetectionTrainer(overrides=dict(
        model="models/v6_c2f_dcn/model.yaml",
        data="data/thermal_data.yaml",
        epochs=1,
        batch=4,
        workers=0,
        device=0,
        project="runs/test_dcn",
        name="test_direct",
        verbose=True
    ))
    print("Calling trainer.train()...")
    trainer.train()
    print("TRAINER TRAIN FINISHED SUCCESSFULLY!")
except BaseException as e:
    print(f"EXCEPTION TYPE: {type(e)}")
    print(f"EXCEPTION STR: {e}")
    traceback.print_exc()
