# finetune.py
import os
from pathlib import Path
import torch

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    from ultralytics import YOLO
    model = YOLO("runs/detect/runs/train/household_v1/weights/best.pt")

    results = model.train(
        data     = str(Path("household-1/data_fixed.yaml").resolve()),
        epochs   = 10,
        imgsz    = 640,
        batch    = 5,
        patience = 5,
        device   = 0,
        workers  = 0,
        amp      = True,
        cache    = False,
        lr0      = 0.0005,
        lrf      = 0.01,
        project  = "runs/train",
        name     = "household_v2_realworld",
        exist_ok = True,
        verbose  = True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n✅ Fine-tuning complete!")
    print(f"📦 New best weights: {best}")
    print(f"\nUpdate detector.py:")
    print(f'   model = YOLO("{best}")')