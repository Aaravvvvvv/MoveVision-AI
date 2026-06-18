# train.py
import os
from pathlib import Path
import yaml
from ultralytics import YOLO

if __name__ == '__main__':

    # ── Set working directory to script location (PyCharm fix) ──
    os.chdir(Path(__file__).parent)

    # ── Verify GPU ───────────────────────────────────────────────
    import torch
    if not torch.cuda.is_available():
        print("❌ GPU not detected — training will be very slow on CPU")
    else:
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Fix data.yaml paths for Windows ─────────────────────────
    yaml_path = Path("household-1/data.yaml")
    data      = yaml.safe_load(yaml_path.read_text())

    base = Path("household-1").resolve()
    data["train"] = str(base / "train"  / "images")
    data["val"]   = str(base / "valid"  / "images")
    data["test"]  = str(base / "test"   / "images")

    fixed_yaml = Path("household-1/data_fixed.yaml")
    fixed_yaml.write_text(yaml.dump(data))

    # ── Verify images exist before starting ─────────────────────
    train_count = len(list(Path(data["train"]).glob("*.*")))
    val_count   = len(list(Path(data["val"]).glob("*.*")))
    print(f"\n📁 Train images: {train_count}")
    print(f"📁 Val images:   {val_count}")

    if train_count == 0:
        print("❌ No training images found — check dataset path")
        exit()

    # ── Load pretrained model ────────────────────────────────────
    model = YOLO("weights/yolov8l.pt")

    # ── Train ────────────────────────────────────────────────────
    print("\n🚀 Starting training...\n")

    results = model.train(
        data    = str(fixed_yaml.resolve()),
        epochs  = 50,
        imgsz   = 640,
        batch   = 5,        # safe for 4GB VRAM without AMP
        patience= 10,       # stops early if no improvement for 10 epochs
        device  = 0,        # GPU
        workers = 0,        # Windows fix — no multiprocessing
        amp     = True,    # disable AMP — was causing silent crash
        cache   = False,    # don't cache images — saves VRAM
        project = "runs/train",
        name    = "household_v1",
        exist_ok= True,
        verbose = True,
    )

    # ── Done ─────────────────────────────────────────────────────
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n✅ Training complete!")
    print(f"📦 Best weights saved to: {best}")
    print(f"\nNext step — update detector.py:")
    print(f'   model = YOLO("{best}")')