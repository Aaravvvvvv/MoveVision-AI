# train_pruned.py
"""
Train YOLOv8-L on the pruned 61-class household dataset.

Prerequisites:
    1. Run prune_dataset.py first to create household-pruned/
    2. GPU recommended (training is configured for ~4 GB VRAM)

Usage:  python train_pruned.py
"""

import os
from pathlib import Path

import yaml
from ultralytics import YOLO


if __name__ == "__main__":

    # ── Working directory = script location ──────────────────────────
    os.chdir(Path(__file__).parent)

    # ── GPU check ────────────────────────────────────────────────────
    import torch

    if not torch.cuda.is_available():
        print("❌ GPU not detected — training will be extremely slow on CPU.")
        print("   Consider using Google Colab or a cloud GPU.")
    else:
        gpu = torch.cuda.get_device_properties(0)
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {gpu.total_mem / 1e9:.1f} GB")

    # ── Verify pruned dataset ────────────────────────────────────────
    data_yaml = Path("household-pruned/data.yaml")
    if not data_yaml.exists():
        print("\n❌ Pruned dataset not found!")
        print("   Run this first:  python prune_dataset.py")
        exit(1)

    data = yaml.safe_load(data_yaml.read_text())
    print(f"\n📋 Classes: {data['nc']}")

    train_dir = Path(data["train"])
    val_dir = Path(data["val"])
    train_count = len(list(train_dir.glob("*.*"))) if train_dir.exists() else 0
    val_count = len(list(val_dir.glob("*.*"))) if val_dir.exists() else 0
    print(f"📁 Train images: {train_count}")
    print(f"📁 Val images:   {val_count}")

    if train_count == 0:
        print("\n❌ No training images found — check prune_dataset.py output.")
        exit(1)

    # ── Load COCO-pretrained base model ──────────────────────────────
    base_weights = Path("weights/yolov8l.pt")
    if not base_weights.exists():
        print(f"\n❌ Base weights not found at {base_weights}")
        print("   Download: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt")
        exit(1)

    model = YOLO(str(base_weights))

    # ── Train ────────────────────────────────────────────────────────
    print("\n🚀 Starting training on pruned dataset...\n")

    results = model.train(
        data     = str(data_yaml.resolve()),
        epochs   = 50,
        imgsz    = 640,
        batch    = 5,          # safe for 4 GB VRAM
        patience = 10,         # early stopping if no improvement for 10 epochs
        device   = 0,          # GPU
        workers  = 0,          # Windows compatibility
        amp      = True,       # mixed precision — saves VRAM
        cache    = False,      # don't cache images in RAM
        project  = "runs/train",
        name     = "household_pruned_v1",
        exist_ok = True,
        verbose  = True,
    )

    # ── Done ─────────────────────────────────────────────────────────
    best = Path(results.save_dir) / "weights" / "best.pt"

    print(f"\n{'='*60}")
    print(f"  ✅  Training complete!")
    print(f"{'='*60}")
    print(f"  📦 Best weights:  {best}")
    print(f"  📊 Logs:          {results.save_dir}")
    print()
    print(f"  Next steps:")
    print(f"    1. Evaluate:")
    print(f"       python eval.py --model \"{best}\"")
    print()
    print(f"    2. Compare against current model:")
    print(f"       python eval.py --model weights/household_v2_best.pt --data household-1/data_fixed.yaml")
    print()
    print(f"    3. Deploy (if better):")
    print(f"       copy \"{best}\" weights\\household_v3_best.pt")
    print(f"       Then update MODEL_PATH in detector.py")
