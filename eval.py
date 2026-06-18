# eval.py
"""
Evaluate a YOLO household model — prints mAP, per-class AP, and saves
confusion matrix / PR curves automatically.

Usage:
    # Evaluate current production model on the original 101-class test set
    python eval.py

    # Evaluate a newly trained pruned model
    python eval.py --model runs/train/household_pruned_v1/weights/best.pt

    # Evaluate on a specific data config / split
    python eval.py --model weights/household_v2_best.pt --data household-1/data_fixed.yaml --split val

    # Also save annotated prediction images on sample photos
    python eval.py --save-predictions
"""

import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO


def evaluate_model(model_path, data_yaml, split, imgsz, save_predictions, save_dir):
    """Run full evaluation and print a report."""

    print()
    print("=" * 60)
    print("  MoveVision AI — Model Evaluation")
    print("=" * 60)
    print(f"  📦 Model:    {model_path}")
    print(f"  📄 Dataset:  {data_yaml}")
    print(f"  🔍 Split:    {split}")
    print(f"  📐 Img size: {imgsz}")
    print()

    model = YOLO(str(model_path))

    # ── Run validation ───────────────────────────────────────────────
    print("🔄 Running evaluation (this may take a few minutes)...\n")

    metrics = model.val(
        data      = str(Path(data_yaml).resolve()),
        split     = split,
        imgsz     = imgsz,
        batch     = 5,
        workers   = 0,
        device    = 0,
        verbose   = False,
        save_json = False,
        plots     = True,      # auto-saves confusion matrix, P/R curves, F1 curve
        project   = save_dir,
        name      = "results",
        exist_ok  = True,
    )

    # ── Overall metrics ──────────────────────────────────────────────
    print("=" * 60)
    print("  OVERALL METRICS")
    print("=" * 60)
    print(f"  {'mAP@50:':<20} {metrics.box.map50:.4f}   ({metrics.box.map50:.1%})")
    print(f"  {'mAP@50-95:':<20} {metrics.box.map:.4f}   ({metrics.box.map:.1%})")
    print(f"  {'Precision:':<20} {metrics.box.mp:.4f}")
    print(f"  {'Recall:':<20} {metrics.box.mr:.4f}")
    print()

    # ── Per-class breakdown ──────────────────────────────────────────
    names = model.names
    per_class_ap = metrics.box.maps  # per-class AP@50-95

    class_results = []
    for i, ap in enumerate(per_class_ap):
        if i < len(names):
            class_results.append((names[i], ap))

    class_results.sort(key=lambda x: x[1], reverse=True)

    print("=" * 60)
    print("  PER-CLASS AP@50-95  (sorted best → worst)")
    print("=" * 60)
    print(f"  {'':3} {'Class':<25} {'AP':>8}")
    print(f"  {'':3} {'-' * 25} {'-' * 8}")

    good = medium = poor = 0
    for name, ap in class_results:
        if ap >= 0.5:
            icon = "🟢"
            good += 1
        elif ap >= 0.2:
            icon = "🟡"
            medium += 1
        else:
            icon = "🔴"
            poor += 1
        print(f"  {icon} {name:<25} {ap:>8.4f}")

    print()
    print(f"  Summary:  🟢 {good} good (≥0.5)  |  🟡 {medium} medium (0.2–0.5)  |  🔴 {poor} poor (<0.2)")
    print()

    # ── Optional: save predictions on sample images ──────────────────
    if save_predictions:
        _save_sample_predictions(model, imgsz, save_dir)

    # ── Output files ─────────────────────────────────────────────────
    # Ultralytics 8.4+ saves plots to its own save_dir (may differ from
    # the project arg).  Retrieve the actual path from the validator.
    try:
        results_dir = Path(model.validator.save_dir)
    except AttributeError:
        results_dir = Path(save_dir) / "results"

    pred_dir = Path(save_dir) / "predictions"

    print("=" * 60)
    print("  OUTPUT FILES")
    print("=" * 60)
    print(f"  📂 Results folder:     {results_dir}")
    print(f"  📊 Confusion matrix:   {results_dir / 'confusion_matrix.png'}")
    print(f"  📊 Normalized matrix:  {results_dir / 'confusion_matrix_normalized.png'}")
    print(f"  📈 Precision-Recall:   {results_dir / 'BoxPR_curve.png'}")
    print(f"  📈 F1 curve:           {results_dir / 'BoxF1_curve.png'}")
    print(f"  📈 P curve:            {results_dir / 'BoxP_curve.png'}")
    print(f"  📈 R curve:            {results_dir / 'BoxR_curve.png'}")
    if save_predictions:
        print(f"  🖼️  Predictions:       {pred_dir}")
    print()

    return metrics


def _save_sample_predictions(model, imgsz, save_dir):
    """Run the model on local test images and save annotated results."""
    print("🖼️  Generating sample predictions on test images...\n")

    test_images = [
        "test_bedroom.jpg",
        "test_kitchen.jpg",
        "test_room2.jpg",
        "test_home.jpg",
        "myroom.jpeg",
        "test_yolo_couch_tv_room.jpg",
    ]

    pred_dir = Path(save_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for img_name in test_images:
        img_path = Path(img_name)
        if not img_path.exists():
            print(f"  ⏭️  {img_name} — not found, skipping")
            continue

        results = model(str(img_path), imgsz=imgsz, conf=0.25, verbose=False)
        annotated = results[0].plot()
        out_path = pred_dir / f"pred_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated)

        det_count = len(results[0].boxes)
        det_names = [model.names[int(c)] for c in results[0].boxes.cls]
        print(f"  ✅ {img_name}: {det_count} detections → {out_path.name}")
        if det_names:
            print(f"     Items: {', '.join(det_names)}")

    print()


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO household model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py
  python eval.py --model runs/train/household_pruned_v1/weights/best.pt
  python eval.py --model weights/household_v2_best.pt --data household-1/data_fixed.yaml
  python eval.py --save-predictions
        """,
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model weights (.pt).  Default: weights/household_v2_best.pt",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to data.yaml.  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--save-predictions", action="store_true",
        help="Save annotated predictions on local test images",
    )
    parser.add_argument(
        "--save-dir", type=str, default="runs/eval",
        help="Directory for output files (default: runs/eval)",
    )

    args = parser.parse_args()
    os.chdir(Path(__file__).parent)

    # ── Resolve model path ───────────────────────────────────────────
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = Path("weights/household_v2_best.pt")

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        exit(1)

    # ── Resolve data.yaml ────────────────────────────────────────────
    if args.data:
        data_yaml = Path(args.data)
    else:
        # Auto-detect: pruned dataset for pruned models, original otherwise
        pruned_yaml = Path("household-pruned/data.yaml")
        original_yaml = Path("household-1/data_fixed.yaml")

        if "pruned" in str(model_path).lower() and pruned_yaml.exists():
            data_yaml = pruned_yaml
        elif pruned_yaml.exists():
            data_yaml = pruned_yaml
        elif original_yaml.exists():
            data_yaml = original_yaml
        else:
            print("❌ No data.yaml found! Specify with --data")
            exit(1)

    if not data_yaml.exists():
        print(f"❌ Data config not found: {data_yaml}")
        exit(1)

    # ── Run ──────────────────────────────────────────────────────────
    evaluate_model(
        model_path=model_path,
        data_yaml=data_yaml,
        split=args.split,
        imgsz=args.imgsz,
        save_predictions=args.save_predictions,
        save_dir=args.save_dir,
    )
