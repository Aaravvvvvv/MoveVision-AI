# prune_dataset.py
"""
Prune the household-1 dataset from 101 → 61 classes.

Drops non-inventory labels (Toilet, Egg, LightSwitch, Sink, etc.) and
remaps the remaining class indices to contiguous 0-based IDs.

Creates:   household-pruned/  (train/valid/test splits + data.yaml)
Requires:  household-1/ with the original Roboflow dataset

Run once:  python prune_dataset.py
Then:      python train_pruned.py
"""

import shutil
from pathlib import Path

import yaml

# ── Classes to KEEP — matches estimator.SUPPORTED_MODEL_LABELS exactly ──
KEEP_LABELS = {
    "AlarmClock", "ArmChair", "Bed", "Blinds", "Book", "Bottle", "Bowl",
    "Box", "Cabinet", "Candle", "CellPhone", "Chair", "Cloth",
    "CoffeeMachine", "CoffeeTable", "Cup", "Curtains", "Desk", "DeskLamp",
    "DiningTable", "Drawer", "Dresser", "FloorLamp", "Footstool", "Fork",
    "Fridge", "GarbageCan", "HandTowel", "HousePlant", "Kettle", "Knife",
    "Laptop", "LaundryHamper", "Microwave", "Mirror", "Mug", "Newspaper",
    "Ottoman", "Painting", "Pan", "Pen", "Pencil", "Pillow", "Plate",
    "Poster", "Pot", "RemoteControl", "Safe", "Shelf", "SideTable", "Sofa",
    "Spoon", "Statue", "TeddyBear", "Television", "TissueBox", "Toaster",
    "Towel", "Vase", "WateringCan", "WineBottle",
}


def build_index_mapping(original_names):
    """Build old_index → new_index mapping for kept classes."""
    old_to_new = {}
    new_names = []
    new_idx = 0
    for old_idx, name in enumerate(original_names):
        if name in KEEP_LABELS:
            old_to_new[old_idx] = new_idx
            new_names.append(name)
            new_idx += 1
    return old_to_new, new_names


def prune_label_file(src_path, dst_path, old_to_new):
    """Filter and remap one YOLO label file.  Returns (kept, dropped)."""
    kept = dropped = 0
    lines_out = []

    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old_cls = int(parts[0])
            if old_cls in old_to_new:
                parts[0] = str(old_to_new[old_cls])
                lines_out.append(" ".join(parts))
                kept += 1
            else:
                dropped += 1

    # Write even if empty — YOLO treats it as a negative sample
    with open(dst_path, "w") as f:
        f.write("\n".join(lines_out))

    return kept, dropped


def prune_split(src_base, dst_base, split_name, img_sub, lbl_sub, old_to_new):
    """Prune one dataset split (train / valid / test)."""
    src_img = src_base / img_sub
    src_lbl = src_base / lbl_sub
    dst_img = dst_base / split_name / "images"
    dst_lbl = dst_base / split_name / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    total_kept = total_dropped = images_copied = images_skipped = 0

    img_files = sorted(
        list(src_img.glob("*.jpg"))
        + list(src_img.glob("*.jpeg"))
        + list(src_img.glob("*.png"))
    )

    for img_path in img_files:
        lbl_path = src_lbl / (img_path.stem + ".txt")
        if not lbl_path.exists():
            images_skipped += 1
            continue

        kept, dropped = prune_label_file(
            lbl_path, dst_lbl / lbl_path.name, old_to_new
        )
        total_kept += kept
        total_dropped += dropped

        shutil.copy2(img_path, dst_img / img_path.name)
        images_copied += 1

    print(
        f"  {split_name:<6}  {images_copied:>5} images  |  "
        f"{total_kept:>6} annotations kept, {total_dropped:>5} dropped  |  "
        f"{images_skipped} skipped (no label)"
    )
    return images_copied


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SRC = Path("household-1")
    DST = Path("household-pruned")
    YAML_PATH = SRC / "data.yaml"

    if not YAML_PATH.exists():
        print(f"❌ {YAML_PATH} not found — make sure the Roboflow dataset is downloaded.")
        exit(1)

    # Handle existing output
    if DST.exists():
        answer = input(f"⚠️  '{DST}' already exists. Delete and recreate? (y/n): ").strip().lower()
        if answer == "y":
            shutil.rmtree(DST)
            print(f"   Deleted {DST}\n")
        else:
            print("   Aborted.")
            exit(0)

    # ── Read original class list ─────────────────────────────────────
    data = yaml.safe_load(YAML_PATH.read_text())
    original_names = data["names"]
    old_to_new, new_names = build_index_mapping(original_names)

    dropped_names = [n for n in original_names if n not in KEEP_LABELS]

    print(f"📋 Original:  {len(original_names)} classes")
    print(f"✂️  Keeping:   {len(new_names)} classes")
    print(f"🗑️  Dropping:  {len(dropped_names)} → {', '.join(dropped_names)}")
    print()

    # ── Prune each split ─────────────────────────────────────────────
    SPLITS = [
        ("train", "train/images", "train/labels"),
        ("valid", "valid/images", "valid/labels"),
        ("test",  "test/images",  "test/labels"),
    ]

    print("🔄 Processing splits...")
    print(f"  {'split':<6}  {'images':>11}  |  {'annotations':<35}  |  notes")
    print(f"  {'-'*6}  {'-'*11}  |  {'-'*35}  |  {'-'*20}")

    total_images = 0
    for split_name, img_sub, lbl_sub in SPLITS:
        if (SRC / img_sub).exists():
            total_images += prune_split(SRC, DST, split_name, img_sub, lbl_sub, old_to_new)
        else:
            print(f"  {split_name:<6}  ⏭️  directory not found, skipping")

    # ── Write new data.yaml ──────────────────────────────────────────
    new_data = {
        "names": new_names,
        "nc": len(new_names),
        "train": str((DST / "train" / "images").resolve()),
        "val":   str((DST / "valid" / "images").resolve()),
        "test":  str((DST / "test"  / "images").resolve()),
    }

    new_yaml = DST / "data.yaml"
    new_yaml.write_text(yaml.dump(new_data, default_flow_style=False, sort_keys=False))

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅  Pruned dataset ready!")
    print(f"{'='*60}")
    print(f"  📁 Location:   {DST.resolve()}")
    print(f"  📋 Classes:    {len(new_names)}  (was {len(original_names)})")
    print(f"  🖼️  Images:     {total_images} total")
    print(f"  📄 Config:     {new_yaml}")
    print(f"\n  Next step → python train_pruned.py")
