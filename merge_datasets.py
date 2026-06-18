# merge_datasets.py
import shutil
from pathlib import Path

# HomeObjects-3K → Your model class mapping
# HomeObjects-3K: bed, sofa, chair, table, lamp, tv, laptop, wardrobe, window, door, potted plant, photo frame
REMAP = {
    0:  6,    # bed          → Bed
    1:  79,   # sofa         → Sofa
    2:  18,   # chair        → Chair
    3:  21,   # table        → CoffeeTable
    4:  34,   # lamp         → FloorLamp
    5:  86,   # tv           → Television
    6:  46,   # laptop       → Laptop
    7:  31,   # wardrobe     → Dresser
    8:  99,   # window       → Window
    9:  None, # door         → skip (not in your model)
    10: 41,   # potted plant → HousePlant
    11: 55,   # photo frame  → Painting
}

def remap_label(src, dst):
    lines_out = []
    with open(src) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            new_cls = REMAP.get(int(parts[0]))
            if new_cls is None:
                continue  # skip door
            parts[0] = str(new_cls)
            lines_out.append(' '.join(parts))
    if lines_out:
        with open(dst, 'w') as f:
            f.write('\n'.join(lines_out))
        return True
    return False

def merge_split(src_img, src_lbl, dst_img, dst_lbl, tag):
    src_img, src_lbl = Path(src_img), Path(src_lbl)
    dst_img, dst_lbl = Path(dst_img), Path(dst_lbl)
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    imgs = list(src_img.glob("*.jpg")) + list(src_img.glob("*.png"))
    copied = skipped = 0

    for img in imgs:
        lbl = src_lbl / (img.stem + '.txt')
        if not lbl.exists():
            skipped += 1
            continue
        new_lbl = dst_lbl / f"ho_{tag}_{img.stem}.txt"
        if remap_label(lbl, new_lbl):
            shutil.copy2(img, dst_img / f"ho_{tag}_{img.name}")
            copied += 1
        else:
            skipped += 1

    print(f"  {tag}: ✅ {copied} copied, ⏭️ {skipped} skipped")
    return copied

if __name__ == '__main__':
    homeobj  = Path("datasets/homeobjects-3K")
    household = Path("household-1")

    if not homeobj.exists():
        print(f"❌ HomeObjects-3K not found at {homeobj}")
        print("Run download_homeobjects.py first!")
        exit(1)

    print("🔄 Merging HomeObjects-3K into your dataset...\n")

    total = merge_split(
        homeobj  / "images" / "train", homeobj  / "labels" / "train",
        household / "train" / "images", household / "train" / "labels",
        "train"
    )
    total += merge_split(
        homeobj  / "images" / "val", homeobj  / "labels" / "val",
        household / "valid" / "images", household / "valid" / "labels",
        "val"
    )

    print(f"\n✅ Done! Added {total} real-world images to your dataset!")
    print(f"📁 Your train folder now has real + synthetic images")
    print(f"\nNow run finetune.py 🚀")