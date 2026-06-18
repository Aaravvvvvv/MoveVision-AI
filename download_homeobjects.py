# download_homeobjects.py
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import SETTINGS
from pathlib import Path

print("⬇️ Downloading HomeObjects-3K (~390MB)...")
check_det_dataset("HomeObjects-3K.yaml")

path = Path(SETTINGS['datasets_dir']) / 'homeobjects-3K'
print(f"\n✅ Downloaded to: {path}")
print(f"📁 Train images: {len(list((path / 'train' / 'images').glob('*.*')))}")
print(f"📁 Val images:   {len(list((path / 'val' / 'images').glob('*.*')))}")