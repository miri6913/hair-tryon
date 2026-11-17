"""Downscale the Kaggle 'age detection human faces 18-60' photos into the Phase 0 test set.

Source: https://www.kaggle.com/datasets/trainingdatapro/age-detection-human-faces-18-60-years
Place the extracted images under data/kaggle/ (gitignored; do not redistribute the dataset).
Output: data/photos_kaggle/*.jpg (long side 1024) and data/manifest_kaggle.csv.
"""
import csv
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "kaggle"
OUT = ROOT / "data" / "photos_kaggle"
MANIFEST = ROOT / "data" / "manifest_kaggle.csv"
LONG_SIDE = 1024


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(SRC.rglob("*.jpg")):
        img = cv2.imread(str(path))  # applies EXIF orientation
        if img is None:
            print(f"skip unreadable {path}")
            continue
        h, w = img.shape[:2]
        scale = LONG_SIDE / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        rel = path.relative_to(SRC)
        name = f"kaggle_{rel.parts[0]}_{rel.parts[1]}_{path.stem}.jpg"  # split_agegroup_index
        cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({"file": name, "tag": "selfie", "title": str(rel).replace("\\", "/"),
                     "license": "Kaggle/TrainingData (non-redistribution)", "author": "TrainingData", "source": "kaggle"})
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "tag", "title", "license", "author", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} photos -> {OUT}")


if __name__ == "__main__":
    main()
