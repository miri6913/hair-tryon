"""Prepare the hairstyle test set: data/hairstyle_test (jpg + HEIC) -> data/hairstyle_set.

Per photo: an upright RGB jpg (EXIF rotation applied, long side capped), a hair mask png (the Phase 0 MediaPipe
segmenter, 0-255 confidence) and a row in data/hairstyle_set.csv. The csv is committed (names and sizes only); the photos,
which include personal ones, are git-ignored. Everything under data/hairstyle_set is what gets copied to the server.
  python scripts/prepare_hairstyle_set.py
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from vto_spike.segment import HairSegmenter  # noqa: E402

SRC = ROOT / "data" / "hairstyle_test"
DST = ROOT / "data" / "hairstyle_set"
CSV = ROOT / "data" / "hairstyle_set.csv"
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-side", type=int, default=1536, help="long side cap in px; the models resize anyway")
    args = ap.parse_args()

    register_heif_opener()
    (DST / "masks").mkdir(parents=True, exist_ok=True)
    seg = HairSegmenter()
    rows = []
    seen = set()
    # sorted so a .HEIC is met before its converted .jpg; both have the same stem, keep one row per stem
    for path in sorted(SRC.iterdir(), key=lambda p: (p.suffix.lower() in {".heic", ".heif"}, p.name)):
        if path.suffix.lower() not in EXTS or path.stem.lower() in seen:
            continue
        stem = path.stem.lower()
        seen.add(stem)
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        if max(img.size) > args.max_side:
            s = args.max_side / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
        img.save(DST / f"{stem}.jpg", quality=95)

        mask = seg(np.asarray(img))
        Image.fromarray((mask * 255).round().astype(np.uint8)).save(DST / "masks" / f"{stem}.png")
        hair = float((mask > 0.5).mean())
        rows.append({"id": stem, "source": path.name, "width": img.width, "height": img.height, "hair_frac": f"{hair:.3f}"})
        print(f"{stem}: {img.width}x{img.height} hair {hair:.1%}")
    seg.close()

    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "width", "height", "hair_frac"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {CSV} ({len(rows)} photos)")


if __name__ == "__main__":
    main()
