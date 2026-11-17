"""Look review sheet: diverse photos x shades, hair-region crops, v1 (flat target) next to v2 (measured shades).

Output: results/review/<name>.png. This is for a human to judge; no number here is a pass/fail gate.
  python scripts/review_sheet.py [--name sheet] [--shades blonde,light_brown,brown,black]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vto_spike import viz  # noqa: E402
from vto_spike.recolor import recolor_hair  # noqa: E402
from vto_spike.recolor2 import SHADES, recolor_hair2  # noqa: E402

MASKS = ROOT / "results" / "masks" / "npy"
# dark / mid / light men and women, curly, coily, braids, glasses, dyed
PHOTOS = [
    "kaggle_train_18-20_19", "kaggle_train_21-30_11", "kaggle_train_51-60_4", "kaggle_test_18-20_25",
    "coily_010", "curly_005", "braids_015", "glasses_025", "dyed_022",
]
OLD_GOLDEN_BLONDE = ((214, 170, 90), 0.8)


def find_photo(stem: str) -> Path:
    for d in ("photos_kaggle", "photos_wiki", "photos"):
        for ext in (".jpg", ".jpeg", ".png"):
            p = ROOT / "data" / d / f"{stem}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(stem)


def crop_to_hair(img: np.ndarray, mask: np.ndarray, pad: float = 0.12):
    ys, xs = np.where(mask > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
    sl = (slice(max(y0 - py, 0), min(y1 + py, img.shape[0])), slice(max(x0 - px, 0), min(x1 + px, img.shape[1])))
    return sl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sheet")
    ap.add_argument("--shades", default="golden_blonde,honey,chestnut,black,blue,violet,emerald,copper")
    ap.add_argument("--height", type=int, default=300)
    args = ap.parse_args()
    names = args.shades.split(",")

    rows = []
    for stem in PHOTOS:
        rgb = cv2.cvtColor(cv2.imread(str(find_photo(stem))), cv2.COLOR_BGR2RGB)
        mask = np.load(MASKS / f"{stem}.npy").astype(np.float32)
        if mask.shape != rgb.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        if (mask > 0.5).sum() < 500:
            print("skip (no hair in mask):", stem)
            continue
        sl = crop_to_hair(rgb, mask)
        tiles = [viz.label(rgb[sl], stem[:22])]
        c, lift = OLD_GOLDEN_BLONDE
        tiles.append(viz.label(recolor_hair(rgb, mask, c, intensity=0.85, lift=lift)[sl], "v1 golden"))
        for n in names:
            tiles.append(viz.label(recolor_hair2(rgb, mask, SHADES[n], intensity=1.0)[sl], f"v2 {n}"))
        rows.append(viz.row(tiles, args.height))
    out = ROOT / "results" / "review" / f"{args.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(viz.stack(rows), cv2.COLOR_RGB2BGR))
    print("wrote", out)


if __name__ == "__main__":
    main()
