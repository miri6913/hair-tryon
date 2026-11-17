"""Phase 1 look work, step 1: per-photo hair color statistics from the Phase 0 masks.

Answers "what does real hair of each shade look like in Lab?" so shades can be built from data instead of a single
target color. Writes results/review/hair_stats.csv (one row per photo) and prints a summary grouped by lightness.

Per photo: L percentiles over confident hair pixels, mean a/b, mean chroma, and mean (a, b) in lightness bands
(shadow / mid / highlight), which is what a flat (a, b) target throws away.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT / "data" / "photos_kaggle"
MASKS = ROOT / "results" / "masks" / "npy"
OUT = ROOT / "results" / "review"


def lab_of(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in sorted(MASKS.glob("kaggle_*.npy")):
        stem = m.stem
        photo = PHOTOS / f"{stem}.jpg"
        if not photo.exists():
            continue
        mask = np.load(m).astype(np.float32)
        lab = lab_of(photo)
        if lab.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_LINEAR)
        # confident interior only: erode so edge mixing with background does not pollute the statistics
        sel = cv2.erode((mask > 0.9).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if sel.sum() < 3000 or sel.mean() > 0.6:
            continue
        L, a, b = lab[..., 0][sel], lab[..., 1][sel], lab[..., 2][sel]
        p = np.percentile(L, [2, 10, 50, 90, 98])
        lo, hi = np.percentile(L, [25, 75])
        band = lambda s: (float(a[s].mean()), float(b[s].mean())) if s.sum() > 50 else (np.nan, np.nan)  # noqa: E731
        (a_s, b_s), (a_m, b_m), (a_h, b_h) = band(L <= lo), band((L > lo) & (L < hi)), band(L >= hi)
        rows.append({
            "stem": stem, "hair_px": int(sel.sum()),
            "L_p2": p[0], "L_p10": p[1], "L_p50": p[2], "L_p90": p[3], "L_p98": p[4],
            "a_mean": float(a.mean()), "b_mean": float(b.mean()),
            "chroma": float(np.hypot(a, b).mean()),
            "a_shadow": a_s, "b_shadow": b_s, "a_mid": a_m, "b_mid": b_m, "a_high": a_h, "b_high": b_h,
        })
    rows.sort(key=lambda r: r["L_p50"])
    with open(OUT / "hair_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    print(f"{len(rows)} photos with usable hair -> {OUT / 'hair_stats.csv'}")
    # group by median lightness into rough shade families
    bins = [(0, 12, "black"), (12, 22, "dark brown"), (22, 35, "brown"), (35, 50, "light brown / dark blonde"),
            (50, 101, "blonde / grey / white")]
    print(f"{'family':28s} {'n':>3s} {'L p2':>6s} {'L p50':>6s} {'L p98':>6s} {'chroma':>7s}"
          f"  (a,b) shadow / mid / high")
    for lo_, hi_, name in bins:
        g = [r for r in rows if lo_ <= r["L_p50"] < hi_]
        if not g:
            continue
        mean = lambda k: float(np.nanmean([r[k] for r in g]))  # noqa: E731
        print(f"{name:28s} {len(g):3d} {mean('L_p2'):6.1f} {mean('L_p50'):6.1f} {mean('L_p98'):6.1f} {mean('chroma'):7.1f}"
              f"  ({mean('a_shadow'):4.1f},{mean('b_shadow'):4.1f}) / ({mean('a_mid'):4.1f},{mean('b_mid'):4.1f})"
              f" / ({mean('a_high'):4.1f},{mean('b_high'):4.1f})")
    print("\nrule of thumb to compare with: current golden-blonde target is L=%.0f a=%.0f b=%.0f" % tuple(
        cv2.cvtColor(np.array([[[214, 170, 90]]], np.float32) / 255.0, cv2.COLOR_RGB2LAB)[0, 0]))


if __name__ == "__main__":
    sys.exit(main())
