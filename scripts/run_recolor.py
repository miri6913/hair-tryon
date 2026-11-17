"""Phase 0 step 4: recolor experiments using the refined masks from run_masks.py.

Outputs (results/recolor/):
  palette_<stem>.png     original + palette sweep (LAB method, lift tuned per color)
  dark_to_blonde.png     rows = dark-haired photos; cols = original | naive HSV | LAB no lift | LAB lift
  intensity_<stem>.png   intensity slider 0.25 .. 1.0
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vto_spike import viz  # noqa: E402
from vto_spike.recolor import naive_hue_shift, recolor_hair  # noqa: E402

PHOTOS = ROOT / "data" / "photos_kaggle"
MASKS = ROOT / "results" / "masks" / "npy"
OUT = ROOT / "results" / "recolor"

# name: (rgb, lift). lift > 0 for colors lighter than typical hair.
PALETTE = {
    "platinum": ((235, 220, 190), 0.9),
    "golden blonde": ((214, 170, 90), 0.8),
    "copper": ((178, 82, 40), 0.35),
    "burgundy": ((110, 20, 45), 0.0),
    "chestnut": ((100, 60, 35), 0.0),
    "ash blue": ((90, 120, 175), 0.3),
    "violet": ((130, 70, 170), 0.2),
    "emerald": ((20, 130, 90), 0.2),
}


def load(stem: str):
    rgb = cv2.cvtColor(cv2.imread(str(PHOTOS / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
    mask = np.load(MASKS / f"{stem}.npy").astype(np.float32)
    return rgb, mask


def hair_lightness(rgb: np.ndarray, mask: np.ndarray) -> float:
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)[..., 0]
    w = mask > 0.5
    return float(lab[w].mean()) if w.sum() > 500 else float("nan")


def save(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in MASKS.glob("kaggle_*.npy"))
    light = {}
    for s in stems:
        rgb, mask = load(s)
        light[s] = hair_lightness(rgb, mask)
    valid = {s: v for s, v in light.items() if not np.isnan(v) and 0.03 < (load(s)[1] > 0.5).mean() < 0.4}
    ranked = sorted(valid, key=valid.get)
    dark, mid = ranked[:6], ranked[len(ranked) // 2: len(ranked) // 2 + 2]
    print("darkest hair:", [(s, round(valid[s], 1)) for s in dark])

    # palette sweep + intensity slider on a few photos (dark + mid)
    for s in dark[:2] + mid[:1]:
        rgb, mask = load(s)
        tiles = [viz.label(rgb, s)] + [viz.label(recolor_hair(rgb, mask, c, lift=lift), name) for name, (c, lift) in PALETTE.items()]
        save(OUT / f"palette_{s}.png", viz.stack([viz.row(tiles[:5], 300), viz.row(tiles[5:], 300)]))
        c, lift = PALETTE["copper"]
        sl = [viz.label(rgb, "0.0")] + [viz.label(recolor_hair(rgb, mask, c, intensity=i, lift=lift), f"{i:.2f}") for i in (0.25, 0.5, 0.75, 1.0)]
        save(OUT / f"intensity_{s}.png", viz.row(sl, 300))

    # dark -> blonde: naive vs LAB (no lift) vs LAB (lift)
    blonde = PALETTE["golden blonde"][0]
    rows = []
    for s in dark:
        rgb, mask = load(s)
        rows.append(viz.row([
            viz.label(rgb, f"{s} L={valid[s]:.0f}"),
            viz.label(naive_hue_shift(rgb, mask, blonde), "naive HSV"),
            viz.label(recolor_hair(rgb, mask, blonde, lift=0.0), "LAB lift 0"),
            viz.label(recolor_hair(rgb, mask, blonde, lift=0.8, detail_gain=1.0), "lift .8 gain 1.0 eps 12"),
            viz.label(recolor_hair(rgb, mask, blonde, lift=0.8, detail_gain=1.6, eps=20.0), "lift .8 gain 1.6 eps 20"),
        ], 260))
    save(OUT / "dark_to_blonde.png", viz.stack(rows))

    # zoomed edge crops: guard off vs on, to inspect the mask-boundary rim
    crops = []
    for s in dark[:4]:
        rgb, mask = load(s)
        ys, xs = np.where(mask > 0.5)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        pad = 20
        sl = (slice(max(y0 - pad, 0), y1 + pad), slice(max(x0 - pad, 0), x1 + pad))
        kw = dict(lift=0.8, detail_gain=1.0)
        off = recolor_hair(rgb, mask, blonde, edge_guard=False, mask_contrast=False, **kw)
        guard = recolor_hair(rgb, mask, blonde, edge_guard=True, mask_contrast=False, **kw)
        both = recolor_hair(rgb, mask, blonde, edge_guard=True, mask_contrast=True, **kw)
        crops.append(viz.row([viz.label(rgb[sl], s), viz.label(off[sl], "before"), viz.label(guard[sl], "guard"),
                              viz.label(both[sl], "guard + contrast")], 300))
    save(OUT / "edge_crops.png", viz.stack(crops))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
