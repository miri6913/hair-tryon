"""Diagnostic for the hair-colour word of the hairstyle pipeline: what does each metric say about each photo?

  python scripts/hair_colour_check.py PHOTO_DIR [--outputs OUT_DIR] [--limit N]

Per photo prints hair L p25/p50, face-skin L, the effective L, the OLD word (median L, old bins) and the NEW word (exposure-corrected
p25). With --outputs, photos in OUT_DIR with the same file stem are the generated results: it also prints the hair drift
(output minus input hair L, each relative to its own skin L), which shows how much lighter the model paints the hair.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vto_spike.recolor2 import colour_word, measure_hair_lightness  # noqa: E402
from vto_spike.segment import HAIR_MODEL, HairSegmenter, SelfieParser  # noqa: E402

OLD_BINS = ((12, "black"), (22, "dark brown"), (35, "brown"), (50, "light brown"), (101, "blonde"))


def load(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", type=Path)
    ap.add_argument("--outputs", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    seg, parser = HairSegmenter(HAIR_MODEL), SelfieParser()
    files = sorted(p for p in args.photos.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        files = files[: args.limit]
    print(f"{'photo':36s} {'p25':>5s} {'p50':>5s} {'skin':>5s} {'eff':>5s}  {'old':12s} {'new':12s}  drift")
    counts = {"old": {}, "new": {}}
    for f in files:
        rgb = load(f)
        hair, face = parser(rgb)
        m = measure_hair_lightness(rgb, seg(rgb), face)
        if m is None:
            continue
        old = next(w for hi, w in OLD_BINS if m["p50"] < hi)
        new = colour_word(m)
        counts["old"][old] = counts["old"].get(old, 0) + 1
        counts["new"][new] = counts["new"].get(new, 0) + 1
        drift = ""
        if args.outputs:
            outs = list(args.outputs.glob(f"{f.stem}*"))
            if outs:
                o = load(outs[0])
                mo = measure_hair_lightness(o, seg(o), parser(o)[1])
                if mo and m["skin"] and mo["skin"]:
                    drift = f"{(mo['p25'] / mo['skin']) * 62 - (m['p25'] / m['skin']) * 62:+.1f}"
        skin = f"{m['skin']:.0f}" if m["skin"] else "-"
        print(f"{f.name[:36]:36s} {m['p25']:5.1f} {m['p50']:5.1f} {skin:>5s} {m['eff']:5.1f}  {old:12s} {new:12s}  {drift}")
    print("\nword counts  old:", counts["old"], "\n             new:", counts["new"])


if __name__ == "__main__":
    main()
