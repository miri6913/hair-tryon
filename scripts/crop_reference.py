"""Crop candidate reference photos to the framing the hairstyle model works with.

Why: the model copies the reference's framing, clothes and even its face when the reference differs a lot from the user's photo.
Nano Banana portraits with a small head, lots of headroom and visible shoulders made the result a different man in a grey T-shirt
(identity drift 53 vs 16 for the same photo and style, tight crop). The synthetic references were tight head crops: the head filling
about 65% of the frame width, a little headroom, 3:4. This script gives any candidate that framing.

  python scripts/crop_reference.py nano        # data/hairapi_refs/incoming/nano/<id>.jpg -> <id>__crop.jpg (768x1024)

The head box is the union of the selfie-multiclass hair and face-skin masks. Files already ending in __crop are skipped; outputs are
overwritten. Check the crops by eye: a hat, a hand in the hair or a second person will confuse the head box.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from vto_spike.segment import SelfieParser  # noqa: E402

HEAD_FRAC = 0.65      # head width / crop width
TOP_FRAC = 0.07       # space above the hair / crop height
OUT_SIZE = (768, 1024)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = ROOT / "data" / "hairapi_refs" / "incoming" / sys.argv[1]
    parser = SelfieParser()
    for src in sorted(folder.glob("*.jpg")):
        if src.stem.endswith("__crop"):
            continue
        im = Image.open(src).convert("RGB")
        hair, face = parser(np.asarray(im))
        ys, xs = np.nonzero((hair > 0.5) | (face > 0.5))
        if len(xs) < 1000:
            print(f"{src.name}: no head found, skipped")
            continue
        x0, x1, y0 = xs.min(), xs.max(), ys.min()
        cw = (x1 - x0) / HEAD_FRAC
        ch = cw * 4 / 3
        cx = (x0 + x1) / 2
        top = y0 - TOP_FRAC * ch
        box = (round(cx - cw / 2), round(top), round(cx + cw / 2), round(top + ch))
        if box[0] < 0 or box[1] < 0 or box[2] > im.width or box[3] > im.height:
            print(f"{src.name}: crop {box} leaves the image {im.size} (head too close to the edge); skipped")
            continue
        im.crop(box).resize(OUT_SIZE, Image.LANCZOS).save(folder / f"{src.stem}__crop.jpg", quality=95)
        print(f"{src.name} -> {src.stem}__crop.jpg (crop {box})")
    parser.close()


if __name__ == "__main__":
    main()
