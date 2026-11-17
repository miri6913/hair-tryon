"""Paste each model's new hairstyle back onto the original photo so the face and background stay the original pixels.

Per output results/hairstyle/<model>/<photo>__<style>.png: resize to the photo, segment the new hair, take the union of the
old and new hair masks (the old region matters when the new style is shorter: whatever the model drew there is kept),
feather it and blend. Writes results/hairstyle/<model>/composited/<same name>.png.
  python scripts/hairstyle_composite.py [--models qwen_text,qwen_ref,stable_hair,hairfast]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from vto_spike.composite import composite  # noqa: E402
from vto_spike.recolor2 import measure_shade, recolor_hair2  # noqa: E402
from vto_spike.segment import HairSegmenter, SelfieParser  # noqa: E402

SET = ROOT / "data" / "hairstyle_set"
RES = ROOT / "results" / "hairstyle"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen_text,qwen_ref,stable_hair,hairfast")
    ap.add_argument("--keep-color", action="store_true",
                    help="recolour the new hair to the person's own measured hair shade; writes <model>_keep/composited/")
    args = ap.parse_args()

    seg = HairSegmenter()
    parser = SelfieParser()   # face-skin masks, for the revealed-skin tone match in composite()
    for model in args.models.split(","):
        src = RES / model
        if not src.is_dir():
            print(f"{model}: nothing pulled yet, skipping")
            continue
        dst = (RES / f"{model}_keep" / "composited") if args.keep_color else src / "composited"
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob("*.png"))
        for p in files:
            pid = p.stem.split("__")[0]
            orig_img = Image.open(SET / f"{pid}.jpg").convert("RGB")
            out_img = Image.open(p).convert("RGB").resize(orig_img.size, Image.LANCZOS)
            old_mask = np.asarray(Image.open(SET / "masks" / f"{pid}.png"), dtype=np.float32) / 255
            new_mask = seg(np.asarray(out_img))
            if args.keep_color:
                shade = measure_shade(np.asarray(orig_img), old_mask)
                if shade is None:
                    print(f"{p.stem}: no usable original hair, colour kept")
                else:
                    out_img = Image.fromarray(recolor_hair2(np.asarray(out_img), new_mask, shade))
            face_out = parser(np.asarray(out_img))[1]
            comp = composite(np.asarray(orig_img, dtype=np.float32), old_mask, np.asarray(out_img, dtype=np.float32), new_mask,
                             face_out)
            Image.fromarray(comp).save(dst / p.name)
        print(f"{model}: {len(files)} composited")
    seg.close()
    parser.close()


if __name__ == "__main__":
    main()
