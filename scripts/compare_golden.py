"""Golden test: the on-device GLSL recolor vs. the Python reference `recolor_hair` on the same photo and mask.

Inputs come from the app's still-image mode (scripts/run_golden.ps1 collects them):
  --photo   the source photo that was pushed to the device
  --out     still_out.png   what the GLSL pipeline produced (whole photo, no screen crop)
  --mask    still_mask.f32  the smoothed hair probability the shader used (512x512 float32, top row first)

The device run uses Refine off, so the mask the shader sees is exactly: bilinear upsample of --mask, then
mask contrast. The reference gets the same upsampled mask. Everything left is shader-vs-reference approximation
(half-res separable blur, smoothstep edge guard, live mean-base estimate, fp16 intermediates, 8-bit quantisation).

Error is CIE76 delta-E in Lab. Exit code 1 if a tolerance is exceeded.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vto_spike.recolor2 import SHADES, recolor_hair2  # noqa: E402

# Must match app defaults: Palette.swatches[1], TryOnScreen intensity.
DEFAULT_SHADE = "honey"
INTENSITY = 0.85
MASK_SIZE = 512


def lab(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def stats(de: np.ndarray, sel: np.ndarray) -> str:
    v = de[sel]
    if v.size == 0:
        return "n=0"
    return f"n={v.size:7d} mean={v.mean():5.2f} p95={np.percentile(v, 95):5.2f} p99={np.percentile(v, 99):5.2f} max={v.max():5.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mask", type=Path, required=True)
    ap.add_argument("--shade", default=DEFAULT_SHADE, choices=sorted(SHADES))
    ap.add_argument("--save", type=Path, help="write a side-by-side PNG: device | reference | delta-E x8")
    ap.add_argument("--tol-hair-mean", type=float, default=1.0)
    ap.add_argument("--tol-hair-p99", type=float, default=6.0)
    ap.add_argument("--tol-bg-max", type=float, default=1.5)
    args = ap.parse_args()

    device = read_rgb(args.out)
    h, w = device.shape[:2]
    photo = read_rgb(args.photo)
    if photo.shape[:2] != (h, w):
        # the app downscales to <= 1280 px on the long side; plain bilinear is the closest to its GL sampling
        print(f"note: resizing photo {photo.shape[1]}x{photo.shape[0]} -> {w}x{h}")
        photo = cv2.resize(photo, (w, h), interpolation=cv2.INTER_LINEAR)

    raw = np.fromfile(args.mask, dtype="<f4")
    if raw.size != MASK_SIZE * MASK_SIZE:
        sys.exit(f"mask has {raw.size} floats, expected {MASK_SIZE * MASK_SIZE}")
    mask = cv2.resize(raw.reshape(MASK_SIZE, MASK_SIZE), (w, h), interpolation=cv2.INTER_LINEAR)

    ref = recolor_hair2(photo, mask, SHADES[args.shade], intensity=INTENSITY)

    de = np.linalg.norm(lab(device) - lab(ref), axis=-1)
    m = np.clip(mask, 0, 1)
    hair, edge, bg = m > 0.5, (m > 0.05) & (m <= 0.5), m <= 0.05
    # pixels the recolor should not touch at all: the device output must equal the camera frame
    de_untouched = np.linalg.norm(lab(device) - lab(photo), axis=-1)

    print(f"{args.out.name}: {w}x{h}, hair px {int(hair.sum())} ({hair.mean() * 100:.1f}%)")
    print("  device vs reference, hair  (m>0.5)   ", stats(de, hair))
    print("  device vs reference, edge  (0.05-0.5)", stats(de, edge))
    print("  device vs reference, background      ", stats(de, bg))
    print("  device vs ORIGINAL,  background      ", stats(de_untouched, bg))
    print(f"  mean signed dL (device-ref) in hair: {(lab(device)[..., 0] - lab(ref)[..., 0])[hair].mean():+.2f}")

    if args.save:
        vis = np.clip(de * 8.0, 0, 255).astype(np.uint8)
        panel = np.concatenate([device, ref, np.dstack([vis] * 3)], axis=1)
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print("  wrote", args.save)

    failures = []
    if hair.any():
        if de[hair].mean() > args.tol_hair_mean:
            failures.append(f"hair mean dE {de[hair].mean():.2f} > {args.tol_hair_mean}")
        if np.percentile(de[hair], 99) > args.tol_hair_p99:
            failures.append(f"hair p99 dE {np.percentile(de[hair], 99):.2f} > {args.tol_hair_p99}")
    else:
        failures.append("no hair pixels in the mask: nothing was tested")
    if bg.any() and de_untouched[bg].max() > args.tol_bg_max:
        failures.append(f"background changed: max dE vs original {de_untouched[bg].max():.2f} > {args.tol_bg_max}")
    for f in failures:
        print("FAIL:", f)
    print("PASS" if not failures else "FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
