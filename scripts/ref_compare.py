"""Compare candidate reference photos (served by a test API on :8091) with the live catalogue's references, on the same photos.

Runs the requests that are missing (resumable), then prints an identity-drift score per output and writes one sheet per style:
  reference(candidate) | reference(baseline) | original | baseline output | candidate output

Identity drift = mean absolute RGB difference between the original (resized to the output) and the output over the original's
face-skin region (selfie-multiclass mask, eroded 10 px so hair and edges are out). Lower = the face is closer to the photo. It also
counts overall tone shifts, so it is a relative score between sources, not an absolute identity measure.

  ssh -N -L 8091:127.0.0.1:8091 <host>         (the test API: scripts\\hs api-test-start -g 1 --refs-dir refs_nano)
  set HAIRAPI_KEY=...   python scripts/ref_compare.py --source nano
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from vto_spike.segment import SelfieParser  # noqa: E402

SET = ROOT / "data" / "hairstyle_set"
REFS = ROOT / "data" / "hairapi_refs"
BASE = ROOT / "results" / "api" / "val"          # baseline outputs: the live catalogue's references, raw mode, same seed
PAIRS = {  # gender-matched test photos per style gender
    "female": ["kaggle_train_18-20_6", "kaggle_test_31-40_27"],
    "male": ["photo_a", "photo_d"],
}
TILE = (240, 320)


def drift(parser: SelfieParser, orig: Image.Image, out: Image.Image) -> float:
    face = parser(np.asarray(orig))[1] > 0.5
    face = cv2.erode(face.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    face = cv2.resize(face, out.size, interpolation=cv2.INTER_NEAREST) > 0
    o = np.asarray(orig.resize(out.size, Image.LANCZOS), np.float32)
    x = np.asarray(out, np.float32)
    return float(np.abs(o - x).mean(axis=2)[face].mean()) if face.sum() > 100 else float("nan")


def tile(path_or_img, label: str) -> np.ndarray:
    im = cv2.imread(str(path_or_img)) if not isinstance(path_or_img, np.ndarray) else path_or_img
    im = cv2.resize(im, TILE)
    cv2.putText(im, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="candidate set, e.g. nano (data/hairapi_refs/incoming/<source>)")
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--styles", help="no candidate references (e.g. a prompt-only test API): comma list of style ids of the "
                                     "test API's catalogue; columns become original | baseline output | test output")
    args = ap.parse_args()
    key = os.environ.get("HAIRAPI_KEY", "")
    cand_dir = REFS / "incoming" / args.source
    out_dir = ROOT / "results" / "api" / f"{args.source}_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    parser = SelfieParser()

    rows_by_style: dict[str, list] = {}
    table = []
    if args.styles:
        jobs = [(s, s.split("__")[0], None) for s in args.styles.split(",")]
    else:   # candidate references: the catalogue id of the test API is the file stem
        jobs = [(ref.stem, ref.stem.split("__")[0], ref) for ref in sorted(cand_dir.glob("*.jpg"))]
    for style_id, base_id, ref in jobs:
        gender = "female" if base_id.startswith("f_") else "male"
        for photo in PAIRS[gender]:
            dst = out_dir / f"{photo}__{style_id}.jpg"
            if not dst.exists():
                src = SET / f"{photo}.jpg"
                r = requests.post(f"{args.url}/v1/hairstyle", headers={"X-API-Key": key}, data={"style_id": style_id},
                                  files={"photo": (src.name, src.read_bytes(), "image/jpeg")}, timeout=300)
                if r.status_code != 200:
                    sys.exit(f"{photo} {style_id}: {r.status_code} {r.text[:200]}")
                dst.write_bytes(r.content)
            base = BASE / f"{photo}__{base_id}.jpg"
            orig = Image.open(SET / f"{photo}.jpg").convert("RGB")
            d_cand = drift(parser, orig, Image.open(dst).convert("RGB"))
            d_base = drift(parser, orig, Image.open(base).convert("RGB")) if base.exists() else float("nan")
            table.append((style_id, photo, d_base, d_cand))
            ref_tiles = [tile(ref, f"candidate ref {style_id}"), tile(REFS / f"{base_id}.jpg", "baseline ref (synthetic)")] if ref else []
            row = np.hstack(ref_tiles + [tile(SET / f"{photo}.jpg", f"original {photo[:16]}"),
                                         tile(base, f"baseline (ref) {base_id[2:]} drift {d_base:.1f}"),
                                         tile(dst, f"{args.source} {style_id[2:]} drift {d_cand:.1f}")])
            rows_by_style.setdefault(style_id, []).append(row)
    for style_id, rows in rows_by_style.items():
        out = out_dir / f"compare_{style_id}.jpg"
        cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("wrote", out)
    print(f"\n{'style':24s} {'photo':24s} {'baseline drift':>15s} {args.source + ' drift':>13s}")
    for s, p, b, c in table:
        print(f"{s:24s} {p:24s} {b:15.1f} {c:13.1f}")
    print("mean  baseline %.1f  %s %.1f   (lower = face closer to the photo)" % (
        np.nanmean([t[2] for t in table]), args.source, np.nanmean([t[3] for t in table])))
    parser.close()


if __name__ == "__main__":
    main()
