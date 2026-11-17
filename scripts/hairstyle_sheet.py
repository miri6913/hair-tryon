"""Review sheets for the hairstyle comparison: one image per style, a row per photo, columns original | each model.

Uses the composited outputs (hairstyle_composite.py) unless --raw. Also creates results/hairstyle/scores.csv, one row per
job x model with seconds and peak VRAM from the server's timings.csv and empty columns to fill in by eye (1-5).
  python scripts/hairstyle_sheet.py [--raw]
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SET = ROOT / "data" / "hairstyle_set"
RES = ROOT / "results" / "hairstyle"
MODELS = ["qwen_text", "qwen_ref", "stable_hair", "hairfast", "hairclipv2",
          "flux2_4b_text", "flux2_4b_ref", "flux2_9b_text", "flux2_9b_ref", "flux2_4b_ref_keep"]
TILE_H = 288


def tile(path: Path, label: str = "", th: int = TILE_H) -> np.ndarray:
    img = cv2.imread(str(path)) if path.exists() else None
    if img is None:
        img = np.full((th, th * 3 // 4, 3), 40, np.uint8)
        cv2.putText(img, "missing", (8, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    else:
        img = cv2.resize(img, (round(img.shape[1] * th / img.shape[0]), th))
    if label:
        cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def pad_to(img: np.ndarray, w: int) -> np.ndarray:
    if img.shape[1] >= w:
        return img[:, :w]
    return cv2.copyMakeBorder(img, 0, 0, 0, w - img.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="use the raw model outputs instead of the composited ones")
    ap.add_argument("--models", help="comma list of models to show (default: every pulled one)")
    ap.add_argument("--photos", help="comma list of photo ids (default: all)")
    ap.add_argument("--styles", help="comma list of styles (default: all)")
    ap.add_argument("--tile-h", type=int, default=TILE_H, help="tile height in px")
    ap.add_argument("--name", default="sheet", help="output file prefix")
    args = ap.parse_args()
    th = args.tile_h

    jobs = json.loads((SET / "jobs.json").read_text(encoding="utf-8"))
    models = [m for m in MODELS if (RES / m).is_dir() and (not args.models or m in args.models.split(","))]
    if not models:
        raise SystemExit("no outputs in results/hairstyle yet (run hairstyle_sync.ps1 pull)")

    def sub_for(m: str) -> str:   # GAN models: final/ from common/finish.py; the others: composited/ from hairstyle_composite.py
        if args.raw:
            return ""
        return "final" if (RES / m / "final").is_dir() else "composited"

    if args.photos:
        jobs = [j for j in jobs if j["photo"].rsplit(".", 1)[0] in args.photos.split(",")]
    if args.styles:
        jobs = [j for j in jobs if j["style"] in args.styles.split(",")]
    by_style: dict[str, list[dict]] = {}
    for j in jobs:
        by_style.setdefault(j["style"], []).append(j)
    for style, js in by_style.items():
        rows = []
        for j in js:
            pid = j["id"].split("__")[0]
            cells = [tile(SET / j["photo"], pid[-18:], th)]
            cells += [tile(RES / m / sub_for(m) / f"{j['id']}.png", m, th) for m in models]
            rows.append(cells)
        col_w = [max(r[c].shape[1] for r in rows) for c in range(len(rows[0]))]
        sheet = np.vstack([np.hstack([pad_to(cell, col_w[c]) for c, cell in enumerate(r)]) for r in rows])
        out = RES / f"{args.name}_{style}{'_raw' if args.raw else ''}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"wrote {out} ({len(rows)} rows)")

    scores = RES / "scores.csv"
    if not scores.exists():
        timing = {}
        for m in models:
            t = RES / m / "timings.csv"
            if t.exists():
                with open(t, encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        timing[(m, r["job"])] = r
        with open(scores, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["job", "model", "seconds", "peak_vram_mib", "realism", "style_match", "identity"])
            for j in jobs:
                for m in models:
                    t = timing.get((m, j["id"]), {})
                    w.writerow([j["id"], m, t.get("seconds", ""), t.get("peak_vram_mib", ""), "", "", ""])
        print(f"wrote {scores} (fill realism / style_match / identity by eye, 1-5)")


if __name__ == "__main__":
    main()
