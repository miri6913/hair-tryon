"""Phase 0 step 3: hair masks on the test set (hair-only vs multiclass model, raw vs guided-filter refined).

Outputs (results/, gitignored PNGs):
  masks/sheet_XX.png    review sheets: original | hair raw | hair refined | multiclass refined
  masks/npy/<stem>.npy  refined hair mask (hair-only model) for the recolor step
  mask_scores.csv       auto-metrics + empty score columns for the 1-5 by-eye rating
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vto_spike import viz  # noqa: E402
from vto_spike.refine import guided_refine  # noqa: E402
from vto_spike.segment import HAIR_MODEL, MULTICLASS_MODEL, HairSegmenter  # noqa: E402

PHOTO_DIRS = [ROOT / "data" / "photos_kaggle", ROOT / "data" / "photos_wiki"]
EXCLUDE_FILE = ROOT / "data" / "wiki_exclude.txt"
RESULTS = ROOT / "results"
PER_SHEET = 6
PANEL_H = 240


def metrics(mask: np.ndarray) -> dict:
    return {
        "area": float((mask > 0.5).mean()),
        "soft_edge": float(((mask > 0.1) & (mask < 0.9)).mean()),  # share of mid-confidence pixels
    }


def main() -> None:
    (RESULTS / "masks" / "npy").mkdir(parents=True, exist_ok=True)
    hair, multi = HairSegmenter(HAIR_MODEL), HairSegmenter(MULTICLASS_MODEL)
    excluded = {l.strip() for l in EXCLUDE_FILE.read_text().splitlines() if l.strip() and not l.startswith("#")}
    files = sorted((p for d in PHOTO_DIRS for p in d.glob("*.jpg") if p.name not in excluded), key=lambda p: (p.parent.name, p.name))
    rows, tiles, times = [], [], []
    for i, path in enumerate(files):
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        raw = hair(rgb)
        times.append((time.perf_counter() - t0) * 1000)
        refined = guided_refine(rgb, raw)
        m_raw = multi(rgb)
        m_ref = guided_refine(rgb, m_raw)
        np.save(RESULTS / "masks" / "npy" / f"{path.stem}.npy", refined.astype(np.float16))

        tiles.append(viz.stack([viz.row([
            viz.label(rgb, path.stem),
            viz.label(viz.overlay_mask(rgb, raw), "hair raw"),
            viz.label(viz.overlay_mask(rgb, refined), "hair refined"),
            viz.label(viz.overlay_mask(rgb, m_ref), "multiclass refined"),
        ], PANEL_H)]))
        a, b = metrics(refined), metrics(m_ref)
        rows.append({
            "file": path.name, "hair_area": f"{a['area']:.4f}", "hair_soft_edge": f"{a['soft_edge']:.4f}",
            "multi_area": f"{b['area']:.4f}", "agreement_iou": f"{_iou(refined > 0.5, m_ref > 0.5):.3f}",
            "auto_flag": _flag(a["area"]), "score_hair": "", "score_multi": "", "notes": "",
        })
        if len(tiles) == PER_SHEET or i == len(files) - 1:
            cv2.imwrite(str(RESULTS / "masks" / f"sheet_{i // PER_SHEET:02d}.png"),
                        cv2.cvtColor(viz.stack(tiles), cv2.COLOR_RGB2BGR))
            tiles = []

    with (RESULTS / "mask_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    flagged = sum(r["auto_flag"] != "" for r in rows)
    print(f"{len(rows)} photos, {flagged} auto-flagged, hair model CPU median {np.median(times):.0f} ms/img (1024px, incl. resize)")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def _flag(area: float) -> str:
    if area < 0.02:
        return "tiny/no hair mask"
    if area > 0.45:
        return "mask suspiciously large"
    return ""


if __name__ == "__main__":
    main()
