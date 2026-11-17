"""Run hairstyle jobs through HairFastGAN on aligned crops -> /work/outputs/hairfast/raw/<job id>.png (1024x1024, the aligned crop).

Input crops come from common/align.py (run `hs align` first). Jobs whose photo is flagged in aligned/framing.csv (head cut off by
the frame) are skipped. common/finish.py turns the raw crops into full-size images. Resumable. Appends seconds and peak VRAM
per job to timings.csv.

--sweep smooth:mixing[,...] --photos id[,id]: every setting on those photos x all styles -> /work/outputs/hairfast_sweep/
  <job>__sm<smooth>_mx<mixing>.png. For choosing HairFast's two knobs, not for the full run.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from pipeline import HairStyler


def flagged_ids(aligned: Path) -> set[str]:
    with open(aligned / "framing.csv", encoding="utf-8") as f:
        return {r["id"] for r in csv.DictReader(f) if r["flagged"] == "True"}


def load(aligned: Path, pid: str) -> np.ndarray:
    return np.asarray(Image.open(aligned / f"{pid}.png").convert("RGB"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--outputs", default="/work/outputs")
    ap.add_argument("--models", default="/work/models/hairfast")
    ap.add_argument("--limit", type=int, default=0, help="only the first N jobs (smoke test); 0 = all")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--mixing", type=float, default=0.95)
    ap.add_argument("--sweep", help="comma list of smooth:mixing, e.g. 5:0.95,3:0.9 (needs --photos)")
    ap.add_argument("--photos", help="comma list of photo ids to restrict the jobs to")
    args = ap.parse_args()

    inputs = Path(args.inputs)
    aligned = inputs / "aligned"
    bad = flagged_ids(aligned)
    jobs = json.loads((inputs / "jobs.json").read_text(encoding="utf-8"))
    if args.photos:
        keep = set(args.photos.split(","))
        jobs = [j for j in jobs if j["photo"].rsplit(".", 1)[0] in keep]
    skipped = [j["id"] for j in jobs if j["photo"].rsplit(".", 1)[0] in bad or j["reference"].rsplit(".", 1)[0] in bad]
    jobs = [j for j in jobs if j["id"] not in skipped]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} jobs, {len(skipped)} skipped (framing flagged)", flush=True)

    if args.sweep:
        if not args.photos:
            sys.exit("--sweep needs --photos")
        out = Path(args.outputs) / "hairfast_sweep"
        settings = [(int(s), float(m), f"__sm{s}_mx{m}") for s, m in (c.split(":") for c in args.sweep.split(","))]
    else:
        out = Path(args.outputs) / "hairfast" / "raw"
        settings = [(args.smooth, args.mixing, "")]
    out.mkdir(parents=True, exist_ok=True)

    timings = out / "timings.csv"
    new = not timings.exists()
    with open(timings, "a", newline="", encoding="utf-8") as tf:
        tw = csv.writer(tf)
        if new:
            tw.writerow(["job", "seconds", "peak_vram_mib"])
        for smooth, mixing, suffix in settings:
            styler = HairStyler(args.models, smooth=smooth, mixing=mixing)
            for job in jobs:
                name = job["id"] + suffix
                dst = out / f"{name}.png"
                if dst.exists():
                    continue
                face = load(aligned, job["photo"].rsplit(".", 1)[0])
                shape = load(aligned, job["reference"].rsplit(".", 1)[0])
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                res = styler.swap(face, shape)
                secs = time.time() - t0
                if res.shape[:2] != (1024, 1024):
                    res = cv2.resize(res, (1024, 1024), interpolation=cv2.INTER_AREA)
                Image.fromarray(res).save(dst)
                tw.writerow([name, f"{secs:.1f}", int(torch.cuda.max_memory_allocated() / 2**20)])
                tf.flush()
                print(f"{name}: {secs:.1f}s", flush=True)
            del styler


if __name__ == "__main__":
    main()
