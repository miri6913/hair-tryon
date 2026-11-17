"""Run hairstyle jobs through Stable-Hair (photo + reference hairstyle photo) -> /work/outputs/stable_hair/<job id>.png

The model squashes its input to 512x512, so each photo is edge-padded to a square first and the result is cropped back to the
photo's aspect ratio (output is 512 px on the long side; hairstyle_composite.py pastes it onto the full-resolution photo).
Resumable. Appends seconds and peak VRAM per job to timings.csv.
"""
import argparse
import csv
import json
import os
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO = Path("/opt/Stable-Hair")
SIZE = 512


def pad_square(path: Path, dst: Path) -> tuple[int, int, int, int]:
    """Edge-pad to a square, save to dst; returns (top, bottom, left, right) padding."""
    img = np.asarray(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    padded = cv2.copyMakeBorder(img, top, side - h - top, left, side - w - left, cv2.BORDER_REPLICATE)
    Image.fromarray(padded).save(dst, quality=95)
    return top, side - h - top, left, side - w - left


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--outputs", default="/work/outputs")
    ap.add_argument("--models", default="/work/models")
    ap.add_argument("--limit", type=int, default=0, help="only the first N jobs (smoke test); 0 = all")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    inputs = Path(args.inputs)
    out = Path(args.outputs) / "stable_hair"
    tmp = out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((inputs / "jobs.json").read_text(encoding="utf-8"))
    if args.limit:
        jobs = jobs[: args.limit]

    # config: point the repo's relative / retired paths at the mounted weights
    cfg = OmegaConf.load(REPO / "configs" / "hair_transfer.yaml")
    cfg.pretrained_model_path = f"{args.models}/sd15"
    cfg.pretrained_folder = f"{args.models}/stable_hair/stage2"
    cfg.bald_converter_path = f"{args.models}/stable_hair/stage1/pytorch_model.bin"
    cfg_path = tmp / "hair_transfer.yaml"
    OmegaConf.save(cfg, cfg_path)

    # infer_full.py imports gradio only for its demo; the pinned old gradio clashes with the rest of the stack, so stub it
    sys.modules.setdefault("gradio", types.ModuleType("gradio"))
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    from infer_full import StableHair  # noqa: E402

    model = StableHair(config=str(cfg_path), weight_dtype=torch.float32)  # fp32 as in the repo's own __main__

    timings = out / "timings.csv"
    new = not timings.exists()
    with open(timings, "a", newline="", encoding="utf-8") as tf:
        tw = csv.writer(tf)
        if new:
            tw.writerow(["job", "seconds", "peak_vram_mib"])
        for job in jobs:
            dst = out / f"{job['id']}.png"
            if dst.exists():
                continue
            src = tmp / "src.jpg"
            top, bottom, left, right = pad_square(inputs / job["photo"], src)
            ref = tmp / "ref.jpg"
            pad_square(inputs / job["reference"], ref)

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            _, sample, _, _ = model.Hair_Transfer(
                source_image=str(src), reference_image=str(ref), random_seed=args.seed, step=args.steps,
                guidance_scale=1.5, scale=1, controlnet_conditioning_scale=1, size=SIZE,
            )
            secs = time.time() - t0

            res = (np.clip(sample, 0, 1) * 255).astype(np.uint8)  # SIZE x SIZE, still the padded square
            # the padding was measured on the full-size square; scale it down to the SIZE square and crop it off
            f = res.shape[0] / max(Image.open(inputs / job["photo"]).size)
            res = res[round(top * f): res.shape[0] - round(bottom * f), round(left * f): res.shape[1] - round(right * f)]
            Image.fromarray(res).save(dst)
            tw.writerow([job["id"], f"{secs:.1f}", int(torch.cuda.max_memory_allocated() / 2**20)])
            tf.flush()
            print(f"{job['id']}: {secs:.1f}s", flush=True)


if __name__ == "__main__":
    main()
