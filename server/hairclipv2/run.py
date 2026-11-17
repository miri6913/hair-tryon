"""Run hairstyle jobs through HairCLIPv2 (hairstyle from a reference photo) on aligned crops -> /work/outputs/hairclipv2/raw/<job>.png

Follows the repo's hairclip_v2_demo.ipynb: invert the photo (cached), bald proxy, reference proxy (hairstyle), feature blending.
--color keep additionally runs the colour proxy with the photo itself as the colour reference (keeps the person's hair colour);
none (default) stops after the hairstyle edit. Inputs are the FFHQ crops from common/align.py (`hs align`); photos flagged in
aligned/framing.csv are skipped. Optimisation-based, so seconds-to-minutes per job. Resumable; timings.csv has seconds and peak VRAM.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO = Path("/opt/HairCLIPv2")
WORK = Path("/work/hairclipv2")


def to_uint8(t: torch.Tensor) -> np.ndarray:
    return ((t.detach().float().cpu().squeeze(0).clamp(-1, 1) + 1) / 2 * 255).round().permute(1, 2, 0).numpy().astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--outputs", default="/work/outputs")
    ap.add_argument("--models", default="/work/models/hairclipv2")
    ap.add_argument("--limit", type=int, default=0, help="only the first N jobs (smoke test); 0 = all")
    ap.add_argument("--photos", help="comma list of photo ids to restrict the jobs to")
    ap.add_argument("--color", choices=["none", "keep"], default="none")
    ap.add_argument("--bald", action="store_true", help="only write bald-proxy faces for the photos of the jobs -> outputs/hairclipv2/bald/")
    args = ap.parse_args()

    inputs = Path(args.inputs)
    aligned = inputs / "aligned"
    with open(aligned / "framing.csv", encoding="utf-8") as f:
        bad = {r["id"] for r in csv.DictReader(f) if r["flagged"] == "True"}
    jobs = json.loads((inputs / "jobs.json").read_text(encoding="utf-8"))
    if args.photos:
        keep = set(args.photos.split(","))
        jobs = [j for j in jobs if j["photo"].rsplit(".", 1)[0] in keep]
    jobs = [j for j in jobs if j["photo"].rsplit(".", 1)[0] not in bad and j["reference"].rsplit(".", 1)[0] not in bad]
    if args.limit:
        jobs = jobs[: args.limit]
    out = Path(args.outputs) / "hairclipv2" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs", flush=True)

    # the repo reads ./pretrained_models and uses folders of .jpg images; the aligned crops are copied there as high-quality jpgs
    link = REPO / "pretrained_models"
    if not link.is_symlink():
        if link.exists():   # the repo ships a placeholder folder of that name
            shutil.rmtree(link)
        link.symlink_to(Path(args.models) / "pretrained_models")
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    for d in ("src_img", "src_F", "ref_img", "ref_latent"):
        (WORK / d).mkdir(parents=True, exist_ok=True)

    from scripts.bald_proxy import BaldProxy  # noqa: E402
    from scripts.color_proxy import ColorProxy  # noqa: E402
    from scripts.Embedding import Embedding  # noqa: E402
    from scripts.feature_blending import hairstyle_feature_blending  # noqa: E402
    from scripts.ref_proxy import RefProxy  # noqa: E402
    from utils.model_utils import load_base_models  # noqa: E402
    from utils.options import Options  # noqa: E402

    opts = Options().parse(jupyter=True)
    opts.src_img_dir, opts.src_latent_dir = str(WORK / "src_img"), str(WORK / "src_F")
    opts.ref_img_dir, opts.ref_latent_dir = str(WORK / "ref_img"), str(WORK / "ref_latent")
    g_ema, mean_latent_code, seg = load_base_models(opts)
    ii2s = Embedding(opts, g_ema, mean_latent_code[0, 0])
    bald_proxy = BaldProxy(g_ema, opts.bald_path)
    ref_proxy = RefProxy(opts, g_ema, seg, ii2s)
    color_proxy = ColorProxy(opts, g_ema, seg)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])

    def stage(pid: str, dst_dir: str) -> str:
        dst = Path(dst_dir) / f"{pid}.jpg"
        if not dst.exists():
            Image.open(aligned / f"{pid}.png").convert("RGB").save(dst, quality=98)
        return dst.name

    if args.bald:
        # bald proxy (HairMapper): the person's face with the hair removed, StyleGAN-synthesised in the same aligned crop frame.
        # common/finish.py --skin-src bald takes the exposed forehead/scalp/ears from it instead of from the hairstyle GAN's output
        bald_dir = Path(args.outputs) / "hairclipv2" / "bald"
        bald_dir.mkdir(parents=True, exist_ok=True)
        for pid in sorted({j["photo"].rsplit(".", 1)[0] for j in jobs}):
            dst = bald_dir / f"{pid}.png"
            if dst.exists():
                continue
            t0 = time.time()
            src_path = Path(opts.src_img_dir) / stage(pid, opts.src_img_dir)
            lat_path = Path(opts.src_latent_dir) / f"{pid}.npz"
            if not lat_path.exists():
                w_plus, feat = ii2s.invert_image_in_FS(image_path=str(src_path))
                np.savez(lat_path, latent_in=w_plus.detach().cpu().numpy(), latent_F=feat.detach().cpu().numpy())
            src_latent = torch.from_numpy(np.load(lat_path)["latent_in"]).cuda()
            latent_bald, _ = bald_proxy(src_latent)
            with torch.no_grad():
                img, _ = g_ema([latent_bald], input_is_latent=True, return_latents=False)
            Image.fromarray(to_uint8(img)).save(dst)
            print(f"bald {pid}: {time.time() - t0:.1f}s", flush=True)
        return

    timings = out / "timings.csv"
    new = not timings.exists()
    with open(timings, "a", newline="", encoding="utf-8") as tfile:
        tw = csv.writer(tfile)
        if new:
            tw.writerow(["job", "seconds", "peak_vram_mib"])
        for job in jobs:
            dst = out / f"{job['id']}.png"
            if dst.exists():
                continue
            pid, rid = job["photo"].rsplit(".", 1)[0], job["reference"].rsplit(".", 1)[0]
            src_name, ref_name = stage(pid, opts.src_img_dir), stage(rid, opts.ref_img_dir)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()

            lat_path = Path(opts.src_latent_dir) / f"{pid}.npz"
            src_path = Path(opts.src_img_dir) / src_name
            if not lat_path.exists():
                w_plus, feat = ii2s.invert_image_in_FS(image_path=str(src_path))
                np.savez(lat_path, latent_in=w_plus.detach().cpu().numpy(), latent_F=feat.detach().cpu().numpy())
            z = np.load(lat_path)
            src_latent = torch.from_numpy(z["latent_in"]).cuda()
            src_feature = torch.from_numpy(z["latent_F"]).cuda()
            src_image = tf(Image.open(src_path).convert("RGB")).unsqueeze(0).cuda()
            input_mask = torch.argmax(seg(src_image)[1], dim=1).long().clone().detach()

            latent_bald, _ = bald_proxy(src_latent)
            latent_global, _ = ref_proxy(ref_name, src_image, painted_mask=None)
            src_feature, edited = hairstyle_feature_blending(
                g_ema, seg, src_latent, src_feature, input_mask, latent_bald=latent_bald, latent_global=latent_global,
                latent_local=None, local_blending_mask=None)
            result = edited
            if args.color == "keep":
                _, finals = color_proxy(src_name, edited, src_latent, src_feature)  # the photo itself as colour reference
                result = finals[-1]

            secs = time.time() - t0
            Image.fromarray(to_uint8(result)).save(dst)
            tw.writerow([job["id"], f"{secs:.1f}", int(torch.cuda.max_memory_allocated() / 2**20)])
            tfile.flush()
            print(f"{job['id']}: {secs:.1f}s", flush=True)


if __name__ == "__main__":
    main()
