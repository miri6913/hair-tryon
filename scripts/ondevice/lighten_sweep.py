"""Throwaway spike 2: cheaper ways to run the hairstyle edit (int8 weights simulated, FLUX.2-klein-4B, 6 photos x 2 styles, seed 0).

  --exp steps : 4 / 3 / 2 / 1 denoising steps (+ a hand-set 2-step schedule) at 512 and 384 px
  --exp crop  : edit only a head-sized crop (judged on its own, no paste-back) vs the whole photo
  --exp kv    : Flux2KleinKVPipeline (reference K/V cached after step 1) on the 4B weights vs the normal pipeline

Run in the vto-hair-api image on GPU 1 (live API is on GPU 0); scripts/ondevice is mounted at /app/od:
  docker run --rm --gpus device=1 -v hf-cache:/hf:ro -v in:/inputs:ro -v od:/app/od:ro -v prev:/prev:ro -v out:/out \
      vto-hair-api python /app/od/lighten_sweep.py --exp steps
Writes /out/<exp>/results.csv, sheet_<photo>__<style>.jpg and the images. S20 seconds are ESTIMATES from the block timings measured in
spike 1 (results/ondevice/README.md), blocks only, plus the CPU int8 VAE scaled by pixels.
"""
import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app/server")
sys.path.insert(0, "/app/od")

import cv2
import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image, ImageOps

from pipeline import COLOUR_OVERRIDE, COLOUR_PHRASE, MODEL, SEED, TEXT_PROMPT, fit
from quality_sweep import JOBS, drift, fake_quant, psnr, tile
from vto_spike.recolor2 import colour_word, measure_hair_lightness
from vto_spike.segment import HAIR_MODEL, HairSegmenter, SelfieParser

IN, PREV = Path("/inputs"), Path("/prev")
SIZES = [512, 384]


def s20_seconds(tok: int, steps: int, cached_steps: int = 0) -> float:
    """Estimate: 4-step blocks time on the S20 GPU fp16 + CPU int8 VAE. tok = tokens of ONE image; the model sees 2 images + 80 text tokens.
    Linear fit through the measured single/double block times at 944 and 1616 tokens; a cached step is costed at the mean of its
    query count and the full token count (rough)."""
    def step(n: float) -> float:
        return 20 * (0.240 + (n - 944) * 0.266 / 672) + 5 * (0.294 + (n - 944) * 0.119 / 672)
    full = 2 * tok + 80
    cached = (tok + 80 + full) / 2
    blocks = (steps - cached_steps) * step(full) + cached_steps * step(cached)
    return blocks + 6.6 * tok / 432


def grid(tiles: list, cols: int = 5) -> np.ndarray:
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def to_long_side(img: Image.Image, long_side: int) -> Image.Image:
    s = long_side / max(img.size)
    return img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)


def head_box(parser, img: Image.Image) -> tuple[int, int, int, int]:
    """Bounding box of hair + face skin, +12 % margin, widened to 3:4 portrait, kept inside the photo."""
    hair, face = parser(np.asarray(img))
    ys, xs = np.where((hair > 0.5) | (face > 0.5))
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    m = 0.12 * max(x1 - x0, y1 - y0)
    x0, x1, y0, y1 = x0 - m, x1 + m, y0 - m, y1 + m
    w, h = x1 - x0, y1 - y0
    if w / h > 0.75:
        h = w / 0.75
    else:
        w = h * 0.75
    w, h = min(w, img.width), min(h, img.height)
    if w / h > 0.75:
        w = h * 0.75
    else:
        h = w / 0.75
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = min(max(cx - w / 2, 0), img.width - w)
    y0 = min(max(cy - h / 2, 0), img.height - h)
    return int(x0), int(y0), int(x0 + w), int(y0 + h)


def setup():
    hair_seg, parser = HairSegmenter(HAIR_MODEL), SelfieParser()
    catalog = json.loads(Path("/app/server/catalog.json").read_text(encoding="utf-8"))
    hair_prompt = {s["id"]: s["hair_prompt"] for s in catalog["styles"]}
    origs, colours = {}, {}
    for p in JOBS:
        img = fit(ImageOps.exif_transpose(Image.open(next(IN.glob(p + ".*")))).convert("RGB"), 1536)
        origs[p] = img
        rgb = np.asarray(img)
        meas = measure_hair_lightness(rgb, hair_seg(rgb), parser(rgb)[1])
        colours[p] = COLOUR_OVERRIDE.get(colour_word(meas), colour_word(meas))
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    embeds = {}
    for p, styles in JOBS.items():
        for s in styles:
            key = (s, colours[p])
            if key not in embeds:
                with torch.no_grad():
                    enc = pipe.encode_prompt(TEXT_PROMPT.format(hair=hair_prompt[s], colour=COLOUR_PHRASE[colours[p]]), device="cuda")
                embeds[key] = (enc[0] if isinstance(enc, tuple) else enc).detach().clone()
    pipe.text_encoder.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    pipe.set_progress_bar_config(disable=True)
    for _, m in pipe.transformer.named_modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data = fake_quant(m.weight.data, "int8")   # simulated int8 weights, as the on-device plan
    return pipe, parser, origs, colours, embeds


def gen(p, photo: Image.Image, emb, steps: int = 4, sigmas=None, kv: bool = False):
    w, h = photo.size
    kw = dict(image=photo, prompt_embeds=emb, width=w // 16 * 16, height=h // 16 * 16, num_inference_steps=steps,
              generator=torch.Generator(device="cuda").manual_seed(SEED))
    if sigmas:
        kw["sigmas"] = sigmas
    if not kv:
        kw["guidance_scale"] = 1.0   # the KV pipeline has no such argument
    torch.cuda.synchronize()
    t0 = time.time()
    out = p(**kw).images[0]
    torch.cuda.synchronize()
    return out, time.time() - t0


def tok_of(img: Image.Image) -> int:
    return (img.width // 16) * (img.height // 16)


def exp_steps(pipe, parser, origs, colours, embeds, out: Path) -> list[dict]:
    variants = [("s4", 4, None), ("s3", 3, None), ("s2", 2, None), ("s1", 1, None), ("s2_sig", 2, [1.0, 0.5])]
    rows, imgs = [], {}
    gen(pipe, fit(origs[next(iter(JOBS))], 512), embeds[(JOBS[next(iter(JOBS))][0], colours[next(iter(JOBS))])])   # warm-up
    for size in SIZES:
        for p, styles in JOBS.items():
            photo = fit(origs[p], size)
            for s in styles:
                emb = embeds[(s, colours[p])]
                base = None
                for name, steps, sig in variants:
                    img, t = gen(pipe, photo, emb, steps, sig)
                    base = img if name == "s4" else base
                    imgs[(size, name, p, s)] = img
                    (out / f"{size}_{name}").mkdir(exist_ok=True)
                    img.save(out / f"{size}_{name}" / f"{p}__{s}.jpg", quality=92)
                    chk = ""
                    if name == "s4" and size == 512 and (PREV / "512_int8" / f"{p}__{s}.jpg").exists():
                        chk = round(psnr(Image.open(PREV / "512_int8" / f"{p}__{s}.jpg").convert("RGB"), img), 1)
                    rows.append(dict(size=size, variant=name, photo=p, style=s, psnr_vs_4step=round(psnr(base, img), 2) if name != "s4" else "",
                                     drift=round(drift(parser, photo, img), 2), t_3090=round(t, 2), s20_est=round(s20_seconds(tok_of(photo), steps), 1),
                                     repro_vs_spike1=chk))
                    print(rows[-1], flush=True)
    for p, styles in JOBS.items():
        for s in styles:
            tiles = [tile(origs[p], "original")]
            for size in SIZES:
                for name, _, _ in variants:
                    r = next(x for x in rows if (x["size"], x["variant"], x["photo"], x["style"]) == (size, name, p, s))
                    tiles.append(tile(imgs[(size, name, p, s)], f"{size} {name} d{r['drift']}"))
            cv2.imwrite(str(out / f"sheet_{p}__{s}.jpg"), grid(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return rows


def exp_crop(pipe, parser, origs, colours, embeds, out: Path) -> list[dict]:
    rows = []
    for p, styles in JOBS.items():
        box = head_box(parser, origs[p])
        crop = origs[p].crop(box)
        share = (box[2] - box[0]) * (box[3] - box[1]) / (origs[p].width * origs[p].height)
        print(f"{p}: head box {box}, {share:.0%} of the frame", flush=True)
        for s in styles:
            emb = embeds[(s, colours[p])]
            full_in = fit(origs[p], 512)
            full_out, t_full = gen(pipe, full_in, emb)
            tiles = [tile(origs[p], f"original (box {share:.0%})"), tile(full_out, "full photo 512 (baseline)")]
            rows.append(dict(photo=p, style=s, mode="full_512", head_share=round(share, 2), tokens=tok_of(full_in), t_3090=round(t_full, 2),
                             drift=round(drift(parser, full_in, full_out), 2), s20_est=round(s20_seconds(tok_of(full_in), 4), 1)))
            for L in (512, 384):
                cin = to_long_side(crop, L)
                cout, t = gen(pipe, cin, emb)
                (out / f"crop_{L}").mkdir(exist_ok=True)
                cout.save(out / f"crop_{L}" / f"{p}__{s}.jpg", quality=92)
                tiles.append(tile(cout, f"head crop {L}"))
                rows.append(dict(photo=p, style=s, mode=f"crop_{L}", head_share=round(share, 2), tokens=tok_of(cin), t_3090=round(t, 2),
                                 drift=round(drift(parser, cin, cout), 2), s20_est=round(s20_seconds(tok_of(cin), 4), 1)))
                print(rows[-1], flush=True)
            tiles.insert(2, tile(crop, "head crop (input)"))
            cv2.imwrite(str(out / f"sheet_{p}__{s}.jpg"), grid(tiles, 5), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return rows


def exp_kv(pipe, parser, origs, colours, embeds, out: Path) -> list[dict]:
    from diffusers import Flux2KleinKVPipeline
    kvp = Flux2KleinKVPipeline(**pipe.components)   # shares the loaded (int8-simulated) weights; sets the KV-aware attention processors
    kvp.set_progress_bar_config(disable=True)
    rows = []
    first = next(iter(JOBS))
    for pp, kvflag in ((pipe, False), (kvp, True)):   # warm-up both
        gen(pp, fit(origs[first], 512), embeds[(JOBS[first][0], colours[first])], kv=kvflag)
    for size in SIZES:
        for p, styles in JOBS.items():
            photo = fit(origs[p], size)
            for s in styles:
                emb = embeds[(s, colours[p])]
                a, ta = gen(pipe, photo, emb)
                b, tb = gen(kvp, photo, emb, kv=True)
                for nm, im in (("normal", a), ("kv", b)):
                    (out / f"{size}_{nm}").mkdir(exist_ok=True)
                    im.save(out / f"{size}_{nm}" / f"{p}__{s}.jpg", quality=92)
                rows.append(dict(size=size, photo=p, style=s, psnr_kv_vs_normal=round(psnr(a, b), 2), drift_normal=round(drift(parser, photo, a), 2),
                                 drift_kv=round(drift(parser, photo, b), 2), t_normal=round(ta, 2), t_kv=round(tb, 2),
                                 s20_est_normal=round(s20_seconds(tok_of(photo), 4), 1), s20_est_kv=round(s20_seconds(tok_of(photo), 4, 3), 1)))
                print(rows[-1], flush=True)
                if size == 512:
                    cv2.imwrite(str(out / f"sheet_{p}__{s}.jpg"),
                                np.hstack([tile(origs[p], "original"), tile(a, f"normal d{rows[-1]['drift_normal']}"),
                                           tile(b, f"kv cache d{rows[-1]['drift_kv']} psnr {rows[-1]['psnr_kv_vs_normal']}")]),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["steps", "crop", "kv"], required=True)
    a = ap.parse_args()
    out = Path("/out") / a.exp
    out.mkdir(parents=True, exist_ok=True)
    pipe, parser, origs, colours, embeds = setup()
    rows = {"steps": exp_steps, "crop": exp_crop, "kv": exp_kv}[a.exp](pipe, parser, origs, colours, embeds, out)
    with open(out / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    key = {"steps": ("size", "variant"), "crop": ("mode",), "kv": ("size",)}[a.exp]
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in key), []).append(r)
    for g, rs in groups.items():
        nums = {k: float(np.nanmean([float(x[k]) for x in rs if x[k] != ""])) for k in rs[0]
                if k not in ("photo", "style", "variant", "mode", "repro_vs_spike1") and any(x[k] != "" for x in rs)}
        print("SUMMARY", g, {k: round(v, 2) for k, v in nums.items()}, flush=True)
    parser.close()


if __name__ == "__main__":
    main()
