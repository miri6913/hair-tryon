"""Throwaway test: does cutting the 512-token prompt embedding down to its real tokens change the hairstyle result?

Runs inside the vto-hair-api image (needs the FLUX weights in /hf and the photos in /inputs):
  docker run --rm --gpus device=1 -v hf-cache:/hf:ro -v inputs:/inputs:ro -v trim_embed_test.py:/app/trim_test.py:ro -v out:/out \
      vto-hair-api python /app/trim_test.py
Writes /out/results.csv and /out/sheet_<size>_<photo>__<style>.jpg (original | full 512 | trimmed).
"""
import csv
import sys

sys.path.insert(0, "/app/server")
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image, ImageOps

from pipeline import COLOUR_OVERRIDE, COLOUR_PHRASE, MODEL, SEED, STEPS, TEXT_PROMPT, fit
from vto_spike.recolor2 import colour_word, measure_hair_lightness
from vto_spike.segment import HAIR_MODEL, HairSegmenter, SelfieParser
import json

IN, OUT = Path("/inputs"), Path("/out")
PHOTOS = ["kaggle_train_18-20_6", "kaggle_test_31-40_27", "photo_a", "photo_d"]
STYLES = ["m_short_buzz", "f_long_volume", "f_short_french_bob", "m_medium_modern_quiff"]
SIZES = [1024, 256]
TILE = (240, 320)


def drift(parser, orig: Image.Image, out: Image.Image) -> float:
    face = parser(np.asarray(orig))[1] > 0.5
    face = cv2.erode(face.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    face = cv2.resize(face, out.size, interpolation=cv2.INTER_NEAREST) > 0
    o = np.asarray(orig.resize(out.size, Image.LANCZOS), np.float32)
    x = np.asarray(out, np.float32)
    return float(np.abs(o - x).mean(axis=2)[face].mean()) if face.sum() > 100 else float("nan")


def psnr(a: Image.Image, b: Image.Image) -> float:
    d = np.asarray(a, np.float32) - np.asarray(b, np.float32)
    mse = float((d ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def tile(img: Image.Image, label: str) -> np.ndarray:
    im = cv2.cvtColor(np.asarray(img.convert("RGB").resize(TILE)), cv2.COLOR_RGB2BGR)
    cv2.putText(im, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
    return im


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(Path("/app/server/catalog.json").read_text(encoding="utf-8"))
    hair_prompt = {s["id"]: s["hair_prompt"] for s in catalog["styles"]}
    hair_seg, parser = HairSegmenter(HAIR_MODEL), SelfieParser()
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    tok = pipe.tokenizer

    def encode(prompt: str):
        with torch.no_grad():
            enc = pipe.encode_prompt(prompt, device="cuda")
        full = (enc[0] if isinstance(enc, tuple) else enc).detach().clone()
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        mask = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)["attention_mask"][0].bool()
        assert mask.numel() == full.shape[1], (mask.numel(), full.shape)
        idx = mask.nonzero().flatten()
        contiguous_start = bool(idx[0] == 0)
        return full, full[:, mask.to(full.device)].clone(), int(mask.sum()), ("right" if contiguous_start else "left")

    def gen(photo: Image.Image, emb: torch.Tensor) -> tuple[Image.Image, float]:
        w, h = photo.size
        torch.cuda.synchronize()
        t0 = time.time()
        out = pipe(image=photo, prompt_embeds=emb, width=w // 16 * 16, height=h // 16 * 16, guidance_scale=1.0,
                   num_inference_steps=STEPS, generator=torch.Generator(device="cuda").manual_seed(SEED)).images[0]
        torch.cuda.synchronize()
        return out, time.time() - t0

    pipe.set_progress_bar_config(disable=True)
    origs, colours = {}, {}
    for p in PHOTOS:
        img = ImageOps.exif_transpose(Image.open(IN / f"{p}.jpg")).convert("RGB")
        img = fit(img, 1536)
        origs[p] = img
        rgb = np.asarray(img)
        meas = measure_hair_lightness(rgb, hair_seg(rgb), parser(rgb)[1])
        colours[p] = COLOUR_OVERRIDE.get(colour_word(meas), colour_word(meas))
        print(p, "colour", colours[p], flush=True)

    embeds = {}
    for p in PHOTOS:
        for s in STYLES:
            key = (s, colours[p])
            if key not in embeds:
                embeds[key] = encode(TEXT_PROMPT.format(hair=hair_prompt[s], colour=COLOUR_PHRASE[colours[p]]))
                print("encoded", key, "tokens", embeds[key][2], "padding", embeds[key][3], flush=True)

    rows = []
    for size in SIZES:
        w0 = fit(origs[PHOTOS[0]], size)
        gen(w0, embeds[(STYLES[0], colours[PHOTOS[0]])][0])   # warm-up, not timed
        for p in PHOTOS:
            photo = fit(origs[p], size)
            for s in STYLES:
                full, trim, n, side = embeds[(s, colours[p])]
                o_full, t_full = gen(photo, full)
                o_trim, t_trim = gen(photo, trim)
                ps = psnr(o_full, o_trim)
                d_full, d_trim = drift(parser, photo, o_full), drift(parser, photo, o_trim)
                rows.append(dict(size=size, photo=p, style=s, colour=colours[p], n_tokens=n, padding=side, psnr=round(ps, 2),
                                 drift_full=round(d_full, 2), drift_trim=round(d_trim, 2),
                                 t_full=round(t_full, 3), t_trim=round(t_trim, 3)))
                sheet = np.hstack([tile(photo, f"original {p[:14]}"), tile(o_full, f"full 512 drift {d_full:.1f}"),
                                   tile(o_trim, f"trim {n} drift {d_trim:.1f} psnr {ps:.1f}")])
                cv2.imwrite(str(OUT / f"sheet_{size}_{p}__{s}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
                print(rows[-1], flush=True)

    with open(OUT / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for size in SIZES:
        r = [x for x in rows if x["size"] == size]
        m = lambda k: float(np.mean([x[k] for x in r]))   # noqa: E731
        print(f"SUMMARY {size}px: psnr {m('psnr'):.1f} (min {min(x['psnr'] for x in r):.1f})  drift full {m('drift_full'):.2f} "
              f"trim {m('drift_trim'):.2f}  time full {m('t_full'):.2f}s trim {m('t_trim'):.2f}s", flush=True)
    parser.close()


if __name__ == "__main__":
    main()
