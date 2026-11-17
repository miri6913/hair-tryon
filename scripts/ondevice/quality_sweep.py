"""Throwaway spike: how much quality does the hairstyle edit lose at a smaller resolution and with int8/int4 weights?

Weight-only quantization is simulated (quantize -> dequantize every nn.Linear of the transformer, per-channel int8 and
group-wise int4), which gives the same numbers an int8/int4 runtime would; speed on this GPU means nothing here.
Prompts are the live ones (fixed per style x colour), encoded once, then the text encoder leaves the GPU.

Runs inside the vto-hair-api image on GPU 1 (the live API is on GPU 0):
  docker run --rm --gpus device=1 -v hf-cache:/hf:ro -v in:/inputs:ro -v quality_sweep.py:/app/sweep.py:ro -v out:/out \
      vto-hair-api python /app/sweep.py
Writes /out/<config>/<photo>__<style>.jpg, /out/sheet_<photo>__<style>.jpg (original | every config), /out/results.csv.
"""
import csv
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app/server")

import cv2
import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image, ImageOps

from pipeline import COLOUR_OVERRIDE, COLOUR_PHRASE, MODEL, SEED, STEPS, TEXT_PROMPT, fit
from vto_spike.recolor2 import colour_word, measure_hair_lightness
from vto_spike.segment import HAIR_MODEL, HairSegmenter, SelfieParser

IN, OUT = Path("/inputs"), Path("/out")
# photo -> styles (women get two f_ styles, men two m_ styles)
JOBS = {
    "Hongyoonhee_front": ["f_long_volume", "f_short_bob"],
    "kaggle_train_18-20_6": ["f_long_volume", "f_short_bob"],
    "Labina_front": ["f_long_volume", "f_short_bob"],
    "Nisha_front": ["f_long_volume", "f_short_bob"],
    "photo_a": ["m_short_buzz", "m_medium_quiff"],
    "photo_d": ["m_short_buzz", "m_medium_quiff"],
}
SIZES = [1024, 768, 512, 384]
WEIGHTS = ["bf16", "int8", "int4"]
CONFIGS = [(s, w, False) for w in WEIGHTS for s in SIZES] + [(512, "int8", True)]   # last: trimmed text tokens
TILE = (240, 320)
GROUP = 64   # int4 group size along the input dimension


def name(cfg) -> str:
    s, w, trim = cfg
    return f"{s}_{w}" + ("_trim" if trim else "")


def fake_quant(w: torch.Tensor, kind: str) -> torch.Tensor:
    x = w.float()
    if kind == "int8":   # symmetric, one scale per output channel
        sc = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
        return ((x / sc).round().clamp(-128, 127) * sc).to(w.dtype)
    out_f, in_f = x.shape   # int4: symmetric, one scale per GROUP inputs
    g = GROUP if in_f % GROUP == 0 else (32 if in_f % 32 == 0 else in_f)
    x = x.reshape(out_f, in_f // g, g)
    sc = x.abs().amax(dim=2, keepdim=True).clamp_min(1e-8) / 7
    return ((x / sc).round().clamp(-8, 7) * sc).reshape(out_f, in_f).to(w.dtype)


def drift(parser, orig: Image.Image, out: Image.Image) -> float:   # same as scripts/ref_compare.py / trim_embed_test.py
    face = parser(np.asarray(orig))[1] > 0.5
    face = cv2.erode(face.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    face = cv2.resize(face, out.size, interpolation=cv2.INTER_NEAREST) > 0
    o = np.asarray(orig.resize(out.size, Image.LANCZOS), np.float32)
    x = np.asarray(out, np.float32)
    return float(np.abs(o - x).mean(axis=2)[face].mean()) if face.sum() > 100 else float("nan")


def psnr(a: Image.Image, b: Image.Image) -> float:
    b = b.resize(a.size, Image.LANCZOS)
    d = np.asarray(a, np.float32) - np.asarray(b, np.float32)
    mse = float((d ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def tile(img: Image.Image, label: str) -> np.ndarray:
    im = cv2.cvtColor(np.asarray(img.convert("RGB").resize(TILE)), cv2.COLOR_RGB2BGR)
    cv2.putText(im, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return im


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(Path("/app/server/catalog.json").read_text(encoding="utf-8"))
    hair_prompt = {s["id"]: s["hair_prompt"] for s in catalog["styles"]}
    hair_seg, parser = HairSegmenter(HAIR_MODEL), SelfieParser()

    origs, colours = {}, {}
    for p in JOBS:
        img = fit(ImageOps.exif_transpose(Image.open(next(IN.glob(p + ".*")))).convert("RGB"), 1536)
        origs[p] = img
        rgb = np.asarray(img)
        meas = measure_hair_lightness(rgb, hair_seg(rgb), parser(rgb)[1])
        colours[p] = COLOUR_OVERRIDE.get(colour_word(meas), colour_word(meas))
        print(p, "colour", colours[p], flush=True)

    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    tok = pipe.tokenizer
    embeds = {}   # (style, colour) -> (full, trimmed)
    for p, styles in JOBS.items():
        for s in styles:
            key = (s, colours[p])
            if key in embeds:
                continue
            prompt = TEXT_PROMPT.format(hair=hair_prompt[s], colour=COLOUR_PHRASE[colours[p]])
            with torch.no_grad():
                enc = pipe.encode_prompt(prompt, device="cuda")
            full = (enc[0] if isinstance(enc, tuple) else enc).detach().clone()
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            mask = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)["attention_mask"][0].bool()
            assert mask.numel() == full.shape[1]
            embeds[key] = (full, full[:, mask.to(full.device)].clone())
            print("encoded", key, "real tokens", int(mask.sum()), flush=True)
    pipe.text_encoder.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    pipe.set_progress_bar_config(disable=True)

    linears = [(n, m) for n, m in pipe.transformer.named_modules() if isinstance(m, torch.nn.Linear)]
    original = {n: m.weight.data.cpu().clone() for n, m in linears}
    print(f"{len(linears)} Linear layers, {sum(w.numel() for w in original.values()) / 1e9:.2f} B weights", flush=True)

    def set_weights(kind: str) -> None:
        for n, m in linears:
            w = original[n].to("cuda")
            m.weight.data = w if kind == "bf16" else fake_quant(w, kind)

    def gen(photo: Image.Image, emb: torch.Tensor) -> tuple[Image.Image, float]:
        w, h = photo.size
        torch.cuda.synchronize()
        t0 = time.time()
        out = pipe(image=photo, prompt_embeds=emb, width=w // 16 * 16, height=h // 16 * 16, guidance_scale=1.0,
                   num_inference_steps=STEPS, generator=torch.Generator(device="cuda").manual_seed(SEED)).images[0]
        torch.cuda.synchronize()
        return out, time.time() - t0

    results = {}   # (cfg name, photo, style) -> image
    rows = []
    cur = None
    for cfg in CONFIGS:
        size, kind, trim = cfg
        if kind != cur:
            set_weights(kind)
            cur = kind
        cname = name(cfg)
        (OUT / cname).mkdir(exist_ok=True)
        for p, styles in JOBS.items():
            photo = fit(origs[p], size)
            for s in styles:
                full, tr = embeds[(s, colours[p])]
                img, t = gen(photo, tr if trim else full)
                results[(cname, p, s)] = img
                img.save(OUT / cname / f"{p}__{s}.jpg", quality=92)
                base = results.get(("1024_bf16", p, s))
                rows.append(dict(config=cname, photo=p, style=s, size=img.size, drift=round(drift(parser, photo, img), 2),
                                 psnr_vs_base=round(psnr(base, img), 2) if base is not None else "", t=round(t, 2)))
                print(rows[-1], flush=True)

    for p, styles in JOBS.items():
        for s in styles:
            tiles = [tile(origs[p], "original")]
            for cfg in CONFIGS:
                cn = name(cfg)
                r = next(x for x in rows if x["config"] == cn and x["photo"] == p and x["style"] == s)
                tiles.append(tile(results[(cn, p, s)], f"{cn} d{r['drift']}"))
            while len(tiles) % 5:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
            cv2.imwrite(str(OUT / f"sheet_{p}__{s}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])

    with open(OUT / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for cfg in CONFIGS:
        cn = name(cfg)
        r = [x for x in rows if x["config"] == cn]
        ps = [x["psnr_vs_base"] for x in r if x["psnr_vs_base"] != ""]
        print(f"SUMMARY {cn}: drift {np.nanmean([x['drift'] for x in r]):.2f}"
              + (f"  psnr vs 1024_bf16 {np.mean(ps):.1f}" if ps else ""), flush=True)
    parser.close()


if __name__ == "__main__":
    main()
