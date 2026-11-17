"""Run hairstyle jobs through FLUX.2 [klein] (4B Apache-2.0, 9B non-commercial).

  variant text: edit prompt only            -> /work/outputs/flux2_<size>_text/<job id>.png
  variant ref : photo + reference hairstyle -> /work/outputs/flux2_<size>_ref/<job id>.png
Resumable (existing outputs are skipped). Appends seconds and peak VRAM per job to timings.csv in the output folder.
"""
import argparse
import csv
import gc
import json
import time
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

MODELS = {"4b": "black-forest-labs/FLUX.2-klein-4B", "9b": "black-forest-labs/FLUX.2-klein-9B"}
REF_PROMPT = (
    "Image 1 is the person. Image 2 shows the target hairstyle. Give the person in image 1 the hairstyle (cut, length, "
    "volume, parting and hair color) of the hair in image 2. Keep the face, skin, expression, clothes, pose and background "
    "of image 1 exactly the same."
)


def fit(img: Image.Image, max_side: int) -> Image.Image:
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["text", "ref"], required=True)
    ap.add_argument("--size", choices=list(MODELS), default="4b")
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--outputs", default="/work/outputs")
    ap.add_argument("--limit", type=int, default=0, help="only the first N jobs (smoke test); 0 = all")
    ap.add_argument("--steps", type=int, default=4, help="klein is step-distilled: 4 steps, guidance 1.0")
    ap.add_argument("--max-side", type=int, default=1024, help="longest side fed to the model (0 = the photo as is)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--job", help="only this job id")
    ap.add_argument("--no-offload", action="store_true", help="keep the whole pipeline on the GPU (~17 GB) instead of CPU offload")
    ap.add_argument("--cache-prompt", action="store_true",
                    help="encode each distinct prompt once, then move the text encoder off the GPU (~8 GB less)")
    ap.add_argument("--tag", default="", help="suffix for the output folder, so benchmark runs do not mix with the main outputs")
    args = ap.parse_args()

    inputs = Path(args.inputs)
    jobs = json.loads((inputs / "jobs.json").read_text(encoding="utf-8"))
    if args.job:
        jobs = [j for j in jobs if j["id"] == args.job]
        if not jobs:
            raise SystemExit(f"no job {args.job}")
    if args.limit:
        jobs = jobs[: args.limit]
    out = Path(args.outputs) / f"flux2_{args.size}_{args.variant}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)

    pipe = Flux2KleinPipeline.from_pretrained(MODELS[args.size], torch_dtype=torch.bfloat16)
    if args.no_offload:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()  # text encoder, transformer and VAE take turns on the GPU
    pipe.set_progress_bar_config(disable=True)

    embeds: dict[str, torch.Tensor] = {}
    if args.cache_prompt:
        prompts = {REF_PROMPT} if args.variant == "ref" else {j["prompt"] for j in jobs}
        for p in sorted(prompts):
            with torch.no_grad():   # without it the result keeps an autograd graph that pins the text encoder's GPU weights
                e = pipe.encode_prompt(p, device="cuda")
            embeds[p] = (e[0] if isinstance(e, tuple) else e).detach().clone()
            del e
        if args.no_offload:
            pipe.text_encoder.to("cpu")   # not needed per request any more
            gc.collect()
            torch.cuda.empty_cache()
        print(f"cached {len(embeds)} prompt embeddings, text encoder off the GPU", flush=True)

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
            photo = fit(Image.open(inputs / job["photo"]).convert("RGB"), args.max_side)
            if args.variant == "ref":
                image = [photo, fit(Image.open(inputs / job["reference"]).convert("RGB"), args.max_side)]
                prompt = REF_PROMPT
            else:
                image = photo
                prompt = job["prompt"]
            # keep the photo's aspect ratio: the output size defaults to the input's, rounded to the model's multiple
            w, h = photo.size
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            text = {"prompt_embeds": embeds[prompt]} if args.cache_prompt else {"prompt": prompt}
            result = pipe(
                image=image,
                **text,
                width=w // 16 * 16,
                height=h // 16 * 16,
                guidance_scale=1.0,
                num_inference_steps=args.steps,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
            ).images[0]
            secs = time.time() - t0
            result.save(dst)
            tw.writerow([job["id"], f"{secs:.1f}", int(torch.cuda.max_memory_allocated() / 2**20)])
            tf.flush()
            print(f"{job['id']}: {secs:.1f}s ({result.width}x{result.height})", flush=True)


if __name__ == "__main__":
    main()
