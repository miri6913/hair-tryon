"""Run hairstyle jobs through Qwen-Image-Edit-2511.

  variant text: edit prompt only            -> /work/outputs/qwen_text/<job id>.png
  variant ref : photo + reference hairstyle -> /work/outputs/qwen_ref/<job id>.png
Resumable (existing outputs are skipped). Appends seconds and peak VRAM per job to timings.csv in the output folder.

--sweep steps:max_side:cfg[,...] --job <id>: one job, several settings, one model load -> /work/outputs/qwen_<variant>_sweep/
  <job>__s<steps>_m<max_side>_c<cfg>.png (max_side 0 = the photo as is). For finding what causes artifacts, not for the full run.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from diffusers import BitsAndBytesConfig, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from PIL import Image

MODEL = "Qwen/Qwen-Image-Edit-2511"
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
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--outputs", default="/work/outputs")
    ap.add_argument("--limit", type=int, default=0, help="only the first N jobs (smoke test); 0 = all")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--job", help="only this job id")
    ap.add_argument("--sweep", help="comma list of steps:max_side:cfg, e.g. 40:768:4,20:768:4 (needs --job)")
    ap.add_argument("--quant", choices=["nf4", "none"], default="nf4",
                    help="nf4: 4-bit transformer (~11 GB, fits one 24 GB GPU); none: bf16 (~40 GB, needs more than one 24 GB GPU)")
    args = ap.parse_args()

    inputs = Path(args.inputs)
    jobs = json.loads((inputs / "jobs.json").read_text(encoding="utf-8"))
    if args.job:
        jobs = [j for j in jobs if j["id"] == args.job]
        if not jobs:
            raise SystemExit(f"no job {args.job}")
    if args.limit:
        jobs = jobs[: args.limit]

    # (job, steps, max_side, cfg, output name): one entry per image to make
    if args.sweep:
        if not args.job:
            raise SystemExit("--sweep needs --job")
        out = Path(args.outputs) / f"qwen_{args.variant}_sweep"
        runs = []
        for combo in args.sweep.split(","):
            steps, max_side, cfg = combo.split(":")
            runs.append((jobs[0], int(steps), int(max_side), float(cfg), f"{jobs[0]['id']}__s{steps}_m{max_side}_c{cfg}"))
    else:
        out = Path(args.outputs) / f"qwen_{args.variant}"
        runs = [(j, args.steps, 0, 4.0, j["id"]) for j in jobs]
    out.mkdir(parents=True, exist_ok=True)

    if args.quant == "nf4":
        # whole-model CPU offload alone does not fit: the bf16 transformer is ~40 GB and is moved to the GPU as one piece
        transformer = QwenImageTransformer2DModel.from_pretrained(
            MODEL, subfolder="transformer", torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
        )
        pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, transformer=transformer, torch_dtype=torch.bfloat16)
    else:
        pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()  # text encoder (~16 GB) and VAE take turns on the GPU
    pipe.set_progress_bar_config(disable=True)

    timings = out / "timings.csv"
    new = not timings.exists()
    with open(timings, "a", newline="", encoding="utf-8") as tf:
        tw = csv.writer(tf)
        if new:
            tw.writerow(["job", "seconds", "peak_vram_mib"])
        for job, steps, max_side, cfg, name in runs:
            dst = out / f"{name}.png"
            if dst.exists():
                continue
            photo = fit(Image.open(inputs / job["photo"]).convert("RGB"), max_side)
            if args.variant == "ref":
                images = [photo, fit(Image.open(inputs / job["reference"]).convert("RGB"), max_side)]
                prompt = REF_PROMPT
            else:
                images = [photo]
                prompt = job["prompt"]
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            result = pipe(
                image=images,
                prompt=prompt,
                negative_prompt=" ",
                true_cfg_scale=cfg,
                guidance_scale=1.0,
                num_inference_steps=steps,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
            ).images[0]
            secs = time.time() - t0
            result.save(dst)
            tw.writerow([name, f"{secs:.1f}", int(torch.cuda.max_memory_allocated() / 2**20)])
            tf.flush()
            print(f"{name}: {secs:.1f}s ({result.width}x{result.height})", flush=True)


if __name__ == "__main__":
    main()
