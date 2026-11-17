"""Feasibility spike: can a generative editor produce believable hair recolors on our review photos?

Teachers (both SD 1.5-class, fp16, fit a 4 GB GTX 1650):
  inpaint : stable-diffusion-inpainting, hair mask repainted with a color prompt
  ip2p    : instruct-pix2pix, "make the hair <color>", composited through the hair mask
Face and background are bit-identical to the photo (output is only used inside the feathered hair mask), so any
difference in believability is down to the hair itself.

Outputs: results/review/teacher/<stem>__<shade>__<teacher>.png (crop-size result), and a sheet
results/review/teacher_spike.png: original | v2 shader | inpaint | ip2p for each photo x shade.
  python scripts/spike_teacher.py --photos kaggle_train_21-30_11,curly_005 --shades golden_blonde,blue
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from review_sheet import PHOTOS, find_photo  # noqa: E402
from vto_spike import viz  # noqa: E402
from vto_spike.recolor2 import SHADES, recolor_hair2  # noqa: E402

MASKS = ROOT / "results" / "masks" / "npy"
OUT = ROOT / "results" / "review" / "teacher"
SIZE = 512
SEED = 1234

# shade -> (inpaint prompt, ip2p instruction)
PROMPTS = {
    "golden_blonde": ("photo of a person with natural golden blonde hair, realistic hair strands, soft shine",
                      "make the hair golden blonde"),
    "chestnut": ("photo of a person with chestnut brown hair, realistic hair strands, soft shine",
                 "make the hair chestnut brown"),
    "blue": ("photo of a person with vivid blue dyed hair, realistic hair strands, glossy highlights",
             "make the hair vivid blue"),
    "copper": ("photo of a person with copper red hair, realistic hair strands, soft shine",
               "make the hair copper red"),
    "black": ("photo of a person with jet black hair, realistic hair strands, soft shine",
              "make the hair jet black"),
}
NEGATIVE = "wig, hat, cartoon, painting, blurry, deformed, extra fingers, text, watermark"


def head_crop(rgb: np.ndarray, mask: np.ndarray, pad: float = 0.35):
    """Square crop (x0, y0, side) around the hair bbox, clamped to the image."""
    ys, xs = np.where(mask > 0.5)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * (1 + 2 * pad))
    side = min(max(side, 128), min(rgb.shape[:2]))
    y0 = int(np.clip(cy - side / 2, 0, rgb.shape[0] - side))
    x0 = int(np.clip(cx - side / 2, 0, rgb.shape[1] - side))
    return x0, y0, side


def composite(photo: np.ndarray, gen: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Generated pixels only inside the feathered hair mask; everything else stays the original photo."""
    alpha = cv2.GaussianBlur(np.clip(mask, 0, 1), (0, 0), 1.5)[..., None]
    return np.clip(photo * (1 - alpha) + gen * alpha, 0, 255).astype(np.uint8)


def load_pipes(teachers: list[str]):
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
    pipes = {}
    # fp16 convs return NaN (black images) on this GTX 1650 (checked: VAE encode is non-finite in fp16), so fp32 with
    # model offload to fit 4 GB. Slow but correct.
    common = dict(torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
    if "inpaint" in teachers:
        p = StableDiffusionInpaintPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-inpainting", **common)
        p.enable_attention_slicing()
        p.enable_model_cpu_offload()
        pipes["inpaint"] = p
    if "ip2p" in teachers:
        p = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", **common)
        p.enable_attention_slicing()
        p.enable_model_cpu_offload()
        pipes["ip2p"] = p
    return pipes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default=",".join(PHOTOS))
    ap.add_argument("--shades", default="golden_blonde,chestnut,blue,copper")
    ap.add_argument("--teachers", default="inpaint,ip2p")
    ap.add_argument("--name", default="teacher_spike")
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()
    teachers = args.teachers.split(",")
    shades = args.shades.split(",")
    OUT.mkdir(parents=True, exist_ok=True)

    print("loading models ...", flush=True)
    pipes = load_pipes(teachers)
    times = {t: [] for t in teachers}
    rows = []
    for stem in args.photos.split(","):
        rgb = cv2.cvtColor(cv2.imread(str(find_photo(stem))), cv2.COLOR_BGR2RGB)
        mask = np.load(MASKS / f"{stem}.npy").astype(np.float32)
        if mask.shape != rgb.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        if (mask > 0.5).sum() < 500:
            print("skip (no hair in mask):", stem)
            continue
        x0, y0, side = head_crop(rgb, mask)
        sl = (slice(y0, y0 + side), slice(x0, x0 + side))
        crop, mcrop = rgb[sl], mask[sl]
        img512 = Image.fromarray(cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
        # inpaint region: hair mask grown a little so strand tips and the soft edge are repainted too
        grown = cv2.dilate((mcrop > 0.3).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
        mask512 = Image.fromarray(cv2.resize(grown, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST))

        for shade in shades:
            tiles = [viz.label(crop, stem[:22]),
                     viz.label(recolor_hair2(crop, mcrop, SHADES[shade]), f"v2 {shade}")]
            for t in teachers:
                gen = torch.Generator("cuda").manual_seed(SEED)
                t0 = time.time()
                if t == "inpaint":
                    res = pipes[t](prompt=PROMPTS[shade][0], negative_prompt=NEGATIVE, image=img512, mask_image=mask512,
                                   num_inference_steps=args.steps, guidance_scale=7.5, strength=0.85,
                                   generator=gen).images[0]
                else:
                    res = pipes[t](prompt=PROMPTS[shade][1], image=img512, num_inference_steps=args.steps,
                                   guidance_scale=7.5, image_guidance_scale=1.5, generator=gen).images[0]
                times[t].append(time.time() - t0)
                gen_full = cv2.resize(np.array(res.convert("RGB")), (side, side), interpolation=cv2.INTER_LANCZOS4)
                out = composite(crop.astype(np.float32), gen_full.astype(np.float32), mcrop)
                cv2.imwrite(str(OUT / f"{stem}__{shade}__{t}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                tiles.append(viz.label(out, t))
                print(f"{stem} {shade} {t}: {times[t][-1]:.1f}s", flush=True)
            rows.append(viz.row(tiles, 320))
    if rows:
        out = ROOT / "results" / "review" / f"{args.name}.png"
        cv2.imwrite(str(out), cv2.cvtColor(viz.stack(rows), cv2.COLOR_RGB2BGR))
        print("wrote", out)
    for t, v in times.items():
        if v:
            print(f"{t}: mean {np.mean(v):.1f}s/image over {len(v)}")


if __name__ == "__main__":
    main()
