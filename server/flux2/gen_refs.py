"""Generate synthetic reference portraits for the hairstyle catalogue with FLUX.2 [klein] 4B (text to image).

Reads /work/refs/styles.json (data/hairapi_refs/styles.json, pushed by `hs push`); writes
/work/outputs/refs_candidates/<style id>__s<seed>.png, several seeds per style, for picking the best by eye.
Resumable (existing files are skipped). The people are synthetic: no real person, so no rights or consent issue.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

MODEL = "black-forest-labs/FLUX.2-klein-4B"
TEMPLATE = (
    "Studio portrait photo of an adult {who}, {hair}, natural dark brown hair, facing the camera, head and shoulders with "
    "clear space above the head, plain light grey background, even soft lighting, neutral expression, no hat, no glasses, "
    "photorealistic."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles", default="/work/refs/styles.json")
    ap.add_argument("--outputs", default="/work/outputs/refs_candidates")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--only", help="comma list of style ids (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N styles (smoke test); 0 = all")
    args = ap.parse_args()

    styles = json.loads(Path(args.styles).read_text(encoding="utf-8"))["styles"]
    if args.only:
        styles = [s for s in styles if s["id"] in args.only.split(",")]
    if args.limit:
        styles = styles[: args.limit]
    out = Path(args.outputs)
    out.mkdir(parents=True, exist_ok=True)

    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    for s in styles:
        prompt = TEMPLATE.format(who="woman" if s["gender"] == "female" else "man", hair=s["hair_prompt"])
        for seed in range(args.seeds):
            dst = out / f"{s['id']}__s{seed}.png"
            if dst.exists():
                continue
            t0 = time.time()
            img = pipe(prompt=prompt, width=768, height=1024, guidance_scale=1.0, num_inference_steps=4,
                       generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]
            img.save(dst)
            print(f"{dst.name}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
