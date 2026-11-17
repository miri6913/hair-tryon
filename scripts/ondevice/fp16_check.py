"""Throwaway spike 3: is fp16 safe for the REAL klein-4B weights? (The phone GPU path computes in fp16; the blocks I timed had random weights.)

Runs the transformer in bf16 (live) vs float16 for 512 px, 2 and 3 steps, 6 photos x 2 styles, seed 0; counts NaN/Inf in the output image
and compares the images. int8 weights are simulated as in the other sweeps (setup() in lighten_sweep.py), so this is the on-device recipe.
  docker run --rm --gpus device=1 -v hf-cache:/hf:ro -v in:/inputs:ro -v od:/app/od:ro -v prev:/prev:ro -v out:/out vto-hair-api python /app/od/fp16_check.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, "/app/server")
sys.path.insert(0, "/app/od")

import cv2
import numpy as np
import torch
from PIL import Image

from lighten_sweep import gen, setup, tok_of
from pipeline import fit
from quality_sweep import JOBS, drift, psnr, tile

OUT = Path("/out/fp16")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pipe, parser, origs, colours, embeds = setup()
    first = next(iter(JOBS))
    gen(pipe, fit(origs[first], 512), embeds[(JOBS[first][0], colours[first])])   # warm-up
    rows, hooks_bad = [], []

    # catch non-finite activations inside the transformer (an fp16 overflow can become NaN only at the very end)
    def hook(name):
        def f(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            if torch.is_tensor(t) and not torch.isfinite(t).all():
                hooks_bad.append(name)
        return f
    for n, m in pipe.transformer.named_children():
        if n in ("transformer_blocks", "single_transformer_blocks"):
            for i, b in enumerate(m):
                b.register_forward_hook(hook(f"{n}.{i}"))

    orig_fwd = pipe.transformer.forward

    def fwd16(*args, **kw):
        """Run only the transformer in fp16 (as the phone GPU does); the VAE and scheduler stay bf16 like the live pipeline."""
        kw = {k: (v.to(torch.float16) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kw.items()}
        out = orig_fwd(*args, **kw)
        if isinstance(out, tuple):
            return (out[0].to(torch.bfloat16),) + tuple(out[1:])
        out.sample = out.sample.to(torch.bfloat16)
        return out

    for steps in (2, 3):
        for p, styles in JOBS.items():
            photo = fit(origs[p], 512)
            for s in styles:
                emb = embeds[(s, colours[p])]
                pipe.transformer.forward = orig_fwd
                pipe.transformer.to(torch.bfloat16)
                a, _ = gen(pipe, photo, emb, steps)
                hooks_bad.clear()
                pipe.transformer.to(torch.float16)
                pipe.transformer.forward = fwd16
                b, _ = gen(pipe, photo, emb, steps)
                bad_layers = sorted(set(hooks_bad))
                arr = np.asarray(b, np.float32)
                rows.append(dict(steps=steps, photo=p, style=s, psnr_fp16_vs_bf16=round(psnr(a, b), 2), drift_bf16=round(drift(parser, photo, a), 2),
                                 drift_fp16=round(drift(parser, photo, b), 2), black_or_flat=bool(arr.std() < 5), nonfinite_layers=len(bad_layers),
                                 first_bad=bad_layers[0] if bad_layers else ""))
                print(rows[-1], flush=True)
                (OUT / f"s{steps}").mkdir(exist_ok=True)
                cv2.imwrite(str(OUT / f"s{steps}" / f"{p}__{s}.jpg"),
                            np.hstack([tile(origs[p], "original"), tile(a, f"bf16 {steps} steps"), tile(b, f"fp16 psnr {rows[-1]['psnr_fp16_vs_bf16']}")]),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(OUT / "results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for steps in (2, 3):
        r = [x for x in rows if x["steps"] == steps]
        print(f"SUMMARY steps={steps}: psnr fp16 vs bf16 mean {np.mean([x['psnr_fp16_vs_bf16'] for x in r]):.1f} min {min(x['psnr_fp16_vs_bf16'] for x in r):.1f}, "
              f"images with non-finite activations {sum(x['nonfinite_layers'] > 0 for x in r)}/{len(r)}, flat/black {sum(x['black_or_flat'] for x in r)}", flush=True)
    parser.close()


if __name__ == "__main__":
    main()
