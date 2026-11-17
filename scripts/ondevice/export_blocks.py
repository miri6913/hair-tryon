"""Throwaway spike: export single FLUX.2-klein-4B transformer blocks (+ VAE) to LiteRT .tflite so the S20 can time them.

Latency depends on shapes, not on weight values, so the blocks use RANDOM weights (the real class, the real sizes).
Runs on the server on CPU, in the vto-hair-api image with ai-edge-torch installed into /work/pylibs:
  docker run --rm --user .. -e HOME=/tmp -e PYTHONPATH=/work/pylibs -v ondevice:/work vto-hair-api python /work/export_blocks.py
Writes /work/tflite/<name>.tflite (float32 weights; the GPU delegate runs them in fp16) and, if the quantizer works, <name>_i8.tflite
(dynamic-range int8 weights, for the CPU/XNNPACK path), and prints the max abs difference tflite-vs-torch on the same input.

Token counts (portrait 3:4, long side L, 16 px per token): target + reference image tokens, plus the trimmed text tokens.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import litert_torch as et
except ImportError:   # older name of the same package
    import ai_edge_torch as et

from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock, Flux2TransformerBlock

import diffusers.models.transformers.transformer_flux2 as tf2

OUT = Path(__import__("os").environ.get("OUTDIR", "/work/tflite"))
ONLY = [s for s in __import__("os").environ.get("ONLY", "").split(",") if s]   # e.g. ONLY=single: export only names starting so
_ORIG_ROPE = tf2.apply_rotary_emb
_R = torch.zeros(128, 128)               # interleaved rotate-half as a matrix: out[2i] = -x[2i+1], out[2i+1] = x[2i]
for _i in range(64):
    _R[2 * _i + 1, 2 * _i], _R[2 * _i, 2 * _i + 1] = -1.0, 1.0


def rope_4d(x, freqs_cis, use_real=True, use_real_unbind_dim=-1, sequence_dim=2):
    """Same result as diffusers' apply_rotary_emb, but no 5-D reshape/unbind/stack (the LiteRT GPU backend rejects rank > 4)."""
    assert use_real and use_real_unbind_dim == -1 and sequence_dim == 1
    cos, sin = freqs_cis
    cos, sin = cos[None, :, None, :].float(), sin[None, :, None, :].float()
    xf = x.float()
    return (xf * cos + (xf @ _R) * sin).to(x.dtype)
DIM, HEADS, HEAD_DIM, MLP = 3072, 24, 128, 3.0
TXT = 80                     # trimmed text tokens (real prompts are 63-70, rounded up)
SIZES = [int(a) for a in sys.argv[1:]] or [384, 512]   # long side in px
torch.manual_seed(0)


class Single(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = Flux2SingleTransformerBlock(DIM, HEADS, HEAD_DIM, MLP)

    def forward(self, x, mod, cos, sin):   # x [1,N,DIM] (text+image already joined), mod [1,3*DIM], cos/sin [N,HEAD_DIM]
        return self.b(hidden_states=x, encoder_hidden_states=None, temb_mod=mod, image_rotary_emb=(cos, sin))


class Double(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = Flux2TransformerBlock(DIM, HEADS, HEAD_DIM, MLP)

    def forward(self, img, txt, mod_img, mod_txt, cos, sin):   # mod_* [1,6*DIM], cos/sin [Ntxt+Nimg,HEAD_DIM]
        t, i = self.b(hidden_states=img, encoder_hidden_states=txt, temb_mod_img=mod_img, temb_mod_txt=mod_txt,
                      image_rotary_emb=(cos, sin))
        return t, i


class VaeDec(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.v = vae

    def forward(self, z):
        return self.v.decode(z, return_dict=False)[0]


class VaeEnc(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.v = vae

    def forward(self, x):
        return self.v.encode(x).latent_dist.mode()


def export(name: str, mod: nn.Module, args: tuple) -> None:
    if ONLY and not any(name.startswith(p) for p in ONLY):
        return
    mod = mod.eval()
    if isinstance(mod, (Single, Double)):   # prove the 4-D rope is the same function before exporting with it
        with torch.no_grad():
            tf2.apply_rotary_emb = _ORIG_ROPE
            a = mod(*args)
            tf2.apply_rotary_emb = rope_4d
            b = mod(*args)
        a, b = (a if isinstance(a, tuple) else (a,)), (b if isinstance(b, tuple) else (b,))
        print(f"rope check {name}: max|orig-4d| {max(float((x - y).abs().max()) for x, y in zip(a, b)):.2e}", flush=True)
    tf2.apply_rotary_emb = rope_4d
    t0 = time.time()
    try:
        with torch.no_grad():
            ref = mod(*args)
            model = et.convert(mod, args)
        path = OUT / f"{name}.tflite"
        model.export(str(path))
        got = model(*args)
        refs = [r for r in (ref if isinstance(ref, (tuple, list)) else (ref,))]
        gots = [g for g in (got if isinstance(got, (tuple, list)) else (got,))]
        diff = max(float(np.abs(np.asarray(g) - r.numpy()).max()) for g, r in zip(gots, refs))
        scale = max(float(np.abs(r.numpy()).max()) for r in refs)
        print(f"OK   {name}: {path.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s, max|tflite-torch| {diff:.2e} (output max {scale:.1f})",
              flush=True)
    except Exception as e:   # noqa: BLE001  keep going: one failed export must not hide the others
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        return
    for tag, fn in (("i8", "dynamic_wi8_afp32"), ("i4", "dynamic_wi4_afp32")):
        try:
            from ai_edge_quantizer import quantizer, recipe
            qt = quantizer.Quantizer(str(path))
            qt.load_quantization_recipe(getattr(recipe, fn)())
            qt.quantize().export_model(str(OUT / f"{name}_{tag}.tflite"), overwrite=True)
            print(f"OK   {name}_{tag}: {(OUT / f'{name}_{tag}.tflite').stat().st_size / 1e6:.0f} MB", flush=True)
        except Exception as e:   # noqa: BLE001
            print(f"FAIL {name}_{tag}: {type(e).__name__}: {str(e)[:200]}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(next(Path("/hf/hub").glob("models--black-forest-labs--FLUX.2-klein-4B/snapshots/*/vae/config.json"))))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    vae = AutoencoderKLFlux2.from_config(cfg).eval()
    for L in SIZES:
        h, w = L, L * 3 // 4
        n_img = (h // 16) * (w // 16)     # one image's tokens; the model sees target + reference
        ni, nj = 2 * n_img, TXT + 2 * n_img
        print(f"== L={L}: {w}x{h}px, {n_img} tokens per image, double: img {ni} txt {TXT}, single: {nj}", flush=True)
        cs = lambda n: (torch.randn(n, HEAD_DIM), torch.randn(n, HEAD_DIM))   # noqa: E731  (cos/sin; values are irrelevant for timing)
        export(f"single_{L}", Single(), (torch.randn(1, nj, DIM), torch.randn(1, 3 * DIM), *cs(nj)))
        export(f"double_{L}", Double(), (torch.randn(1, ni, DIM), torch.randn(1, TXT, DIM), torch.randn(1, 6 * DIM),
                                         torch.randn(1, 6 * DIM), *cs(TXT + ni)))
        export(f"vaedec_{L}", VaeDec(vae), (torch.randn(1, 32, h // 8, w // 8),))
        export(f"vaeenc_{L}", VaeEnc(vae), (torch.randn(1, 3, h, w),))


if __name__ == "__main__":
    main()
