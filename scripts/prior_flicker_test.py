"""Does feeding the previous mask back (4th input channel) steady the hair model's edge?

Controlled test: a still photo + fresh gaussian sensor noise every frame, so any frame-to-frame change in the mask is
pure model jitter. Same edge-flicker definition as the Android app (LiteRtSegmenter): mean |p_t - p_(t-1)| over pixels
that are not clearly background or clearly hair.
"""
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[1]
FRAMES = 40
NOISE_SIGMA = 3.0 / 255.0   # ~ low-light phone sensor noise


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run(interp, inp, out, base, use_prior, rng):
    last_p = np.zeros((512, 512), np.float32)
    flicks, edge_px = [], []
    prev_final = None
    for t in range(FRAMES):
        noisy = np.clip(base + rng.normal(0, NOISE_SIGMA, base.shape), 0, 1).astype(np.float32)
        x = np.zeros((1, 512, 512, 4), np.float32)
        x[0, ..., :3] = noisy
        if use_prior:
            x[0, ..., 3] = last_p
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0]
        p = sigmoid(y[..., 1] - y[..., 0]).astype(np.float32)
        if t > 0:
            band = ((p > 0.05) | (last_p > 0.05)) & ((p < 0.95) | (last_p < 0.95))
            if band.sum() > 200:
                flicks.append(np.abs(p - last_p)[band].mean())
                edge_px.append(int(band.sum()))
        last_p = p
    return float(np.mean(flicks)), int(np.mean(edge_px))


def main():
    photos = sorted((ROOT / "data" / "photos_kaggle").glob("*.jpg"))[::25][:6]
    interp = Interpreter(model_path=str(ROOT / "models" / "hair_segmenter.tflite"))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    print(f"{'photo':34s} {'no prior':>10s} {'prior':>10s}  edge px   change")
    tot = [[], []]
    for path in photos:
        rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(path)), (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        base = rgb.astype(np.float32) / 255.0
        a, n = run(interp, inp, out, base, False, np.random.default_rng(1))
        b, _ = run(interp, inp, out, base, True, np.random.default_rng(1))
        tot[0].append(a)
        tot[1].append(b)
        print(f"{path.name:34s} {a:10.4f} {b:10.4f}  {n:7d}  {100 * (b - a) / a:+6.1f}%")
    print(f"{'MEAN':34s} {np.mean(tot[0]):10.4f} {np.mean(tot[1]):10.4f}  {'':7s}  {100 * (np.mean(tot[1]) - np.mean(tot[0])) / np.mean(tot[0]):+6.1f}%")


if __name__ == "__main__":
    main()
