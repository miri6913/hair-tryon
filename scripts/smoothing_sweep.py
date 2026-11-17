"""Choose the temporal smoothing for the hair mask by measurement.

For a few photos, two synthetic sequences with fresh sensor noise every frame:
  still:  the photo does not move  -> edge flicker of the final mask (lower = steadier)
  moving: the photo slides sideways -> IoU of the final mask against the clean, noise-free mask (higher = tracks better)
The raw model output is computed once per input variant (optionally pre-blurred); the temporal filters are then
applied offline. The filter is the one in mask_ema.frag (alpha + gain * |5-tap-average change|).
"""
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[1]
FRAMES = 36
SHIFT_PX = 2.0          # image slides this many 512-px-space pixels per frame (~fast head turn)
NOISE = 3.0 / 255.0
INPUT_BLUR = [0.0, 1.0]  # gaussian sigma (px at 512) applied to the model input


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class Model:
    def __init__(self):
        self.i = Interpreter(model_path=str(ROOT / "models" / "hair_segmenter.tflite"))
        self.i.allocate_tensors()
        self.inp, self.out = self.i.get_input_details()[0], self.i.get_output_details()[0]

    def __call__(self, rgb01):
        x = np.zeros((1, 512, 512, 4), np.float32)
        x[0, ..., :3] = rgb01
        self.i.set_tensor(self.inp["index"], x)
        self.i.invoke()
        y = self.i.get_tensor(self.out["index"])[0]
        return sig(y[..., 1] - y[..., 0]).astype(np.float32)


def shifted(base, t):
    M = np.float32([[1, 0, SHIFT_PX * t], [0, 1, 0]])
    return cv2.warpAffine(base, M, (512, 512), borderMode=cv2.BORDER_REFLECT)


def avg5(a):
    return 0.2 * (a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1))


def ema(seq, alpha, gain):
    prev = np.zeros_like(seq[0])
    out = []
    for p in seq:
        a = np.clip(alpha + gain * np.abs(avg5(p) - avg5(prev)), 0, 1)
        prev = prev + a * (p - prev)
        out.append(prev)
    return out


def flicker(seq):
    vals = []
    for t in range(6, len(seq)):
        p, q = seq[t], seq[t - 1]
        band = ((p > 0.05) | (q > 0.05)) & ((p < 0.95) | (q < 0.95))
        if band.sum() > 200:
            vals.append(np.abs(p - q)[band].mean())
    return float(np.mean(vals))


def iou(seq, clean):
    vals = []
    for t in range(6, len(seq)):
        a, b = seq[t] > 0.5, clean[t] > 0.5
        u = (a | b).sum()
        vals.append((a & b).sum() / u if u else 1.0)
    return float(np.mean(vals))


def main():
    model = Model()
    photos = sorted((ROOT / "data" / "photos_kaggle").glob("*.jpg"))[::25][:5]
    rng = np.random.default_rng(7)
    configs = [(a, g, b) for b in INPUT_BLUR for a in (1.0, 0.5, 0.3, 0.2, 0.12) for g in (0.0, 3.0)]
    still = {c: [] for c in configs}
    move = {c: [] for c in configs}
    for path in photos:
        base = cv2.cvtColor(cv2.resize(cv2.imread(str(path)), (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clean = [model(shifted(base, t)) for t in range(FRAMES)]
        for blur in INPUT_BLUR:
            def prep(img):
                noisy = np.clip(img + rng.normal(0, NOISE, img.shape), 0, 1).astype(np.float32)
                return cv2.GaussianBlur(noisy, (0, 0), blur) if blur > 0 else noisy
            raw_still = [model(prep(base)) for _ in range(FRAMES)]
            raw_move = [model(prep(shifted(base, t))) for t in range(FRAMES)]
            for (a, g, b) in configs:
                if b != blur:
                    continue
                still[(a, g, b)].append(flicker(ema(raw_still, a, g)))
                move[(a, g, b)].append(iou(ema(raw_move, a, g), clean))
        print("done", path.name, flush=True)

    print(f"\n{'blur':>5s} {'alpha':>6s} {'gain':>5s} | {'flicker':>8s} {'move IoU':>9s}")
    rows = [(c, np.mean(still[c]), np.mean(move[c])) for c in configs]
    for (a, g, b), f, m in rows:
        print(f"{b:5.1f} {a:6.2f} {g:5.1f} | {f:8.4f} {m:9.4f}")


if __name__ == "__main__":
    main()
