"""Are the hair model outputs logits or probabilities? Decides the postprocess in the Android app."""
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[1]
photo = ROOT / "data" / "photos_kaggle" / "kaggle_test_21-30_26.jpg"

interp = Interpreter(model_path=str(ROOT / "models" / "hair_segmenter.tflite"))
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(photo)), (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
x = np.zeros((1, 512, 512, 4), np.float32)
x[0, ..., :3] = rgb.astype(np.float32) / 255.0
interp.set_tensor(inp["index"], x)
interp.invoke()
y = interp.get_tensor(out["index"])[0]  # 512x512x2

s = y.sum(-1)
print("output min/max per channel:", y.reshape(-1, 2).min(0), y.reshape(-1, 2).max(0))
print("channel sum min/max/mean:", s.min(), s.max(), s.mean(), "(all ~1.0 => already probabilities)")
sm = np.exp(y) / np.exp(y).sum(-1, keepdims=True)
print("hair px (argmax==1):", int((y.argmax(-1) == 1).sum()), " softmax(ch1)>0.5:", int((sm[..., 1] > 0.5).sum()))
print("ch1 stats  raw: mean %.4f  softmax: mean %.4f" % (y[..., 1].mean(), sm[..., 1].mean()))
