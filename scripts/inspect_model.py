"""Print input/output tensor shapes and dtypes of a .tflite model (uses ai-edge-litert)."""
import sys
from pathlib import Path

from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[1]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "hair_segmenter.tflite"
interp = Interpreter(model_path=str(path))
interp.allocate_tensors()
print(path.name)
for d in interp.get_input_details():
    print("IN ", d["name"], list(d["shape"]), d["dtype"].__name__, d["quantization"])
for d in interp.get_output_details():
    print("OUT", d["name"], list(d["shape"]), d["dtype"].__name__, d["quantization"])
