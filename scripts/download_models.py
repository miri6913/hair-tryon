"""Download the MediaPipe segmentation models used in the Phase 0 spike."""
from pathlib import Path
import requests

MODELS = {
    "hair_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite",
    "selfie_multiclass_256x256.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}
OUT = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        dest = OUT / name
        if dest.exists():
            print(f"skip {name} (exists)")
            continue
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"downloaded {name} ({len(r.content) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
