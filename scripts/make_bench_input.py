"""Make the fixed 512x512 input image used by the Android LiteRT benchmark (gitignored asset)."""
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "photos_kaggle" / "kaggle_test_21-30_26.jpg"
DST = ROOT / "android" / "litert-bench" / "app" / "src" / "main" / "assets" / "bench_input.jpg"

img = cv2.imread(str(SRC))
DST.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(DST), cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
print("wrote", DST)
