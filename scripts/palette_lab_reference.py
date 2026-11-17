"""Reference sRGB -> Lab values (OpenCV float path, same as src/vto_spike/recolor.py) for the app's palette.

Paste the output into android/hairtryon/app/src/test/.../ColorMathTest.kt when the palette changes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vto_spike.recolor import target_lab  # noqa: E402

PALETTE = {
    "platinum": (235, 220, 190), "golden blonde": (214, 170, 90), "copper": (178, 82, 40),
    "burgundy": (110, 20, 45), "chestnut": (100, 60, 35), "ash blue": (90, 120, 175),
    "violet": (130, 70, 170), "emerald": (20, 130, 90), "black": (25, 20, 20), "white": (255, 255, 255),
}
for name, rgb in PALETTE.items():
    l, a, b = target_lab(rgb)
    print(f"{name:14s} rgb={rgb}  lab=({l:.3f}, {a:.3f}, {b:.3f})")
