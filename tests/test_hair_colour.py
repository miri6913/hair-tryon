import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vto_spike.recolor2 import colour_word, measure_hair_lightness  # noqa: E402


def _scene(hair_gray: int, skin_gray: int, highlight_gray: int | None = None):
    """Image with a hair block (optionally a bright sheen strip in it) and a face-skin block, plus their masks."""
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :100] = hair_gray
    if highlight_gray is not None:
        img[:, 40:100] = highlight_gray     # 60% of the hair is sheen: it lifts the median, the lower quartile stays dark
    img[:, 100:] = skin_gray
    hair = np.zeros((200, 200), np.float32)
    hair[:, :100] = 1
    face = 1 - hair
    return img, hair, face


def _L(gray: int) -> float:
    return float(cv2.cvtColor(np.full((1, 1, 3), gray / 255.0, np.float32), cv2.COLOR_RGB2LAB)[0, 0, 0])


def test_too_little_hair_gives_none_and_default():
    img, hair, face = _scene(20, 180)
    assert measure_hair_lightness(img, hair * 0, face) is None
    assert colour_word(None) == "dark brown"


def test_black_hair_stays_black_with_sheen():
    img, hair, face = _scene(18, 170, highlight_gray=120)
    m = measure_hair_lightness(img, hair, face)
    assert m["p50"] > m["p25"]          # the sheen lifts the median, not the lower quartile
    assert colour_word(m) == "black"


def test_bright_face_lowers_the_reading():
    # the same hair gray reads as brown in a normal photo and as darker when the face is far brighter than a normal one
    normal = measure_hair_lightness(*_scene(80, 150)[:1], _scene(80, 150)[1], _scene(80, 150)[2])
    bright = measure_hair_lightness(*_scene(80, 235)[:1], _scene(80, 235)[1], _scene(80, 235)[2])
    assert bright["eff"] < normal["eff"]


def test_dark_face_does_not_lighten_the_reading():
    dark_skin = measure_hair_lightness(*_scene(30, 60)[:1], _scene(30, 60)[1], _scene(30, 60)[2])
    assert abs(dark_skin["eff"] - dark_skin["p25"]) < 1e-6


def test_blonde_stays_blonde():
    img, hair, face = _scene(215, 190)
    assert colour_word(measure_hair_lightness(img, hair, face)) == "blonde"
