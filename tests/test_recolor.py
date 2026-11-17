import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vto_spike.recolor import recolor_hair  # noqa: E402


def _textured_image(seed=0, size=96):
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), (60, 40, 30), dtype=np.float32)
    noise = rng.normal(0, 18, (size, size, 1)).astype(np.float32)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def test_zero_mask_leaves_image_unchanged():
    img = _textured_image()
    out = recolor_hair(img, np.zeros(img.shape[:2], np.float32), (230, 200, 120))
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_output_dtype_and_shape():
    img = _textured_image()
    out = recolor_hair(img, np.ones(img.shape[:2], np.float32), (200, 30, 40), lift=0.7)
    assert out.dtype == np.uint8 and out.shape == img.shape


def test_strand_detail_survives_lightening():
    img = _textured_image()
    out = recolor_hair(img, np.ones(img.shape[:2], np.float32), (225, 190, 110), lift=1.0)
    l_in = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0].astype(float)
    l_out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)[..., 0].astype(float)
    assert l_out.mean() > l_in.mean() + 20  # actually got lighter
    assert l_out.std() > 0.4 * l_in.std()  # texture not flattened


def test_intensity_zero_is_identity():
    img = _textured_image()
    out = recolor_hair(img, np.ones(img.shape[:2], np.float32), (10, 200, 10), intensity=0.0)
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1
