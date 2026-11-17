"""Hair recolor v2: shades are measured distributions, not a single flat target color.

v1 (recolor.py) pushed every hair pixel to one (a, b) and one lightness offset, which reads as paint: real hair is
warmer/darker in shadow and paler in highlights, and real blonde has a chroma of ~15, not ~45.

v2 per pixel:
  1. base = masked Gaussian of L (as v1). Source quantiles of `base` over the hair are mapped onto the shade's real
     L quantiles with a monotone piecewise-linear curve, so the person's own shading survives but lands in the
     shade's real tonal range (instead of one constant lift). The slope is capped so shadow noise is not blown up.
  2. strand detail comes back as a ratio to the base (as v1), from a lightly denoised L, gated by how dark the
     source is (near-black hair has almost no real detail, only sensor/compression noise).
  3. (a, b) follow the new lightness along the shade's shadow -> mid -> highlight ramp, with separate gains for a
     (red-green) and b (yellow-blue) so blondes get warmth without a pink cast.
  4. Lightness changes only in the solid interior; chroma changes reach further into the soft mask edge, so a bright
     background cannot be lightened into a halo.
Shade tables come from scripts/hair_stats.py + scripts/build_shades.py (real photos from data/photos_kaggle).
"""
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .recolor import _lab_to_rgb, _masked_blur, _rgb_to_lab, target_lab

QUANTILES = (2.0, 10.0, 50.0, 90.0, 98.0)


@dataclass(frozen=True)
class Shade:
    L: tuple[float, float, float, float, float]  # lightness at QUANTILES of the hair pixels
    ab_shadow: tuple[float, float]  # (a, b) at the p10 lightness
    ab_mid: tuple[float, float]  # at p50
    ab_high: tuple[float, float]  # at p90
    gain_a: float = 1.0  # red-green chroma scale
    gain_b: float = 1.0  # yellow-blue chroma scale
    keep_chroma: float = 0.0  # fraction of the source chroma variation kept (reflected light, color casts)

    def warm(self, gain_a: float, gain_b: float) -> "Shade":
        return replace(self, gain_a=gain_a, gain_b=gain_b)


# From scripts/build_shades.py on the 132 usable Kaggle selfies, grouped by median hair lightness.
NATURAL = {
    "black": Shade(L=(2.0, 2.9, 7.2, 19.5, 29.1), ab_shadow=(2.1, 1.1), ab_mid=(4.0, 3.2), ab_high=(5.5, 6.0)),
    "dark_brown": Shade(L=(6.5, 9.0, 17.4, 32.6, 44.0), ab_shadow=(4.4, 4.1), ab_mid=(5.5, 6.8), ab_high=(5.8, 8.9)),
    "brown": Shade(L=(10.5, 15.4, 29.0, 46.6, 56.6), ab_shadow=(6.1, 8.1), ab_mid=(6.5, 11.2), ab_high=(6.1, 12.9)),
    "light_brown": Shade(L=(20.6, 26.8, 42.2, 62.3, 71.7), ab_shadow=(6.9, 10.8), ab_mid=(6.3, 12.7), ab_high=(5.1, 13.2)),
    "blonde": Shade(L=(35.6, 44.7, 64.0, 81.1, 89.5), ab_shadow=(9.3, 13.3), ab_mid=(6.5, 13.5), ab_high=(4.3, 12.3)),
}


def vivid(rgb: tuple[int, int, int], tonal: Shade, keep_chroma: float = 0.15) -> Shade:
    """A fashion color: take the target hue/chroma and let it follow a natural tonal distribution.

    Dyed hair stays saturated in shadow and loses saturation toward white in the highlights.
    """
    _, a0, b0 = (float(v) for v in target_lab(rgb))
    return replace(
        tonal,
        ab_shadow=(a0 * 0.95, b0 * 0.95),
        ab_mid=(a0 * 0.85, b0 * 0.85),
        ab_high=(a0 * 0.5, b0 * 0.5),
        gain_a=1.0, gain_b=1.0, keep_chroma=keep_chroma,
    )


# The measured "blonde" family mixes in grey/white hair (chroma ~15) and reads as ash on its own. The warm variants
# keep the same tonal distribution and raise b (yellow) much more than a (red), which is what avoids a pink cast.
# Gains are chosen by eye on results/review sheets, not measured.
SHADES: dict[str, Shade] = {
    **NATURAL,
    "golden_blonde": NATURAL["blonde"].warm(1.0, 1.9),
    "ash_blonde": NATURAL["blonde"].warm(0.6, 0.75),
    "honey": NATURAL["light_brown"].warm(1.0, 1.6),
    "chestnut": NATURAL["brown"].warm(1.05, 1.35),
    "copper": vivid((178, 82, 40), NATURAL["light_brown"]),
    "burgundy": vivid((120, 25, 50), NATURAL["dark_brown"]),
    "blue": vivid((40, 90, 200), NATURAL["light_brown"]),
    "violet": vivid((125, 60, 170), NATURAL["light_brown"]),
    "emerald": vivid((20, 140, 95), NATURAL["light_brown"]),
}


def measure_shade(rgb: np.ndarray, mask: np.ndarray, min_px: int = 3000) -> Shade | None:
    """The shade of the hair in a photo, measured like scripts/hair_stats.py (None if there is too little hair to measure)."""
    sel = cv2.erode((mask > 0.9).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if sel.sum() < min_px:
        return None
    lab = _rgb_to_lab(rgb)
    L, a, b = lab[..., 0][sel], lab[..., 1][sel], lab[..., 2][sel]
    lo, hi = np.percentile(L, [25, 75])
    bands = (L <= lo, (L > lo) & (L < hi), L >= hi)
    if min(int(s.sum()) for s in bands) < 50:
        return None
    ab = [(float(a[s].mean()), float(b[s].mean())) for s in bands]
    return Shade(L=tuple(float(v) for v in np.percentile(L, QUANTILES)), ab_shadow=ab[0], ab_mid=ab[1], ab_high=ab[2])


# Exposure-robust hair darkness, for naming the colour in a prompt. The median L of the hair moves a lot with light: sheen and a
# bright head lift black hair (true L ~7) to L 15-30 and it was named "dark brown"/"brown". So: the LOWER quartile (highlights
# sit in the upper half) divided by how bright the photo's own face skin is relative to a normally lit face.
REF_SKIN_L = 62.0                 # median Lab L of normally lit face skin
EXPOSURE_CLAMP = (1.0, 1.35)      # one-sided: only a BRIGHT face lowers the hair reading. A dark face is as likely a darker skin tone
                                  # as under-exposure, and lifting the reading for it turned dark hair into brown/blonde
HAIR_COLOURS = ((10, "black"), (19, "dark brown"), (31, "brown"), (46, "light brown"), (101, "blonde"))   # upper bound of effective L


def measure_hair_lightness(rgb: np.ndarray, hair: np.ndarray, face: np.ndarray | None = None, min_px: int = 3000) -> dict | None:
    """-> {"p25", "p50", "skin", "eff"} (None when there is too little hair). eff = p25 / exposure, exposure from the face skin median L."""
    sel = cv2.erode((hair > 0.9).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if sel.sum() < min_px:
        return None
    lab_l = _rgb_to_lab(rgb)[..., 0]
    p25, p50 = (float(v) for v in np.percentile(lab_l[sel], [25, 50]))
    skin = None
    if face is not None:
        fsel = cv2.erode((face > 0.9).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if fsel.sum() >= 1500:
            skin = float(np.median(lab_l[fsel]))
    exposure = 1.0 if skin is None else float(np.clip(skin / REF_SKIN_L, *EXPOSURE_CLAMP))
    return {"p25": p25, "p50": p50, "skin": skin, "eff": p25 / exposure}


def colour_word(measure: dict | None, default: str = "dark brown") -> str:
    """Colour name from the effective lightness; `default` when there is too little hair to measure (shaved head)."""
    if measure is None:
        return default
    return next(w for hi, w in HAIR_COLOURS if measure["eff"] < hi)


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _monotone_map(base: np.ndarray, src_q: np.ndarray, dst_q: np.ndarray, max_slope: float) -> np.ndarray:
    """Piecewise-linear map src_q -> dst_q, slope-1 outside [p2, p98], slope capped at max_slope.

    Where the destination gap is large relative to the source gap (typically the shadows of dark hair, which is
    nearly flat in the photo), the source gap is widened instead of letting a tiny lightness wobble become blotches.
    The median is the anchor; widening goes outward from it.
    """
    src = np.asarray(src_q, dtype=np.float64).copy()
    dst = np.asarray(dst_q, dtype=np.float64)
    mid = 2
    for i in range(mid + 1, len(src)):
        src[i] = max(src[i], src[i - 1] + max((dst[i] - dst[i - 1]) / max_slope, 0.5))
    for i in range(mid - 1, -1, -1):
        src[i] = min(src[i], src[i + 1] - max((dst[i + 1] - dst[i]) / max_slope, 0.5))
    xs = np.concatenate([[src[0] - 100.0], src, [src[-1] + 100.0]])
    ys = np.concatenate([[dst[0] - 100.0], dst, [dst[-1] + 100.0]])
    return np.interp(base, xs, ys).astype(np.float32)


def recolor_hair2(
    rgb: np.ndarray,
    mask: np.ndarray,
    shade: Shade,
    intensity: float = 1.0,
    detail_sigma: float = 4.0,
    detail_gain: float = 1.0,
    eps: float = 12.0,
    guard_px: int = 5,
    max_slope: float = 3.0,
) -> np.ndarray:
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    inten = float(np.clip(intensity, 0.0, 1.0))

    lab = _rgb_to_lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    core = mask > 0.5
    if core.sum() < 200:
        return rgb.copy()

    # soft interior mask used for the lightness work (guard / base statistics)
    m_in = _smoothstep(0.25, 0.75, mask)
    base = _masked_blur(L, m_in, detail_sigma)
    src_q = np.percentile(base[core], QUANTILES)
    new_base = _monotone_map(base, src_q, np.array(shade.L), max_slope)

    # noise-aware detail: denoise harder when the hair is noisy, and trust detail less where the source is near black
    resid = L - cv2.GaussianBlur(L, (0, 0), 1.0)
    noise = 1.4826 * float(np.median(np.abs(resid[core] - np.median(resid[core]))))
    sigma = 0.8 if noise < 0.8 else 1.4
    L_d = cv2.GaussianBlur(L, (0, 0), sigma)
    ratio = np.clip((L_d + eps) / (base + eps), 0.4, 1.8)
    gate = np.clip((base - 3.0) / 12.0, 0.3, 1.0)
    inner = cv2.GaussianBlur(cv2.erode(m_in, np.ones((guard_px, guard_px), np.uint8)), (0, 0), guard_px / 2)
    ratio = 1.0 + (ratio - 1.0) * inner * gate
    new_L = (new_base + eps) * np.power(ratio, detail_gain) - eps
    over = np.maximum(new_L - 88.0, 0.0)
    new_L = np.clip(np.minimum(new_L, 88.0) + 12.0 * (1.0 - np.exp(-over / 12.0)), 0.0, 100.0)

    knots = [shade.L[1], shade.L[2], shade.L[3]]
    new_a = np.interp(new_L, knots, [shade.ab_shadow[0], shade.ab_mid[0], shade.ab_high[0]]) * shade.gain_a
    new_b = np.interp(new_L, knots, [shade.ab_shadow[1], shade.ab_mid[1], shade.ab_high[1]]) * shade.gain_b
    if shade.keep_chroma > 0:
        wn = m_in / max(float(m_in.sum()), 1.0)
        new_a = new_a + shade.keep_chroma * (a - float((a * wn).sum()))
        new_b = new_b + shade.keep_chroma * (b - float((b * wn).sum()))

    # lightness only changes in the solid interior, chroma reaches into the soft edge: no lightened halo
    w_L = (_smoothstep(0.5, 0.9, mask) * inten)
    w_C = (_smoothstep(0.3, 0.8, mask) * inten)
    out = np.stack([L + (new_L - L) * w_L, a + (new_a - a) * w_C, b + (new_b - b) * w_C], axis=-1)
    return _lab_to_rgb(out.astype(np.float32))
