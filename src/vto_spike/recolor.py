"""Luminance-preserving hair recolor in LAB. Per-pixel ops plus one blur, so it ports to GLSL."""
import cv2
import numpy as np


def _rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def target_lab(rgb_color: tuple[int, int, int]) -> np.ndarray:
    return _rgb_to_lab(np.array([[rgb_color]], dtype=np.uint8))[0, 0]


def _masked_blur(channel: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    num = cv2.GaussianBlur(channel * mask, (0, 0), sigma)
    den = cv2.GaussianBlur(mask, (0, 0), sigma)
    return num / np.maximum(den, 1e-4)


def recolor_hair(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    intensity: float = 1.0,
    lift: float = 0.0,
    detail_gain: float = 1.0,
    detail_sigma: float = 4.0,
    eps: float = 12.0,
    edge_guard: bool = True,
    guard_px: int = 5,
    mask_contrast: bool = True,
) -> np.ndarray:
    """Recolor the masked region.

    lift: 0 keeps the hair's own base lightness; 1 moves it fully to the target lightness
    (use for dark -> light). Fine strand detail is kept as a ratio to the blurred base, raised to
    detail_gain. eps keeps the ratio stable (and sensor noise quiet) when the base is near black.
    """
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if mask_contrast:
        # smoothstep(0.25, 0.75): pulls the soft guided-filter skirt in so color does not halo onto background
        t = np.clip((mask - 0.25) / 0.5, 0.0, 1.0)
        mask = t * t * (3.0 - 2.0 * t)
    lab = _rgb_to_lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    t_L, t_a, t_b = target_lab(color)

    base = _masked_blur(L, mask, detail_sigma)
    hair_weight = max(float(mask.sum()), 1.0)
    mean_base = float((base * mask).sum() / hair_weight)

    # Multiplicative detail: near-black hair has almost no additive L detail, so carry it as a
    # ratio to the base and re-apply it on the lifted base. gain > 1 exaggerates strands.
    ratio = (L + eps) / (base + eps)
    if edge_guard:
        # At the boundary L mixes in background while base is hair-only, so the ratio explodes
        # into a bright rim. Clamp it, and fade it to neutral (1.0) within a few px of the edge.
        ratio = np.clip(ratio, 0.4, 1.8)
        inner = cv2.GaussianBlur(cv2.erode(mask, np.ones((guard_px, guard_px), np.uint8)), (0, 0), guard_px / 2)
        ratio = 1.0 + (ratio - 1.0) * inner
    new_base = base + lift * (t_L - mean_base)
    new_L = (new_base + eps) * np.power(ratio, detail_gain) - eps
    # soft shoulder above L=85 so highlights do not clip to flat white
    over = np.maximum(new_L - 85.0, 0.0)
    new_L = np.minimum(new_L, 85.0) + 15.0 * (1.0 - np.exp(-over / 15.0))
    new_L = np.clip(new_L, 0.0, 100.0)

    # chroma follows lightness so bright hair is not over-saturated and dark hair is not neon
    scale = np.clip(new_L / max(float(t_L), 1.0), 0.45, 1.25)
    new_a = t_a * scale
    new_b = t_b * scale

    w = (mask * float(np.clip(intensity, 0.0, 1.0)))[..., None]
    new_lab = np.stack([new_L, new_a, new_b], axis=-1)
    return _lab_to_rgb(lab * (1.0 - w) + new_lab * w)


def naive_hue_shift(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], intensity: float = 1.0) -> np.ndarray:
    """Baseline for comparison: replace hue and saturation in HSV, keep value."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    t_hsv = cv2.cvtColor(np.array([[color]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)
    shifted = hsv.copy()
    shifted[..., 0] = t_hsv[0]
    shifted[..., 1] = t_hsv[1]
    shifted_rgb = cv2.cvtColor(np.clip(shifted, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    w = (np.clip(mask, 0, 1) * intensity)[..., None]
    return np.clip(rgb.astype(np.float32) * (1 - w) + shifted_rgb * w, 0, 255).astype(np.uint8)
