"""Overlay and grid helpers for inspecting masks and recolors."""
import cv2
import numpy as np


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 80), alpha: float = 0.5) -> np.ndarray:
    tint = np.zeros_like(rgb)
    tint[:] = color
    w = (np.clip(mask, 0, 1) * alpha)[..., None]
    return (rgb * (1 - w) + tint * w).astype(np.uint8)


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def row(images: list[np.ndarray], height: int = 320) -> np.ndarray:
    return np.concatenate([resize_to_height(i, height) for i in images], axis=1)


def stack(rows: list[np.ndarray]) -> np.ndarray:
    width = max(r.shape[1] for r in rows)
    padded = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    return np.concatenate(padded, axis=0)
