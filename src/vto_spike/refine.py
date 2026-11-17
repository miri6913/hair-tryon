"""Edge refinement of a low-resolution hair mask using the photo as a guide."""
import cv2
import numpy as np


def guided_refine(rgb: np.ndarray, mask: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """Snap mask edges to image edges. rgb uint8 HxWx3, mask float32 HxW in [0, 1]."""
    guide = rgb.astype(np.float32) / 255.0
    out = cv2.ximgproc.guidedFilter(guide, mask.astype(np.float32), radius, eps)
    return np.clip(out, 0.0, 1.0)


def feather(mask: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma), 0.0, 1.0)
