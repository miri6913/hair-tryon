"""Non-rigid face alignment of a generated crop to the photo's aligned crop.

Both HairFastGAN's output and the bald-proxy face are StyleGAN reconstructions: their faces are a little wider or narrower than the
photo's. Hair and skin are then positioned for the GAN's face edge, and between them and the photo's real face edge a strip of
neither-hair-nor-face is left (the "wig gap"). Warping the generated crop onto the photo's face landmarks first makes the hair's inner
edge meet the real face edge.

All crops are the 1024x1024 FFHQ-aligned crops from align.py, so one landmark set per crop is enough.
"""
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

SIZE = 1024
GRID = 128           # the displacement field is solved on a GRID x GRID lattice and upsampled
MAX_DISP = 60.0      # px; a landmark pair further apart than this is a detection error, not a face difference
SMOOTHING = 2.0      # thin-plate smoothing: dlib landmarks jitter by a few px
# dlib 68: jaw 0-16, brows 17-26, nose tip 30, eye corners 36/39/42/45, mouth corners 48/54
USE = list(range(0, 27)) + [30, 36, 39, 42, 45, 48, 54]


def landmarks(aligner, crop: np.ndarray) -> np.ndarray | None:
    """dlib 68 landmarks of the largest face in a 1024 crop, or None."""
    faces = aligner.detector(crop, 1)
    if not faces:
        return None
    face = max(faces, key=lambda r: r.width() * r.height())
    return np.array([(p.x, p.y) for p in aligner.predictor(crop, face).parts()], np.float64)


def face_warp(crop: np.ndarray, src_lm: np.ndarray | None, dst_lm: np.ndarray | None) -> tuple[np.ndarray, str]:
    """Warp crop (whose face has landmarks src_lm) so its face lies on dst_lm. Returns (crop, note); the crop is unchanged, with the
    reason in note, when landmarks are missing or too few pairs survive."""
    if src_lm is None or dst_lm is None:
        return crop, "no landmarks, not warped"
    s, d = src_lm[USE], dst_lm[USE]
    disp = np.hypot(*(s - d).T)
    keep = disp <= MAX_DISP
    if keep.sum() < 12:
        return crop, "too few landmark pairs, not warped"
    s, d, disp = s[keep], d[keep], disp[keep]
    edge = np.linspace(0, SIZE - 1, 5)
    anchors = np.array([(x, y) for x in edge for y in edge if x in (0, SIZE - 1) or y in (0, SIZE - 1)])   # 16 fixed border points
    # backward map: for every pixel of the photo-aligned frame, where to read in the generated crop
    rbf = RBFInterpolator(np.vstack([d, anchors]), np.vstack([s, anchors]), kernel="thin_plate_spline", smoothing=SMOOTHING)
    g = np.linspace(0, SIZE - 1, GRID)
    gx, gy = np.meshgrid(g, g)
    src = rbf(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(GRID, GRID, 2).astype(np.float32)
    map_x = cv2.resize(src[..., 0], (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    map_y = cv2.resize(src[..., 1], (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    out = cv2.remap(crop, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, f"warped, landmark offset mean {disp.mean():.1f}px max {disp.max():.1f}px"
