"""Paste a model's new hairstyle back onto the original photo so the face and background stay the original pixels.

Things the plain `max(old, new)` blend got wrong (seen on the hairstyle API validation):
  - the old hair mask is soft (0.3-0.7) where thin strands end, so a good part of the dark original pixels stayed over the forehead
    the model had already redrawn. The old mask is now hardened (BOOST_LO..BOOST_HI).
  - it also has HOLES where fringe tips hang over skin (the segmenter misses thin strands): those tips stayed as a jagged dark
    band at the brows. `leftover_strands` finds them by image difference (dark in the original, light in the model output).
  - the model's skin is often lighter than the real skin (up to ~15 L on some photos), which shows as a pale patch where the old
    hair was. The tone difference is measured on real skin just around the revealed region and applied to everything that is
    taken from the model output. The same difference is allowed for when looking for leftover strands: without that, skin that
    is merely lighter in the model output looks like "dark original, light output" and gets pasted in as a bright jagged band.
"""
import cv2
import numpy as np

FEATHER_SIGMA = 2.0
BOOST_LO, BOOST_HI = 0.05, 0.35         # old-mask confidence that ramps from "not hair" to "hair" (0.29 -> ~0.86, >=0.5 -> 1)
RING_NEAR, RING_FAR = 6, 30             # px: the real-skin ring that measures the tone difference
MIN_REGION, MIN_RING = 200, 400         # px: below these there is nothing to fix / too little skin to measure
TONE_CAP = np.array([14.0, 6.0, 6.0], np.float32)   # max |dL|, |da|, |db| applied
TONE_FEATHER = 6.0                      # px sigma: the shift fades out at the region edge, no new seam
SKIN_CHROMA_TOL = 7.0                   # Lab (a, b) distance from the model's revealed skin that still counts as skin
STRAND_NEAR = 40                        # px around the old hair where leftover strands are looked for
STRAND_DL = 22.0                        # L by which the (tone-adjusted) model output must be lighter than the original
STRAND_MAX_L = 45.0                     # ... and the original pixel must be dark (hair), not just a lit skin difference


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lab(rgb255: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(rgb255, 0, 255).astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _rgb255(lab: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB) * 255.0, 0, 255)


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def skin_tone_offset(lab_orig: np.ndarray, lab_out: np.ndarray, old_mask: np.ndarray, new_mask: np.ndarray,
                     face_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """How far the model's skin is from the real skin, measured around the region where hair used to be and is not any more.
    Returns (offset Lab 3-vector to ADD to the model output, boolean revealed-skin region, boolean skin-coloured pixels of the
    model output), or None if there is nothing to measure (a style that keeps the old hair's footprint, or too little real
    skin around).

    The face parser is only used to find the revealed skin (seed). Everything else is found by COLOUR: the parser calls half of a
    revealed forehead "not face" and ignores a bright, overexposed real forehead, so a parser-based ring landed on a shadowed
    temple and measured the wrong tone (seen on the pale-patch photo)."""
    revealed = (old_mask > 0.5) & (new_mask < 0.3)          # was hair, is not any more: skin or background in the model output
    seed = revealed & (face_out > 0.5)
    if seed.sum() < MIN_REGION:
        return None
    skin_ab = np.median(lab_out[..., 1:][seed], axis=0)    # (a, b) of the model's revealed skin
    chroma = lambda lab: np.hypot(lab[..., 1] - skin_ab[0], lab[..., 2] - skin_ab[1])  # noqa: E731
    skin_out = chroma(lab_out) < SKIN_CHROMA_TOL
    region = seed | (revealed & skin_out)
    r8 = region.astype(np.uint8)
    ring = (cv2.dilate(r8, _disk(RING_FAR)) > 0) & (cv2.dilate(r8, _disk(RING_NEAR)) == 0)
    ring &= (np.maximum(old_mask, new_mask) < 0.05) & skin_out & (chroma(lab_orig) < SKIN_CHROMA_TOL)
    if ring.sum() < MIN_RING:
        return None
    offset = np.median((lab_orig - lab_out)[ring], axis=0)   # median: the ring can mix lit and shadowed skin
    return np.clip(offset, -TONE_CAP, TONE_CAP), region, skin_out


def leftover_strands(lab_orig: np.ndarray, lab_out: np.ndarray, old_mask: np.ndarray, new_mask: np.ndarray,
                     dL: float = 0.0) -> np.ndarray:
    """Thin strands of the OLD hair that the hair segmenter missed (fringe tips hanging over skin leave holes in the old mask):
    near the old hair, dark in the original but light in the model's output (after adding the skin tone difference `dL`), and not
    hair in the output. Returns a soft 0..1 mask."""
    near = cv2.dilate((old_mask > 0.5).astype(np.uint8), _disk(STRAND_NEAR)) > 0
    L_orig, L_out = lab_orig[..., 0], lab_out[..., 0] + dL
    strands = near & (new_mask < 0.3) & (L_out - L_orig > STRAND_DL) & (L_orig < STRAND_MAX_L)
    strands = cv2.dilate(strands.astype(np.uint8), _disk(2)).astype(np.float32)   # cover the strand's own edge pixels
    return cv2.GaussianBlur(strands, (0, 0), 1.5)


def composite(orig: np.ndarray, old_mask: np.ndarray, out: np.ndarray, new_mask: np.ndarray,
              face_out: np.ndarray | None = None) -> np.ndarray:
    """orig/out: float32 HxWx3 in 0..255; masks: float32 HxW in 0..1; returns uint8. The union of the old and new hair is
    blended in: the old region matters when the new style is shorter (whatever the model drew there is kept).
    Pass the face-skin mask of the model output to also tone-match the revealed skin."""
    lab_orig, lab_out = _lab(orig), _lab(out)
    measured = skin_tone_offset(lab_orig, lab_out, old_mask, new_mask, face_out) if face_out is not None else None
    offset, region, skin_out = measured if measured is not None else (np.zeros(3, np.float32), None, None)
    strands = leftover_strands(lab_orig, lab_out, old_mask, new_mask, dL=float(offset[0]))
    # only the OLD mask is hardened: the new mask's soft edge is the natural hair edge
    old_taken = _smoothstep(BOOST_LO, BOOST_HI, old_mask)
    if region is not None:
        # EVERY skin pixel taken from the model gets the shift, not just the revealed core: the old mask's soft edge zone
        # (0.05-0.5) is now taken from the model too, and left unshifted it showed as a bright jagged band along the old hairline.
        # Gated by skin colour so a wall revealed behind removed hair is not darkened.
        taken = np.maximum(old_taken * (1.0 - new_mask), strands)
        skin_soft = cv2.GaussianBlur(skin_out.astype(np.float32), (0, 0), 2.0)
        shift = np.maximum(cv2.GaussianBlur(region.astype(np.float32), (0, 0), TONE_FEATHER), taken * skin_soft)
        out = _rgb255(lab_out + offset * np.clip(shift, 0.0, 1.0)[..., None])
    m = np.maximum(np.maximum(old_taken, strands), new_mask)
    m = cv2.GaussianBlur(m, (0, 0), FEATHER_SIGMA)[..., None]
    return np.clip(orig * (1 - m) + out * m, 0, 255).astype(np.uint8)
