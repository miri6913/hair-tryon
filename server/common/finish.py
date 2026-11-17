"""Finish stage, model-independent: raw 1024 GAN crops -> full-size photos with only the hair (and any skin it uncovers) replaced.

Per job:
  1. parse the raw crop with BiSeNet (hair / skin / ...), warp crop and labels back into the photo with the matrix from align.py
  2. clean plate: LaMa inpaints the old hair region (what the background looks like once the hair is gone)
  3. composite onto the ORIGINAL photo: new hair from the GAN; where old hair is no longer covered, GAN skin (forehead, ears, neck)
     recoloured to the photo's local skin tone and with its thin strands removed, else the clean plate. Everything else stays the
     original pixels. (LaMa for the forehead itself was tried and fills it with the background: the forehead is boxed in by hair.)
Runs inside the hairfast image (needs its repo for BiSeNet) on one GPU.

  python /code/common/finish.py --model hairfast [--raw-dir raw] [--photos a,b] [--force]
  reads  outputs/<model>/<raw-dir>/<job>.png (jobs from jobs.json; sweep names <job>__sm.. work too)
  writes outputs/<model>/final/<name>.png
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from align import Aligner   # noqa: E402  (before Parser puts the HairFastGAN repo first on sys.path)
from warp import face_warp, landmarks   # noqa: E402

REPO = Path("/opt/HairFastGAN")
HAIR = 17                                              # BiSeNet labels, see models/CtrlHair/.../my_parsing_util.py
FOREHEAD = 1                # BiSeNet plain-skin label: the tone reference, measured where the photo and the GAN both show skin
SKIN = [1, 7, 8, 14, 15]    # skin, ears, neck: what the GAN may paint where old hair is gone (brows/eyes stay the photo's)
FEATHER = 1.5
OLD_GROW = 12       # px the old-hair mask is grown so strand tips and the soft edge are replaced too
SKIN_BAND = 22      # px around the old hair where exposed skin is re-filled, so thin forehead strands the mask missed vanish
PLATE_GROW = 18     # px the inpaint region extends beyond the old hair
STUBBLE_STYLES = {"buzz_cut"}   # short cuts whose hairline should dissolve into speckles instead of a hard edge
STUBBLE_SIGMA = 9       # px sigma of the hair-density field the speckles follow (ramp about +-15 px around the mask edge)
BROW_RAMP = 22          # px over which the bald skin fades in above the brows (partial mode: next to real forehead skin)
BROW_EDGE = 9           # same, whole-zone mode: the strip above the brows is replaced up to them (the brows' own fine hairs are carried over)
UNDERLAY_BALD = 0.7     # share of the bald skin's tone in the underlay beneath the forehead (whole-zone mode)
SEAM_BILATERAL = 14     # colour sigma of the edge-aware smoothing at the seams
SIDE_SOFT = 28         # px over which the forehead zone's side edge (the face contour) fades in, whole-zone mode
BOUNDARY_CAP_WHOLE = 10  # Lab units: border tone correction cap in whole-zone mode (6 left the corner blocks lighter than the temple skin)
BROW_BAND = 16          # px above a brow within which its fine hairs and shadow are carried over onto the new skin
BROW_FRINGE = 10        # px from the old hair within which no brow-top detail is carried over (its shadow made a dark patch at the left brow start)
BROW_HAIR_GAIN = 0.9   # share of the photo's dark brow-top detail that is carried over
BROW_SHADOW = 0.05      # faint contact shadow right above the brows
WALL_BRIGHT = 40        # gray levels above the forehead's median: bald-face wall background, not skin
HAIR_DARK = 45         # gray levels below the skin's median that, next to the old hair, mark a pixel as hair though BiSeNet calls it face
ZONE_FADE = 70        # rows over which the zone's straight lower edge fades out past the brow ends
BROW_CAP = 10           # Lab units: the largest the brow-skin correction may differ from the global offset
STUBBLE_BAND = 14      # px from skin labels within which the stubble hairline applies (hair-to-skin boundaries only)
FOREHEAD_FEATHER = 4   # px sigma of the replaced forehead part's edge, so the bald skin fades into the photo's skin
PATCH_REACH = 60        # px around replaced forehead patches within which the kept real skin is the tone reference
PATCH_FEATHER = 10      # px sigma of a replaced patch's edge when real skin is kept around it
FH_GROW = 22           # px the replaced forehead part is grown around the hair on it, into one connected band
FH_WHOLE = 0.55         # above this share of the zone, replace the whole zone
STRAND_NEAR = 45        # px from the photo's hair within which a thin dark line is taken for a strand
STRAND_T = 25         # black-hat response (15 px) above which a thin dark line on the forehead counts as a hair strand
BOUNDARY_REACH = 40     # px: real skin within this distance of the filled skin is what the boundary tone match is measured on
BOUNDARY_SIGMA = 14     # px sigma of the boundary tone match (about how far into the filled skin it reaches)
GRAIN_BANDS = [(0.8, 1.6), (1.6, 3.2), (3.2, 6.4), (6.4, 12.8)]   # DoG bands (sigma pairs): pores, grain, fine mottling, soft mottling
FOREHEAD_GRAIN = 0.8    # share of the real skin's detail the forehead gets (foreheads are a bit finer than cheeks)
BOUNDARY_LUM = 18       # Lab L units: reference skin brighter/darker than the median by more than this is not skin (hair shadow, strands)
BOUNDARY_CAP = 6        # Lab units: the largest the border correction may differ from the global offset
SHADE_STRENGTH = 0.06   # how much the skin darkens right under the hair edge (contact shadow); 0.15 drew a visible dark band
SHADE_SIGMA = 6         # px sigma of that shadow
FACE_LABELS = list(range(1, 16))  # BiSeNet labels that are the person (skin, brows, eyes, ears, nose, mouth, neck), not hair/cloth/background
GAP_MAX = 44            # px: the widest hair-to-face gap (per row) that counts as a sliver to close; wider is genuine space
GAP_DEPTH = 70          # px into the hair over which the closing stretch fades out
HALO_NEAR = 70         # px around the old hair where ANY deviation from the smooth wall (flyaways, grey halo) is treated as hair
HALO_BAND = 130         # px: further out, only small deviating blobs touching the near zone count (long background lines stay)
HALO_DEV = 9            # gray levels from a 41 px median of the photo that count as "not smooth wall"
HALO_THIN = 10          # black-hat / top-hat (15 px) response for thin dark / light strands
HALO_MAX_BOX = 160      # px: largest bounding box of a far blob still taken as hair debris


class Parser:
    """BiSeNet face parsing, the weights HairFastGAN ships."""

    def __init__(self, device: str = "cuda", models: str = "/work/models/hairfast"):
        link = REPO / "pretrained_models"
        if not link.exists():
            link.symlink_to(Path(models) / "pretrained_models")
        os.chdir(REPO)
        sys.path.insert(0, str(REPO))
        from models.CtrlHair.external_code.face_parsing.model import BiSeNet  # noqa: E402

        self.net = BiSeNet(n_classes=19)
        self.net.load_state_dict(torch.load("pretrained_models/BiSeNet/face_parsing_79999_iter.pth", map_location="cpu"))
        self.net.eval().to(device)
        self.device = device
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    @torch.no_grad()
    def parse(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (label map uint8 at the input size, hair probability float32 at the input size). One forward pass. The labels are
        nearest-upsampled from 512 px (a staircase contour); the probability is bicubic, so its contour is smooth."""
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = torch.from_numpy(((x - self.mean) / self.std).transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        logits = self.net(x)[0].squeeze(0)
        lab = logits.argmax(0).cpu().numpy().astype(np.uint8)
        prob = torch.softmax(logits, 0)[HAIR].cpu().numpy()
        return (cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST),
                np.clip(cv2.resize(prob, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1))

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return self.parse(rgb)[0]


class Lama:
    """big-lama (TorchScript) inpainting: image float [0,1] + binary mask -> image."""

    def __init__(self, path: str, device: str = "cuda"):
        self.model = torch.jit.load(path, map_location=device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray, mask: np.ndarray, max_side: int = 1024) -> np.ndarray:
        h, w = rgb.shape[:2]
        s = min(1.0, max_side / max(h, w))
        hh, ww = max(8, round(h * s)), max(8, round(w * s))
        img = cv2.resize(rgb, (ww, hh), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(mask, (ww, hh), interpolation=cv2.INTER_NEAREST)
        ph, pw = (-hh) % 8, (-ww) % 8
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        msk = cv2.copyMakeBorder(msk, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        t = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(self.device)
        m = torch.from_numpy((msk > 0).astype(np.float32))[None, None].to(self.device)
        out = self.model(t, m)[0].permute(1, 2, 0).cpu().numpy()
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)[:hh, :ww]
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


def feather(m: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), FEATHER)[..., None]


def tone_offset(photo, gan, ref, edge_ref=None, brow_ref=None, brow_w=None, cap=None) -> np.ndarray:
    """gan recoloured to the photo's colour space by ONE Lab offset: the mean photo-minus-GAN difference over ref (skin in both, away
    from the old hair). A single transform keeps the GAN's own forehead-vs-cheek relationships, which are the realistic part; a
    spatially varying field extrapolated up to a forehead the photo never shows (it is under the fringe) made blotches.
    Boundary term: where the filled skin touches real skin, the offset is pulled towards the one measured on edge_ref (the photo's real
    skin just outside the filled area), by weight clip(density of edge_ref / 0.15, 0, 1) of a sigma BOUNDARY_SIGMA blur: full strength
    at a border, none further than about BOUNDARY_REACH inside the filled area, so no tone step shows at a border, and the field
    cannot spread over the whole forehead (that was the blotchy version)."""
    if ref.sum() < 500:
        return gan
    to_lab = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)  # noqa: E731
    lg, lp = to_lab(gan), to_lab(photo)
    diff = lp - lg
    off = ((diff * ref[..., None]).sum(axis=(0, 1)) / ref.sum())[None, None, :]
    if edge_ref is not None and edge_ref.sum() >= 300:
        # real skin only: label-1 pixels next to hair are shadowed or hair-tinted, and sampling them pulled a dark square into the fill
        lum = lp[..., 0]
        med = float(np.median(lum[edge_ref > 0]))
        edge_ref = edge_ref * (np.abs(lum - med) < BOUNDARY_LUM)
    if edge_ref is not None and edge_ref.sum() >= 300:
        den = cv2.GaussianBlur(edge_ref, (0, 0), BOUNDARY_SIGMA)
        num = cv2.GaussianBlur(diff * edge_ref[..., None], (0, 0), BOUNDARY_SIGMA)
        local = num / np.maximum(den, 1e-4)[..., None]
        wgt = np.clip(den / 0.15, 0, 1)[..., None]
        cap = BOUNDARY_CAP if cap is None else cap
        off = off + np.clip(local - off, -cap, cap) * wgt   # a border correction, never a different tone
    if brow_ref is not None and brow_ref.sum() >= 500:
        # the skin right under and between the brows is the clean real skin nearest to the filled forehead above them: match it there,
        # fading out with the distance from the brows (one bounded correction, not a spatial field)
        loc = ((diff * brow_ref[..., None]).sum(axis=(0, 1)) / brow_ref.sum())[None, None, :]
        off = off + np.clip(loc - off, -BROW_CAP, BROW_CAP) * brow_w[..., None]
    return cv2.cvtColor(np.clip(lg + off, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def patch_tone(img, photo, patch, kept) -> tuple[np.ndarray, str]:
    """Recolour the replaced forehead patches to the real forehead skin around them. Fitted on the photo's kept skin within PATCH_REACH px
    of the patches: Lab mean of the patches in img -> Lab mean of that skin, plus a contrast gain on L (clamped 0.85-1.15). One transform
    for the whole image (not a spatial field, which made blotches), and no 6-unit cap: a patch next to warm, darker real skin has to
    become warm and dark. -> (img, log note); unchanged when there is too little kept skin."""
    ring = kept & (cv2.dilate(patch.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PATCH_REACH + 1,) * 2)) > 0)
    if ring.sum() < 1500 or patch.sum() < 500:
        return img, "patch tone skipped (little kept skin)"
    to_lab = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)  # noqa: E731
    li, lp = to_lab(img), to_lab(photo)
    mi, sp = li[patch].mean(axis=0), li[patch].std(axis=0)
    mr, sr = lp[ring].mean(axis=0), lp[ring].std(axis=0)
    gain = float(np.clip(sr[0] / max(sp[0], 1e-3), 0.85, 1.15))
    out = li - mi
    out[..., 0] *= gain
    out += mr
    note = f"patch tone: dL {mr[0] - mi[0]:+.1f} da {mr[1] - mi[1]:+.1f} db {mr[2] - mi[2]:+.1f} gain {gain:.2f}"
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB), note


def add_skin_detail(img, zone, photo, ref_tex, seed, grain=None) -> tuple[np.ndarray, str]:
    """Give the smooth StyleGAN skin the fine detail the photo's skin has, only to the extent it is missing. Energy (robust std) of the
    real skin in four difference-of-Gaussian bands (pores, grain, fine mottling, soft mottling) is measured on ref_tex (this person's
    cheeks and temples) and, in the same bands, on img inside zone; noise shaped to each band is added with the std that closes the gap
    (independent noise adds in quadrature: sqrt(target^2 - have^2), never negative), where the target is the real skin's energy times
    FOREHEAD_GRAIN since a forehead is a bit finer than a cheek. Same amount on R, G, B. -> (img, log note)."""
    if ref_tex.sum() < 2000 or zone.sum() < 500:
        return img, "skin detail skipped (no reference skin)"
    gray = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)  # noqa: E731
    band = lambda g, s1, s2: cv2.GaussianBlur(g, (0, 0), s1) - cv2.GaussianBlur(g, (0, 0), s2)  # noqa: E731
    mad = lambda v: 1.4826 * float(np.median(np.abs(v - np.median(v))))  # noqa: E731
    g_photo, g_img = gray(photo), gray(img)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(g_img.shape).astype(np.float32)
    add = np.zeros_like(g_img)
    real, have, new = [], [], []
    for s1, s2 in GRAIN_BANDS:
        s_real = mad(band(g_photo, s1, s2)[ref_tex])
        s_have = mad(band(g_img, s1, s2)[zone])
        n = band(white, s1, s2)
        n /= max(float(n.std()), 1e-6)
        add += n * float(np.sqrt(max(((FOREHEAD_GRAIN if grain is None else grain) * s_real) ** 2 - s_have ** 2, 0.0)))   # adds in quadrature
        real.append(s_real)
        have.append(s_have)
    out = np.clip(img.astype(np.float32) + add[..., None], 0, 255).astype(np.uint8)
    g_out = gray(out)
    new = [mad(band(g_out, s1, s2)[zone]) for s1, s2 in GRAIN_BANDS]
    f = lambda v: "[" + " ".join(f"{x:.2f}" for x in v) + "]"  # noqa: E731
    return out, f"skin detail: real {f(real)} filled {f(have)} -> {f(new)}"


def remove_strands(img: np.ndarray, zone: np.ndarray, thresh: int = 30) -> np.ndarray:
    """DullRazor-style hair removal inside zone: thin, strongly dark lines (black-hat filter) are replaced by the smooth skin around them
    (normalised convolution; cv2.inpaint leaves blocky bands on wide strands). The threshold is high on purpose: skin creases and
    shading are not strands, and flagging them is what made the forehead look airbrushed. A flagged area wider than the blur has no
    known pixels near it, so the fill falls back to wider blurs (5 -> 15 -> 40 px) instead of dividing by ~0."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if zone.sum() < 200:
        return img
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))
    strand = ((bh > thresh) & zone).astype(np.uint8)
    if not strand.any():
        return img
    strand = cv2.dilate(strand, np.ones((5, 5), np.uint8))
    known = ((1 - strand) * zone).astype(np.float32)    # only skin inside the zone informs the fill (not the hair around it)
    f = img.astype(np.float32)
    fill, filled = f.copy(), np.zeros(known.shape, bool)
    for sigma in (5.0, 15.0, 40.0):
        den = cv2.GaussianBlur(known, (0, 0), sigma)
        ok = (den > 0.05) & ~filled
        est = cv2.GaussianBlur(f * known[..., None], (0, 0), sigma) / np.maximum(den, 1e-3)[..., None]
        fill[ok] = est[ok]
        filled |= ok
    a = cv2.GaussianBlur(strand.astype(np.float32), (0, 0), 1.0)[..., None] * filled[..., None]
    return np.clip(f * (1 - a) + fill * a, 0, 255).astype(np.uint8)


def forehead_zone(lab_crop: np.ndarray, m: np.ndarray, shape: tuple, lab_photo_crop: np.ndarray | None = None) -> np.ndarray:
    """Everything above the eyebrows as a mask in photo space (the caller ANDs it with the GAN's skin label, which gives the head's side
    boundary: no vertical cut). The aligned crop is upright, so the eyebrows' top row bounds it from below; the old-hair mask is not
    needed (it misses thin fringe strands)."""
    brows = np.isin(lab_crop, [2, 3])
    if brows.sum() < 50:
        return np.zeros(shape, bool)
    y_top = int(np.percentile(np.where(brows)[0], 50))   # brow centre row: the exact border is set by the distance ramp in finish_one
    # sides: the photo's own face width at brow level (its temples), so the zone ends where the real face ends and the hair, which is
    # extended to meet that edge, borders it along one line above and below the brow row
    x0, x1 = 0, lab_crop.shape[1]
    if lab_photo_crop is not None:
        pb = np.where(np.isin(lab_photo_crop, [2, 3]))[0]
        if len(pb) > 50:
            yc = int(np.median(pb))
            cols = np.where(np.isin(lab_photo_crop[max(yc - 30, 0): yc + 30], FACE_LABELS).any(axis=0))[0]
            if len(cols) > 50:
                x0, x1 = int(cols.min()) + 2, int(cols.max()) - 2
    zone = np.zeros(lab_crop.shape, np.uint8)
    zone[: max(y_top, 0), x0:x1] = 1
    h, w = shape
    return cv2.warpPerspective(zone, m, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST, borderValue=0) > 0


def stubble_alpha(prob, gan, near_skin, seed) -> tuple[np.ndarray, np.ndarray]:
    """Hair alpha and hair image for a short cut's hairline: instead of a hard edge, the hair dissolves into speckles whose density
    follows a wide blur of the hair probability (about +-15 px), only where hair meets skin. Speckles outside the mask need a hair
    colour, so beyond the mask the hair image is the mean hair colour nearby (the GAN's own pixels there are skin)."""
    h, w = prob.shape
    # density from the BINARY hair mask: BiSeNet's probability is never exactly 0 on a forehead, so a field built from it put speckles
    # far down the skin
    solid = (prob > 0.5).astype(np.float32)
    wide = cv2.GaussianBlur(solid, (0, 0), STUBBLE_SIGMA)
    n = cv2.GaussianBlur(np.random.default_rng(seed).random((h, w), dtype=np.float32), (0, 0), 0.6)
    n = (n.ravel().argsort().argsort() / n.size).reshape(h, w).astype(np.float32)   # rank-normalised: uniform in [0, 1)
    dots = cv2.GaussianBlur((wide > 0.12 + 0.8 * n).astype(np.float32), (0, 0), 0.5) * 0.9   # reach about 12 px outside, full 12 px inside
    f = gan.astype(np.float32)
    mean_hair = cv2.GaussianBlur(f * solid[..., None], (0, 0), 6) / np.maximum(cv2.GaussianBlur(solid, (0, 0), 6), 1e-3)[..., None]
    hair_img = np.where((solid > 0)[..., None], f, mean_hair)
    return np.where(near_skin, dots, -1.0).astype(np.float32), hair_img


def halo_mask(photo, old_bin, exclude) -> np.ndarray:
    """Old-hair debris the soft old-hair mask misses: flyaway strands and the grey halo around the hair. Anything that deviates from
    the smooth wall (median-filter difference, thin dark/light structures) near the old hair, outside the person and the new hair."""
    ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)  # noqa: E731
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dev = np.abs(gray.astype(np.int16) - cv2.medianBlur(gray, 41).astype(np.int16)) > HALO_DEV
    thin = (cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k) > HALO_THIN) | (cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k) > HALO_THIN)
    near = cv2.dilate(old_bin, ell(HALO_NEAR)) > 0
    cand = (dev | thin) & (cv2.dilate(old_bin, ell(HALO_BAND)) > 0) & ~exclude
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lbl[near & cand])] = True                                        # blobs touching the near zone ...
    keep &= np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) <= HALO_MAX_BOX   # ... if small
    keep[0] = False
    out = (cand & near) | keep[lbl]                                                 # everything near, plus small blobs reaching out
    return cv2.dilate(out.astype(np.uint8), ell(4)) > 0


def face_contour(photo_face, gan_face, yb, h) -> tuple[np.ndarray, np.ndarray]:
    """The person's left and right face edge per image row (NaN where there is none), as ONE smooth contour. Below the brows it is the
    photo's own face edge; above them, where the forehead is under the fringe in the photo, it is the generated head's edge (the GAN's
    hair was drawn for that head, so the two agree there; a straight line extrapolated from the photo cut inward and left a 30 px wedge
    of head outside it). Between the brow row yb - 10 and yb + 60 the two are cross-faded, so there is no step at the brow row (a step
    bit a notch into the hair edge and left a straight horizontal edge). The forehead zone's sides and the hair snap follow it."""
    def edges(mask):
        a, b = np.full(h, np.nan, np.float32), np.full(h, np.nan, np.float32)
        for y in range(h):
            cols = np.flatnonzero(mask[y])
            if cols.size >= 20:
                a[y], b[y] = cols[0], cols[-1]
        return a, b
    pl, pr = edges(photo_face)
    if yb is None:                      # no brows found: the photo's plain per-row edge
        return pl, pr
    gl, gr = edges(gan_face)
    t = np.clip((np.arange(h) - (yb + 25)) / 70.0, 0, 1).astype(np.float32)      # 0 = generated head, 1 = photo (the brow tips are ragged)
    out = []
    for p, g in ((pl, gl), (pr, gr)):
        p2, g2 = np.where(np.isnan(p), g, p), np.where(np.isnan(g), p, g)         # use whichever exists
        out.append((g2 * (1 - t) + p2 * t).astype(np.float32))
    return out[0], out[1]


def snap_hair(hair_img, a_hair, hair, xl_face, xr_face, debug=None) -> tuple[np.ndarray, np.ndarray]:
    """Close the sliver between the new hair and the person's face edge by moving the hair, not by inventing pixels.
    Per image row, the horizontal gap between the hair's inner edge and the face edge (left and right sides separately, at most GAP_MAX
    px: wider is real space) is measured, smoothed over rows, and the hair layer is stretched sideways by that amount: the full amount
    at its inner edge, fading to none GAP_DEPTH px deeper. One continuous warp, so no seams and strands stay strands.
    hair_img float (h,w,3), a_hair float (h,w) its alpha, hair bool mask, face bool mask. -> (hair_img, a_hair) warped."""
    from scipy.ndimage import median_filter
    h, w = hair.shape
    off = {"l": np.zeros(h, np.float32), "r": np.zeros(h, np.float32)}
    edge = {"l": np.zeros(h, np.float32), "r": np.zeros(h, np.float32)}   # column of the hair's inner edge
    xl, xr = np.zeros(h, np.float32), np.zeros(h, np.float32)
    raw_rows = []   # debug: (y, first face column, nearest hair column left of it, number of face runs in the row)
    for y in range(h):
        if np.isnan(xl_face[y]) or np.isnan(xr_face[y]):
            continue
        c0, c1 = int(round(xl_face[y])), int(round(xr_face[y]))
        xl[y], xr[y] = c0, c1
        hl = np.flatnonzero(hair[y, :c0])
        if debug is not None:
            raw_rows.append((y, c0, int(hl[-1]) if hl.size else -1, 1))
        if hl.size and 0 < c0 - hl[-1] - 1 <= GAP_MAX:
            off["l"][y], edge["l"][y] = c0 - hl[-1] - 1, hl[-1]
        hr = np.flatnonzero(hair[y, c1 + 1:])
        if hr.size and 0 < hr[0] <= GAP_MAX:
            off["r"][y], edge["r"][y] = hr[0], c1 + 1 + hr[0]
    def smooth_rows(v, valid, sigma=6.0):
        """Normalised smoothing over the rows that have a value (a jump between rows would tear the warp into one-row streaks)."""
        v, valid = v.astype(np.float32), valid.astype(np.float32)
        num = cv2.GaussianBlur((median_filter(v, size=5 if sigma < 4 else 9) * valid)[None, :], (0, 0), sigma)[0]
        den = cv2.GaussianBlur(valid[None, :], (0, 0), sigma)[0]
        return np.where(den > 0.05, num / np.maximum(den, 1e-3), v).astype(np.float32)

    stats = {k: int((off[k] > 0).sum()) for k in off}
    if debug is not None:
        debug["rows"] = raw_rows
        debug["raw_off"] = {k: off[k].copy() for k in off}
    for k in off:
        found = off[k] > 0
        # smoothed over the rows that HAVE a gap (zeros do not pull it down), fading out where no gap rows are near
        presence = cv2.GaussianBlur(found.astype(np.float32)[None, :], (0, 0), 3.0)[0]
        # little smoothing on purpose: the gaps can be only a few px, and a wide blur shrinks them so the hair stops short of the face;
        # plus 1.5 px so the hair edge slightly overlaps the face edge instead of stopping at it
        off[k] = (smooth_rows(off[k], found, 2.0) * np.clip(presence / 0.5, 0, 1) + 1.5 * (off[k] > 0)).astype(np.float32)
        edge[k] = smooth_rows(edge[k], found, 2.0)
    face_rows = xr > 0
    xl, xr = smooth_rows(xl, face_rows), smooth_rows(xr, face_rows)
    print(f"   hair snap: rows with a left gap {stats['l']}, right gap {stats['r']}; "
          f"mean offset left {off['l'][off['l'] > 0.5].mean() if (off['l'] > 0.5).any() else 0:.1f}px "
          f"right {off['r'][off['r'] > 0.5].mean() if (off['r'] > 0.5).any() else 0:.1f}px", flush=True)
    xs = np.arange(w, dtype=np.float32)[None, :]
    depth_l = np.clip(edge["l"][:, None] - xs, 0, None)                   # how far inside the hair (left side), 0 in the gap
    depth_r = np.clip(xs - edge["r"][:, None], 0, None)
    wl = np.where(xs <= xl[:, None], np.clip(1 - depth_l / GAP_DEPTH, 0, 1), 0)   # nothing is moved on the face itself
    wr = np.where(xs >= xr[:, None], np.clip(1 - depth_r / GAP_DEPTH, 0, 1), 0)
    map_x = (xs - off["l"][:, None] * wl + off["r"][:, None] * wr).astype(np.float32)
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    return (cv2.remap(hair_img.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
            cv2.remap(a_hair.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))


def write_gap_debug(folder: Path, name: str, photo, hair, face, dbg) -> None:
    """Overlay of what snap_hair sees (new hair red, face mask green, rows with a measured gap marked blue on the left / cyan on the right
    edge) plus a text table of the raw per-row numbers, so a missed gap can be read off instead of guessed."""
    folder.mkdir(parents=True, exist_ok=True)
    vis = photo.astype(np.float32) * 0.55
    vis[hair] = vis[hair] * 0.4 + np.array([255, 40, 40], np.float32) * 0.6
    vis[face & ~hair] = vis[face & ~hair] * 0.5 + np.array([40, 255, 40], np.float32) * 0.5
    for y, xl_, hl_, runs in dbg["rows"]:
        if dbg["raw_off"]["l"][y] > 0:
            vis[y, max(xl_ - 6, 0): xl_] = (60, 120, 255)
        if dbg["raw_off"]["r"][y] > 0:
            vis[y, -1] = (0, 255, 255)
    Image.fromarray(np.clip(vis, 0, 255).astype(np.uint8)).save(folder / f"{name}__snap.png")
    lines = ["y first_face_col nearest_hair_col_left face_runs left_offset"]
    for y, xl_, hl_, runs in dbg["rows"][::6]:
        lines.append(f"{y} {xl_} {hl_} {runs} {dbg['raw_off']['l'][y]:.0f}")
    (folder / f"{name}__snap.txt").write_text("\n".join(lines), encoding="utf-8")


def write_owner_debug(folder: Path, name: str, photo, layers: dict, boxes: dict) -> None:
    """Which layer owns each pixel: new hair red, real face green, forehead-zone skin magenta, other GAN skin yellow, wall plate cyan.
    Writes 3x crops of the given boxes {label: (x0, y0, x1, y1)} side by side with the plain photo, to read a spot's cause off the pixels."""
    folder.mkdir(parents=True, exist_ok=True)
    colours = {"hair": (255, 40, 40), "face": (40, 255, 40), "forehead": (255, 40, 255), "skin": (255, 255, 40), "plate": (40, 255, 255)}
    vis = photo.astype(np.float32)
    for key, col in colours.items():
        m = layers[key]
        vis[m] = vis[m] * 0.55 + np.array(col, np.float32) * 0.45
    rows = []
    for label, (x0, y0, x1, y1) in boxes.items():
        if x1 > photo.shape[1] or y1 > photo.shape[0]:      # the boxes are for the 1152x1536 photos
            continue
        a = cv2.resize(photo[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(np.clip(vis[y0:y1, x0:x1], 0, 255).astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        rows.append(np.hstack([a, np.full((a.shape[0], 6, 3), 255, np.uint8), b]))
    if rows:
        wmax = max(r.shape[1] for r in rows)    # boxes can differ in width
        rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        Image.fromarray(np.vstack(rows)).save(folder / f"{name}__owners.png")


def finish_one(photo, old_mask, raw, face_crop, m, parser, lama, skin_crop=None, stubble=False, seed=0, aligner=None,
               debug_dir=None, debug_name="") -> np.ndarray:
    """photo RGB uint8 (h,w,3), old_mask uint8 0/255 (h,w), raw 1024 RGB crop, face_crop the aligned photo, m photo->crop.
    skin_crop: optional 1024 crop (the bald-proxy face) to take the exposed skin from instead of raw; the hair always comes from raw.
    stubble: short cut, so the hairline fades into speckles instead of a hard edge.
    aligner: when given, raw and skin_crop are first warped onto the photo's face landmarks (warp.py), which closes the gap between the
    new hair / GAN skin and the photo's real face edge."""
    if aligner is not None:
        dst_lm = landmarks(aligner, face_crop)
        raw, note = face_warp(raw, landmarks(aligner, raw), dst_lm)
        notes = [f"hair {note}"]
        if skin_crop is not None:
            skin_crop, note = face_warp(skin_crop, landmarks(aligner, skin_crop), dst_lm)
            notes.append(f"skin {note}")
        print("   " + "; ".join(notes), flush=True)
    h, w = photo.shape[:2]
    flags = cv2.WARP_INVERSE_MAP
    warp = lambda img: cv2.warpPerspective(img, m, (w, h), flags=flags | cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)  # noqa: E731
    warp_lab = lambda labels: cv2.warpPerspective(labels, m, (w, h), flags=flags | cv2.INTER_NEAREST, borderValue=255)  # noqa: E731
    gan = warp(raw)
    skin_src = gan if skin_crop is None else warp(skin_crop)
    lab_crop, hair_prob_crop = parser.parse(raw)
    lab_photo_crop = parser(face_crop)
    lab, lab_photo = warp_lab(lab_crop), warp_lab(lab_photo_crop)
    prob = np.clip(cv2.warpPerspective(hair_prob_crop, m, (w, h), flags=flags | cv2.INTER_CUBIC, borderValue=0), 0, 1)
    valid = cv2.erode((lab != 255).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)  # noqa: E731
    new_hair = cv2.morphologyEx((lab == HAIR).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    new_hair &= valid
    old_bin = (old_mask > 127).astype(np.uint8)
    near_old = cv2.dilate(old_bin, ell(SKIN_BAND)) > 0
    photo_face = np.isin(lab_photo, FACE_LABELS)
    # BiSeNet calls the old fringe's dark tips lying over the temple "skin"; "never overwrite the real face" then protected old hair as if it
    # were skin (the dark smear at the right brow end). Dark pixels labelled plain skin right next to the old hair are hair, not face.
    skin_gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32)
    skin_ref = (lab_photo == FOREHEAD) & ~(cv2.dilate(old_bin, ell(SKIN_BAND)) > 0)
    if skin_ref.sum() > 500:
        dark = (lab_photo == FOREHEAD) & (skin_gray < float(np.median(skin_gray[skin_ref])) - HAIR_DARK) & (cv2.dilate(old_bin, ell(10)) > 0)
        dark = cv2.dilate(dark.astype(np.uint8), ell(3)) > 0
        photo_face = photo_face & ~dark
    # one smooth face contour (measured below the brows, a straight temple line above): the forehead zone's sides and the hair snap follow it
    brows_ph = np.isin(lab_photo, [2, 3])
    yb = int(np.median(np.flatnonzero(brows_ph.any(axis=1)))) if brows_ph.sum() > 50 else None
    xl_c, xr_c = face_contour(photo_face, np.isin(lab, FACE_LABELS) & valid, yb, h)
    fh_cand = forehead_zone(lab_crop, m, (h, w)) & ~new_hair & (lab == FOREHEAD) & valid       # the forehead's geometry
    if yb is not None:
        cols_ = np.arange(w, dtype=np.float32)[None, :]
        fh_cand &= (cols_ > xl_c[:, None] + 2) & (cols_ < xr_c[:, None] - 2)   # NaN rows compare False: no zone there
    # only what has hair on it in the PHOTO is replaced: a forehead that is already bare (test photo A) stays the photo's own pixels, with its own
    # pores and tone. Hair on it = the photo's hair/hat labels, the old-hair mask, and thin strands lying over skin (BiSeNet calls those
    # skin and the soft mask misses them: keeping them left the fringe "mountain" on test photo B)
    # a thin dark line only counts as a strand near hair: pores and acne scars give the same filter response on a bare forehead (test photo A),
    # and made every job replace the whole zone
    hair_near = (cv2.dilate(old_bin, ell(STRAND_NEAR)) > 0) | (cv2.dilate(np.isin(lab_photo, [HAIR, 18]).astype(np.uint8), ell(STRAND_NEAR)) > 0)
    strands = (cv2.morphologyEx(cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY), cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))) > STRAND_T) & fh_cand & hair_near
    hair_cover = ((cv2.dilate(np.isin(lab_photo, [HAIR, 18]).astype(np.uint8), ell(5)) > 0) | (cv2.dilate(old_bin, ell(5)) > 0)
                  | (cv2.dilate(strands.astype(np.uint8), ell(7)) > 0))
    # patches that are only the hair itself leave blotches (hair shadow, tone steps against the kept skin): grow the replaced part into one
    # connected band; and when most of the zone has hair anyway (a fringe), replace the whole zone, one even forehead beats a patchwork
    grown = (cv2.dilate((fh_cand & hair_cover).astype(np.uint8), ell(FH_GROW)) > 0) & fh_cand
    share = float(grown.sum()) / max(int(fh_cand.sum()), 1)
    partial = share <= FH_WHOLE
    fh_zone = grown if partial else fh_cand
    if partial:
        # round, cut-out outlines read as patches: close small gaps, then round the contour off with a blur-and-threshold
        sm = cv2.morphologyEx(fh_zone.astype(np.uint8), cv2.MORPH_CLOSE, ell(4))
        fh_zone = (cv2.GaussianBlur(sm.astype(np.float32), (0, 0), 3.0) > 0.5) & fh_cand
    print(f"   forehead: hair on {100 * (fh_cand & hair_cover).sum() / max(int(fh_cand.sum()), 1):.0f}% of the zone; "
          f"replaced {100 * fh_zone.sum() / max(int(fh_cand.sum()), 1):.0f}%{' (whole zone)' if share > FH_WHOLE else ''}", flush=True)
    if not partial and yb is not None:
        # the zone ends at the brow-centre row, but the brows slope down at their tails: a strip of the photo's skin (with the fringe's
        # shadow) was left between the zone and the brow, a lighter rectangle past the tail. Extend the zone down to the top of each brow.
        brows_m = (np.isin(lab, [2, 3]) | np.isin(lab_photo, [2, 3]))
        has = brows_m.any(axis=0)
        top = np.where(has, brows_m.argmax(axis=0), 0)[None, :]
        rows_e = np.arange(h)[:, None]
        bottom_e = (fh_cand * rows_e).max(axis=0)[None, :]
        extra = has[None, :] & (rows_e > bottom_e) & (rows_e < top) & (top - rows_e < 80) & valid
        extra &= (cols_ > xl_c[:, None] + 2) & (cols_ < xr_c[:, None] - 2)
        fh_cand = fh_cand | extra
        fh_zone = fh_zone | extra
    # GAN skin only where the photo has no skin of its own: pixels the photo shows as real face stay the photo's (the band used to reach
    # 22 px into the real face, leaving a hard seam and a different-looking strip beside the hair). The forehead above the brows is the
    # exception: it is under the fringe in the photo.
    skin_fill = (near_old & ~new_hair & np.isin(lab, SKIN) & valid & ~photo_face) | fh_zone
    # flyaway strands and the grey halo around the old hair lie outside its soft mask: add them to what gets inpainted
    halo = halo_mask(photo, old_bin, photo_face | new_hair | skin_fill)
    # the real face is never overwritten by the wall plate: the old-hair mask grown by OLD_GROW reaches into the face edge, and the debug
    # overlay showed the pale strip beside the face was LaMa wall colour pasted over the person's own skin, not a gap
    face_core = cv2.erode(photo_face.astype(np.uint8), ell(2)) > 0
    plate_fill = ((cv2.dilate(old_bin, ell(OLD_GROW)) > 0) | halo) & ~new_hair & ~skin_fill & ~face_core

    # exposed skin comes from the GAN as generated (it draws a coherent face, the photo's own forehead is under the fringe), moved to
    # the photo's colour space by one offset measured on skin away from the old hair (pulled to the real skin right at every border),
    # its thin dark strands removed, and the fine skin detail it lacks added
    ref = ((lab == FOREHEAD) & (lab_photo == FOREHEAD) & valid & ~near_old & ~fh_zone).astype(np.float32)
    brows = (np.isin(lab, [2, 3]) | np.isin(lab_photo, [2, 3])).astype(np.uint8)
    brow_dist = cv2.distanceTransform(1 - brows, cv2.DIST_L2, 3)
    skin_u8 = skin_fill.astype(np.uint8)
    edge_ref = ((lab_photo == FOREHEAD) & valid & ~(cv2.dilate(brows, ell(3)) > 0) & ~skin_fill
                & (cv2.dilate(skin_u8, ell(BOUNDARY_REACH)) > 0)).astype(np.float32)
    brow_ref = brow_w = None
    if not partial and yb is not None:
        rows_ = np.arange(h)[:, None]
        lum_ph = cv2.cvtColor(photo, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
        cand_ref = ((lab_photo == FOREHEAD) & valid & ~hair_cover & (rows_ > yb - 5) & (cv2.dilate(brows, ell(40)) > 0)
                    & ~(cv2.dilate(brows, ell(3)) > 0))
        if cand_ref.sum() > 500:
            cand_ref &= np.abs(lum_ph - np.median(lum_ph[cand_ref])) < BOUNDARY_LUM
        brow_ref = cand_ref.astype(np.float32)
        brow_w = np.exp(-brow_dist / 80.0).astype(np.float32)      # reaches the glabella between the brows too
    gan_skin = tone_offset(photo, skin_src, ref, edge_ref, brow_ref, brow_w, None if partial else BOUNDARY_CAP_WHOLE)
    gan_skin = remove_strands(gan_skin, cv2.dilate(skin_u8, ell(3)) > 0)
    # reference skin for the detail: this person's cheeks and temples, away from features and from the old hair
    features = cv2.dilate(np.isin(lab_photo, [2, 3, 4, 5, 10, 11, 12, 13]).astype(np.uint8), ell(20)) > 0
    ref_tex = cv2.erode(((lab_photo == FOREHEAD) & valid & ~near_old & ~features).astype(np.uint8), ell(5)) > 0
    grain = None
    if partial:
        # patches next to real forehead skin: that skin is the reference for tone and for texture (same skin, so the full detail)
        kept = cv2.erode(((lab_photo == FOREHEAD) & fh_cand & ~fh_zone & valid).astype(np.uint8), ell(3)) > 0
        gan_skin, note = patch_tone(gan_skin, photo, fh_zone, kept)
        print("   " + note, flush=True)
        kept_tex = cv2.erode(kept.astype(np.uint8), ell(5)) > 0
        if kept_tex.sum() >= 1500:
            ref_tex, grain = kept_tex, 1.0
            print("   texture ref: kept forehead", flush=True)
    gan_skin, note = add_skin_detail(gan_skin, skin_fill, photo, ref_tex, seed, grain)
    print("   " + note, flush=True)
    # the bald face is only good for the forehead (hair is gone there and it is the only clean source). At the temples, ears and neck it is
    # narrower than the hair GAN's head and carries hair/wall remnants (a dark smear and a white streak at the right brow end): there the
    # hairstyle GAN's own skin, which already matches its hair edge, is used
    other = skin_fill & ~fh_zone
    if skin_crop is not None and other.any():
        skin_h = tone_offset(photo, gan, ref, edge_ref)
        skin_h = remove_strands(skin_h, cv2.dilate(skin_u8, ell(3)) > 0)
        skin_h, _ = add_skin_detail(skin_h, skin_fill, photo, ref_tex, seed + 1, grain)
        print("   temple/ear/neck skin: hairstyle GAN", flush=True)
        # the bald face's head is narrower than the zone at the sides: its wall background shows as a bright streak along the zone's edge.
        # Pixels much brighter than the rest of the forehead take the hairstyle GAN's skin instead
        gb = cv2.cvtColor(gan_skin, cv2.COLOR_RGB2GRAY).astype(np.float32)
        wall = cv2.dilate(((gb > float(np.median(gb[fh_zone])) + WALL_BRIGHT) & fh_zone).astype(np.uint8), ell(3)) > 0 if fh_zone.any() else None
        if wall is not None and wall.any():
            aw = cv2.GaussianBlur(wall.astype(np.float32), (0, 0), 2.0)[..., None]
            gan_skin = (gan_skin * (1 - aw) + skin_h * aw).astype(np.uint8)
    else:
        skin_h = gan_skin
    plate = lama(photo, ((cv2.dilate(old_bin, ell(PLATE_GROW)) > 0) | halo).astype(np.uint8) * 255).astype(np.float32)
    # backing at the hair-to-face seam: the wall-coloured plate must not show through the hair's anti-aliased edge next to the face (a thin
    # pale line). There, the plate is blended to the nearby hair colour
    d_face = cv2.distanceTransform(1 - photo_face.astype(np.uint8), cv2.DIST_L2, 3)
    d_hair = cv2.distanceTransform(1 - new_hair.astype(np.uint8), cv2.DIST_L2, 3)
    solid = new_hair.astype(np.float32)
    mean_hair = cv2.GaussianBlur(gan.astype(np.float32) * solid[..., None], (0, 0), 6) / np.maximum(cv2.GaussianBlur(solid, (0, 0), 6), 1e-3)[..., None]
    wb = (np.clip(1 - d_hair / 5.0, 0, 1) * (d_face <= 10))[..., None]
    plate = plate * (1 - wb) + mean_hair * wb

    # the hair layer first: the contact shadow on the skin needs its final alpha. From the smooth bicubic probability (the label map is a
    # 512 px nearest-upsampled staircase)
    a_hair = np.clip((cv2.GaussianBlur(prob, (0, 0), 1.0) - 0.3) / 0.4, 0, 1)
    hair_img = gan.astype(np.float32)
    a_seam = seam_colour = None
    if stubble:
        near_skin = cv2.dilate((np.isin(lab, SKIN) & valid).astype(np.uint8), ell(STUBBLE_BAND)) > 0
        dots, hair_stub = stubble_alpha(prob, gan, near_skin, seed)
        use = dots >= 0
        a_hair = np.where(use, dots, a_hair)
        hair_img = np.where(use[..., None], hair_stub, hair_img)
    else:
        # the generated hair's inner edge carries a pale rim (specular/fuzz from the mask edge): next to the face, replace the edge colour
        # with the hair colour a few px deeper, or the rim reads as a light strip between hair and face
        depth = cv2.distanceTransform(new_hair.astype(np.uint8), cv2.DIST_L2, 3)
        inner = (depth > 6).astype(np.float32)
        den_in = cv2.GaussianBlur(inner, (0, 0), 5)
        mean_in = cv2.GaussianBlur(hair_img * inner[..., None], (0, 0), 5) / np.maximum(den_in, 1e-3)[..., None]
        wdec = (np.clip(1 - depth / 6.0, 0, 1) * (d_face <= 14) * (den_in > 0.05) * 0.85)[..., None]
        hair_img = hair_img * (1 - wdec) + mean_in * wdec
        # the sliver between the hair and the face edge is neither skin nor wall: move the hair so it reaches the face. Hair that this adds
        # never covers a brow or an eye (a ragged brow tip bit a notch into the hair edge); hair that was already there (bangs) is untouched
        dbg = {} if debug_dir is not None else None
        a_before = a_hair
        hair_img, a_snap = snap_hair(hair_img, a_hair, new_hair, xl_c, xr_c, dbg)
        protect = cv2.GaussianBlur(cv2.dilate(np.isin(lab, [2, 3, 4, 5]).astype(np.uint8) | np.isin(lab_photo, [2, 3, 4, 5]).astype(np.uint8),
                                              ell(4)).astype(np.float32), (0, 0), 1.0)
        delta = a_snap - a_before
        a_hair = a_before + np.where(delta > 0, delta * (1 - protect), delta)
        # whatever thin sliver is still left between hair and face (a few px of skin or wall the snap did not reach) takes the hair colour
        seam = (d_hair <= 14) & (d_face <= 14) & ~photo_face & ~new_hair & ~fh_zone
        # where the brow protection took snapped hair away, the wall plate used to show through as a white sliver at the brow end: also seam
        seam |= (protect > 0.3) & (delta > 0.05) & (d_hair <= 24) & ~photo_face & ~new_hair
        a_seam = cv2.GaussianBlur(seam.astype(np.float32), (0, 0), 2.5)[..., None]
        seam_colour = np.where((den_in > 0.05)[..., None], mean_in, hair_img)
        if dbg is not None:
            write_gap_debug(Path(debug_dir), debug_name, photo, new_hair, photo_face | fh_zone, dbg)

    if debug_dir is not None:
        write_owner_debug(Path(debug_dir), debug_name, photo,
                          {"hair": new_hair, "face": photo_face, "forehead": fh_zone, "skin": skin_fill & ~fh_zone, "plate": plate_fill},
                          {"left_temple": (150, 500, 350, 700), "right_temple": (800, 500, 1000, 700), "right_brow_end": (860, 540, 1000, 660)})
    out = photo.astype(np.float32)
    a = feather(plate_fill)
    out = out * (1 - a) + plate.astype(np.float32) * a
    # opaque inside (a plain sigma 8 blur left small replaced patches only partly opaque, so strands showed through), soft at the edge
    fs, fk = (PATCH_FEATHER, 1.3) if partial else (FOREHEAD_FEATHER, 1.6)   # patches next to real skin get a longer, softer edge
    a_fh = np.clip(fk * cv2.GaussianBlur(fh_zone.astype(np.float32), (0, 0), fs), 0, 1)[..., None]
    # the border to the brows: bald-skin alpha ramps up with the DISTANCE from the brows (smoothstep over BROW_RAMP px), so the
    # boundary follows the brows' arch instead of a straight line cutting across it, and the brows themselves stay the photo's
    # whole-zone mode: a short ramp, so the bald skin reaches the brow and replaces the fringe's shadow strip instead of mixing with it (a
    # long ramp left photo and synthetic skin 50/50 for ~20 px: a darker, mottled band above the brows)
    t = np.clip((brow_dist - 2.0) / (BROW_RAMP if partial else BROW_EDGE), 0, 1)
    ramp = (t * t * (3 - 2 * t)).astype(np.float32)
    a_fh = a_fh * ramp[..., None]
    if not partial and fh_cand.any():
        # the zone's lower edge is a straight line at the brow-centre row; past the outer ends of the brows it showed as a straight edge
        # with a lighter rectangle above it. There, fade the alpha out over ZONE_FADE rows towards that edge
        rows_ = np.arange(h, dtype=np.float32)[:, None]
        bottom = (fh_cand * rows_).max(axis=0)[None, :]                            # lowest zone row per column
        s = np.clip((bottom - rows_) / ZONE_FADE, 0, 1)
        s = s * s * (3 - 2 * s)
        # not right above a brow (there the brow ramp already shapes the edge, and fading would bring the shadow strip back): everywhere
        # else, including between the brows and past their ends, the straight bottom edge fades out
        brow_cols = (brows.max(axis=0) > 0).astype(np.uint8)[None, :]
        col_dist = cv2.distanceTransform(1 - brow_cols, cv2.DIST_L2, 3)             # horizontal distance to the nearest brow column
        wmix = np.clip((col_dist - 6.0) / 10.0, 0, 1)
        a_fh = a_fh * (1 - wmix * (1 - s))[..., None]
    # the forehead's alpha is a_fh alone (soft, brow ramp); the rest of the skin fill gets a soft edge (sigma 4) that may bleed a few px into
    # the real skin, where the boundary tone match makes it invisible
    # wherever the photo has old hair, the skin that replaces it must be fully opaque: a soft edge or the bottom-edge fade there lets the
    # dark old hair show through as a smear (probe: alpha 0.10-0.56 over RGB ~(20,15,15) at the right brow end)
    on_hair = (cv2.dilate(old_bin, ell(3)) > 0) | np.isin(lab_photo, [HAIR, 18]) | (cv2.dilate(strands.astype(np.uint8), ell(3)) > 0)
    a_other = cv2.GaussianBlur(other.astype(np.float32), (0, 0), 4.0)[..., None]
    if partial:
        a_fh = np.maximum(a_fh, (fh_zone & on_hair)[..., None] * ramp[..., None])
        a_other = np.maximum(a_other, (other & on_hair)[..., None].astype(np.float32))
    else:
        # whole-zone mode: the opaque-over-old-hair region has a soft outline (a hard outline showed as stepped blocks beyond the brow ends);
        # grown first so that the soft edge falls on skin, not on hair
        soft = lambda m: np.clip(1.4 * cv2.GaussianBlur(cv2.dilate(m.astype(np.uint8), ell(4)).astype(np.float32), (0, 0), 5.0), 0, 1)  # noqa: E731
        oh_f, oh_o = soft(fh_zone & on_hair), soft(other & on_hair)
        # the old hair is covered by an UNDERLAY of the hairstyle GAN's skin, under the (faded) bald forehead: making the bald skin itself
        # opaque over hair ended in the zone's straight bottom edge and showed as a block past the brow ends; with the underlay the bald
        # skin can fade out smoothly and what shows through it is skin, not dark hair
        a_other = np.maximum(a_other, np.maximum(oh_o, oh_f)[..., None])
        # the zone's straight side edge (the face contour) fades in over SIDE_SOFT px
        cols_s = np.arange(w, dtype=np.float32)[None, :]
        ed = np.nan_to_num(np.minimum(cols_s - xl_c[:, None], xr_c[:, None] - cols_s), nan=0.0)
        sd = np.clip(ed / SIDE_SOFT, 0, 1)
        sd = sd * sd * (3 - 2 * sd)
        a_fh = a_fh * sd[..., None]
    a_sk = np.maximum(a_other, a_fh)
    if not partial:
        # the underlay under the forehead takes the bald skin's tone (the GAN skin cannot reproduce the old fringe's shadow, and faded
        # through the bald skin it showed as a grey patch left of the brow start)
        wu = (UNDERLAY_BALD * np.clip(1.5 * cv2.GaussianBlur(fh_zone.astype(np.float32), (0, 0), 6.0), 0, 1))[..., None]
        skin_h = skin_h.astype(np.float32) * (1 - wu) + gan_skin.astype(np.float32) * wu
    out = out * (1 - a_other) + skin_h.astype(np.float32) * a_other      # temples, ears, neck: the hairstyle GAN's skin
    out = out * (1 - a_fh) + gan_skin.astype(np.float32) * a_fh          # forehead on top: the bald face's skin
    if not partial and yb is not None:
        # the brows' fine top hairs and their soft shadow lie outside the BiSeNet brow label and were replaced by smooth skin, so the brows
        # looked pasted on. Carry the photo's dark fine detail (closing - gray: hairs and shadow are dark, skin is smooth) over the new skin in
        # a band above each brow, but never the old fringe's hair
        g = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dark = np.clip(cv2.morphologyEx(g, cv2.MORPH_CLOSE, ell(4)) - g - 3.0, 0, 45.0)    # capped: a fringe's shadow is a big dark area, not a hair
        # only thin hairs go over, not the brow's highlight edge (it left pale lumpy blobs); then softened a little
        thin = cv2.dilate((cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))) > 6).astype(np.uint8),
                          np.ones((3, 3), np.uint8)).astype(np.float32)
        dark = cv2.GaussianBlur(dark * thin, (0, 0), 0.8)
        bm = brows > 0
        has_b = bm.any(axis=0)
        top_b = np.where(has_b, bm.argmax(axis=0), 0)[None, :]
        above = has_b[None, :] & (np.arange(h)[:, None] < top_b + 4)
        band_w = np.clip(1.0 - (brow_dist - 2.0) / BROW_BAND, 0, 1) * above * ~(cv2.dilate(old_bin, ell(BROW_FRINGE)) > 0)
        out = out - (dark * band_w * a_fh[..., 0] * BROW_HAIR_GAIN)[..., None]
        out = out * (1 - (BROW_SHADOW * np.exp(-brow_dist / 7.0) * above * a_fh[..., 0])[..., None])
    # contact shadow: skin darkens a little right under the hair edge (only where filled skin was composited, never the real face)
    shade = SHADE_STRENGTH * cv2.GaussianBlur(a_hair.astype(np.float32), (0, 0), SHADE_SIGMA)[..., None]
    out = out * (1 - a_sk * shade)
    if a_seam is not None:
        out = out * (1 - a_seam) + seam_colour * a_seam
    out = out * (1 - a_hair[..., None]) + hair_img * a_hair[..., None]
    if not partial:
        # light edge-aware smoothing only where replaced skin blends into the photo (the seams); texture elsewhere is untouched
        seam_w = ((np.maximum(a_fh[..., 0], a_other[..., 0]) > 0.05) & (np.maximum(a_fh[..., 0], a_other[..., 0]) < 0.95)).astype(np.float32)
        seam_w = cv2.GaussianBlur(seam_w, (0, 0), 3.0)[..., None]
        sm = cv2.bilateralFilter(np.clip(out, 0, 255).astype(np.uint8), 7, SEAM_BILATERAL, 5).astype(np.float32)
        out = out * (1 - 0.6 * seam_w) + sm * (0.6 * seam_w)
    if debug_dir is not None and w == 1152:
        # which layer owns a pixel, and with what weight: read a smear's cause off the numbers instead of guessing
        for px, py in ((922, 560), (926, 575), (930, 590)):
            print(f"   probe ({px},{py}): lab={int(lab[py, px])} lab_photo={int(lab_photo[py, px])} photo_face={bool(photo_face[py, px])} "
                  f"fh_zone={bool(fh_zone[py, px])} skin_fill={bool(skin_fill[py, px])} plate_fill={bool(plate_fill[py, px])} "
                  f"halo={bool(halo[py, px])} old={bool(old_bin[py, px])} new_hair={bool(new_hair[py, px])} a_hair={float(a_hair[py, px]):.2f} "
                  f"a_fh={float(a_fh[py, px, 0]):.2f} a_other={float(a_other[py, px, 0]):.2f} "
                  f"seam={0.0 if a_seam is None else float(a_seam[py, px, 0]):.2f} "
                  f"photo={photo[py, px].tolist()} skin_bald={gan_skin[py, px].tolist()} skin_gan={skin_h[py, px].tolist()} "
                  f"plate={plate[py, px].astype(int).tolist()} out={out[py, px].astype(int).tolist()}", flush=True)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="outputs/<model> folder, e.g. hairfast")
    ap.add_argument("--raw-dir", default="raw", help="subfolder of outputs/<model> with the raw 1024 crops")
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--outputs", default="/work/outputs")
    ap.add_argument("--lama", default="/work/models/lama/big-lama.pt")
    ap.add_argument("--models", default="/work/models/hairfast", help="folder with pretrained_models/ (dlib landmarks for the face warp)")
    ap.add_argument("--no-warp", action="store_true", help="skip the landmark warp of the generated crops onto the photo's face")
    ap.add_argument("--debug-dir", help="write the hair-snap overlay and per-row table for each job to this folder")
    ap.add_argument("--photos", help="comma list of photo ids to restrict to")
    ap.add_argument("--force", action="store_true", help="redo images that already have a final")
    ap.add_argument("--skin-src", choices=["gan", "bald"], default="gan",
                    help="where exposed skin comes from: the hairstyle GAN's own output, or the bald-proxy faces in "
                         "outputs/hairclipv2/bald (hairclipv2/run.py --bald); results go to final_bald")
    args = ap.parse_args()

    inputs = Path(args.inputs)
    root = Path(args.outputs) / args.model
    raw_dir = root / args.raw_dir
    final = root / ("final" if args.raw_dir == "raw" else f"final_{args.raw_dir}")
    bald_dir = Path(args.outputs) / "hairclipv2" / "bald"
    if args.skin_src == "bald":
        final = root / "final_bald"
    final.mkdir(parents=True, exist_ok=True)

    names = sorted(p.stem for p in raw_dir.glob("*.png"))
    if args.photos:
        keep = set(args.photos.split(","))
        names = [n for n in names if n.split("__")[0] in keep]
    if args.skin_src == "bald":
        names = [n for n in names if (bald_dir / f"{n.split('__')[0]}.png").exists()]
    todo = [n for n in names if args.force or not (final / f"{n}.png").exists()]
    print(f"{len(todo)} to finish ({len(names)} raw)", flush=True)
    if not todo:
        return

    parser = Parser()
    lama = Lama(args.lama)
    aligner = None if args.no_warp else Aligner(str(Path(args.models) / "pretrained_models" / "ShapeAdaptor" /
                                                    "shape_predictor_68_face_landmarks.dat"))
    for name in todo:
        pid = name.split("__")[0]
        t0 = time.time()
        photo = np.asarray(Image.open(inputs / f"{pid}.jpg").convert("RGB"))
        old_mask = np.asarray(Image.open(inputs / "masks" / f"{pid}.png").convert("L"))
        raw = np.asarray(Image.open(raw_dir / f"{name}.png").convert("RGB"))
        m = np.load(inputs / "aligned" / f"{pid}.npy")
        face_crop = np.asarray(Image.open(inputs / "aligned" / f"{pid}.png").convert("RGB"))
        skin_crop = np.asarray(Image.open(bald_dir / f"{pid}.png").convert("RGB")) if args.skin_src == "bald" else None
        style = name.split("__")[1] if "__" in name else ""   # job ids are <photo>__<style>[__sweep tag]
        res = finish_one(photo, old_mask, raw, face_crop, m, parser, lama, skin_crop,
                         stubble=style in STUBBLE_STYLES, seed=zlib.crc32(name.encode()), aligner=aligner,
                         debug_dir=args.debug_dir, debug_name=name)
        Image.fromarray(res).save(final / f"{name}.png")
        print(f"{name}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
