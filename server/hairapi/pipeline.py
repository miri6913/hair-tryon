"""Hairstyle pipeline: photo + style id -> the photo with the new hairstyle.

Default (REF_MODE=text): FLUX.2 [klein] 4B edits the photo from a fixed text prompt per style, no reference photo. The person's
hair colour is measured on the photo and named in the prompt (black ... blonde), so the new style keeps their colour. The model's
output is returned as is (at the model's size, ~768x1024); the whole image is a re-draw, so the face and background can drift
slightly from the photo and skin gets smoothed.
REF_MODE=ref: the older route, a reference photo of the style as the second image (hair colour then comes from the reference).
COMPOSITE=1: paste-back path (own hair colour + composite onto the original, scripts/hairstyle_composite.py --keep-color), kept for
comparison; the user judged its results worse than the raw output. Not thread-safe: the app runs it on one worker thread.
"""
import gc
import io
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image, ImageOps

from vto_spike.composite import composite
from vto_spike.recolor2 import HAIR_COLOURS, Shade, colour_word, measure_hair_lightness, measure_shade, recolor_hair2
from vto_spike.segment import HAIR_MODEL, HairSegmenter, SelfieParser

MODEL = "black-forest-labs/FLUX.2-klein-4B"
# The style is named in words as well as shown: with the reference photo alone the length/shape often does not transfer
# (a buzz cut or pixie came out as medium hair); the prompt of each style is encoded once at startup.
REF_PROMPT = (
    "Image 1 is the person. Image 2 shows the target hairstyle: {hair}. Change the hair of the person in image 1 to exactly "
    "this hairstyle (cut, length, volume and parting) as in image 2, replacing their current hair completely. Keep the "
    "face, skin, expression, clothes, pose and background of image 1 exactly the same."
)
# REF_MODE=text: no reference photo at all, one fixed prompt per style (photo + prompt in, photo out). Keeps the person's own
# hair colour because the prompt says so (a reference photo would hand over its own colour).
#
# The colour is NAMED, from the person's measured hair lightness: "keep the current colour" let black hair come out with blond
# highlights, and "no highlights, no dye" made it worse (negations seem to prime the model). One embedding per (style, colour).
TEXT_PROMPT = (
    "Change only the hairstyle of the person in the photo to {hair}, in {colour} hair colour. Keep the face, skin texture, "
    "expression, clothes, pose, lighting and background exactly the same. Photorealistic."
)
# colour_word / HAIR_COLOURS live in vto_spike.recolor2 (exposure-corrected lower-quartile lightness, so bright light does not turn
# black hair into "brown"). The words below are what the prompt says. "black" is "very dark brown, almost black": "jet black" came out as a flat blue-black,
# this keeps natural depth and a soft sheen (a natural look).
COLOUR_PHRASE = {"black": "very dark brown, almost black", "dark brown": "dark brown", "brown": "brown", "light brown": "light brown", "blonde": "blonde"}
COLOUR_OVERRIDE = {"brown": "black", "light brown": "black"}   # temporary: black hair under bright/warm light measures as brown (until the app has a colour picker)
COLOUR_WORDS = [w for _, w in HAIR_COLOURS]
# After generation the hair is darkened (never lightened) back to the photo's own darkness when the model painted it lighter.
HAIR_TONE = os.environ.get("HAIR_TONE", "1") == "1"
TONE_STRENGTH = float(os.environ.get("TONE_STRENGTH", "0.8"))
TONE_TOLERANCE = 3.0      # Lab L the output hair may be lighter than the target before it is corrected
REF_MODE = os.environ.get("REF_MODE", "text")   # "text" (prompt only, the default) or "ref" (reference photo + prompt)
PHOTO_MAX_SIDE = 1536     # the working size of the photo, like scripts/prepare_hairstyle_set.py
MODEL_MAX_SIDE = 1024     # longest side fed to FLUX (what the benchmark runs used)
STEPS = 4                 # klein is step-distilled: 4 steps, guidance 1.0
SEED = 0
FACE_MIN_FRAC = 0.01      # selfie-multiclass face-skin share of the photo below which we call it "no face"
FACE_OTHER_REL = 0.25     # another face-skin blob at least this share of the largest one counts as a second face
FACE_MERGE_FRAC = 0.03    # face-skin blobs closer than this share of the longest side are one face (glasses frames split the skin)
COMPOSITE = os.environ.get("COMPOSITE", "0") == "1"   # default off: return the raw model output
Image.MAX_IMAGE_PIXELS = 40_000_000


class PhotoError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def fit(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    return img


class HairPipeline:
    def __init__(self, catalog_path: Path, inputs_dir: Path, device: str = "cuda"):
        self.device = device
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.version = catalog["version"]
        self.styles = {s["id"]: s for s in catalog["styles"]}
        self.refs: dict[str, Image.Image] = {}
        self.thumbs: dict[str, bytes] = {}
        for sid, s in self.styles.items():
            if REF_MODE == "text":
                continue   # no reference photos in text mode (so no thumbnails either)
            ref = ImageOps.exif_transpose(Image.open(inputs_dir / s["reference"])).convert("RGB")
            self.refs[sid] = fit(ref, MODEL_MAX_SIDE)
            buf = io.BytesIO()
            fit(ref, 320).save(buf, "JPEG", quality=85)
            self.thumbs[sid] = buf.getvalue()

        self.hair_seg = HairSegmenter(HAIR_MODEL)
        self.parser = SelfieParser()
        pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        pipe.to(device)
        self.embeds: dict[tuple[str, str | None], torch.Tensor] = {}   # (style id, colour word or None) -> prompt embedding
        for sid, s in self.styles.items():
            for colour in (COLOUR_WORDS if REF_MODE == "text" else [None]):
                prompt = (TEXT_PROMPT.format(hair=s["hair_prompt"], colour=COLOUR_PHRASE[colour]) if REF_MODE == "text"
                          else REF_PROMPT.format(hair=s["hair_prompt"]))
                with torch.no_grad():   # without no_grad the result keeps an autograd graph that pins the text encoder on the GPU
                    enc = pipe.encode_prompt(prompt, device=device)
                self.embeds[(sid, colour)] = (enc[0] if isinstance(enc, tuple) else enc).detach().clone()
                del enc
        pipe.text_encoder.to("cpu")   # the prompts are fixed per style: the encoder is never needed per request (frees ~7.7 GB)
        gc.collect()
        torch.cuda.empty_cache()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    # -- photo handling ---------------------------------------------------------------------------------------------
    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        except Exception as e:   # noqa: BLE001  (PIL raises many types for bad input)
            raise PhotoError("bad_image", f"cannot read the image: {type(e).__name__}") from e
        return np.asarray(fit(img, PHOTO_MAX_SIDE))

    def check(self, rgb: np.ndarray) -> list[str]:
        """Raises PhotoError for no face / several faces; returns warnings (the photo is still usable)."""
        hair, face = self.parser(rgb)
        face_mask = face > 0.5
        if face_mask.mean() < FACE_MIN_FRAC:
            raise PhotoError("no_face", "no face found: show the whole head facing the camera")
        # Glasses frames (and hair strands) cut one face's skin into several blobs: join blobs closer than FACE_MERGE_FRAC of the
        # photo before counting, but measure each merged blob by its real skin pixels so the thresholds keep their meaning.
        r = max(3, round(FACE_MERGE_FRAC * max(face_mask.shape)))
        merged = cv2.dilate(face_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        n, labels = cv2.connectedComponents(merged, connectivity=8)
        areas = np.bincount(labels[face_mask], minlength=n)[1:]
        if (areas >= max(areas.max() * FACE_OTHER_REL, FACE_MIN_FRAC * face_mask.size)).sum() > 1:
            raise PhotoError("multiple_faces", "more than one face found: only the person to restyle should be in the photo")
        warnings = []
        h = rgb.shape[0]
        if (hair[: max(2, h // 100)] > 0.5).any():
            warnings.append("head_cut_off")   # hair touches the top edge: leave headroom; the model still handles it
        return warnings

    def match_darkness(self, out: Image.Image, meas: dict | None) -> Image.Image:
        """Darken the generated hair back to the photo's darkness when the model painted it lighter (never lightens). Only the hair
        lightness is touched (recolor_hair2 on the raw output, no paste-back); the model's own hue is kept. The target is the photo's
        lower-quartile hair L, scaled by how much the model changed the face skin's brightness (a global exposure change is not drift)."""
        if meas is None or meas["p25"] <= 0:
            return out
        arr = np.asarray(out)
        mask = self.hair_seg(arr)
        mo = measure_hair_lightness(arr, mask, self.parser(arr)[1])
        shade = measure_shade(arr, mask)
        if mo is None or shade is None or mo["p25"] <= 0:
            return out
        target = meas["p25"] * (mo["skin"] / meas["skin"] if mo["skin"] and meas["skin"] else 1.0)
        gap = mo["p25"] - target
        print(f"hair tone: photo p25 {meas['p25']:.1f}, output p25 {mo['p25']:.1f}, target {target:.1f}"
              f"{' -> darken' if gap > TONE_TOLERANCE else ''}", flush=True)
        if gap <= TONE_TOLERANCE:
            return out
        k = target / mo["p25"]   # multiplicative: keeps the quantiles increasing and the darkest hair near black
        darker = Shade(L=tuple(v * k for v in shade.L), ab_shadow=shade.ab_shadow, ab_mid=shade.ab_mid, ab_high=shade.ab_high)
        return Image.fromarray(recolor_hair2(arr, mask, darker, intensity=TONE_STRENGTH))

    # -- the edit ---------------------------------------------------------------------------------------------------
    def run(self, rgb: np.ndarray, style_id: str) -> tuple[bytes, dict[str, float]]:
        t = {}
        colour = None
        if COMPOSITE or REF_MODE == "text":
            t0 = time.time()
            old_mask = self.hair_seg(rgb)
            shade = measure_shade(rgb, old_mask)   # None when there is too little original hair
            meas = measure_hair_lightness(rgb, old_mask, self.parser(rgb)[1])   # None when there is too little original hair
            if REF_MODE == "text":
                colour = COLOUR_OVERRIDE.get(colour_word(meas), colour_word(meas))
            t["seg_old"] = time.time() - t0

        t0 = time.time()
        photo = fit(Image.fromarray(rgb), MODEL_MAX_SIDE)
        w, h = photo.size
        out = self.pipe(
            image=photo if REF_MODE == "text" else [photo, self.refs[style_id]], prompt_embeds=self.embeds[(style_id, colour)],
            width=w // 16 * 16, height=h // 16 * 16, guidance_scale=1.0, num_inference_steps=STEPS,
            generator=torch.Generator(device=self.device).manual_seed(SEED),
        ).images[0]
        t["generate"] = time.time() - t0

        t0 = time.time()
        buf = io.BytesIO()
        if not COMPOSITE:
            if HAIR_TONE and REF_MODE == "text":
                out = self.match_darkness(out, meas)
            out.save(buf, "JPEG", quality=90)
        else:
            out = np.asarray(out.resize((rgb.shape[1], rgb.shape[0]), Image.LANCZOS))
            new_mask = self.hair_seg(out)
            face_out = self.parser(out)[1]   # face skin of the model output: where the revealed forehead is, for the tone match
            if shade is not None:
                out = recolor_hair2(out, new_mask, shade)
            comp = composite(rgb.astype(np.float32), old_mask, out.astype(np.float32), new_mask, face_out)
            Image.fromarray(comp).save(buf, "JPEG", quality=90)
        t["post"] = time.time() - t0
        if colour:
            print(f"hair colour word: {colour}", flush=True)   # a category, not personal data: for judging the measurement
        return buf.getvalue(), t
