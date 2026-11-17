"""FFHQ face alignment shared by every GAN model runner and the finish stage.

A photo becomes a 1024x1024 aligned crop plus the 3x3 matrix M (photo -> crop), so a 1024 px result can be warped back into the
photo. Each photo also gets a framing check: GAN models see whole heads, and a selfie whose head top is cut off by the frame
cannot be handled by the crop (the missing part would have to be invented), so it is flagged and the app's capture guide must
prevent it.

CLI (inside the container):  python /code/common/align.py [--inputs /work/inputs] [--models /work/models/hairfast]
  reads inputs/hairstyle_set.csv; writes inputs/aligned/<id>.png, <id>.npy (M) and framing.csv
"""
import argparse
import csv
from pathlib import Path

import cv2
import dlib
import numpy as np
from PIL import Image

SIZE = 1024
MAX_OUTSIDE = 0.30      # share of the crop square that may fall outside the photo (overflow at the sides/bottom is edge-padded)
MIN_HEADROOM = 0.02     # hair must start at least this fraction of the photo height below the top edge


class NoFaceError(Exception):
    pass


def ffhq_quad(lm: np.ndarray) -> np.ndarray:
    """Oriented square (4x2, clockwise from top-left) the FFHQ dataset crops around a face; from dlib 68 landmarks."""
    eye_left = lm[36:42].mean(axis=0)
    eye_right = lm[42:48].mean(axis=0)
    eye_avg = (eye_left + eye_right) / 2
    eye_to_eye = eye_right - eye_left
    mouth = lm[48:60]
    mouth_avg = (mouth[0] + mouth[6]) / 2
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(np.float32)


class Aligner:
    def __init__(self, predictor_path: str):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def quad(self, rgb: np.ndarray) -> np.ndarray:
        faces = self.detector(rgb, 1)
        if not faces:
            raise NoFaceError("no face found")
        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self.predictor(rgb, face)
        return ffhq_quad(np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float64))

    def align(self, rgb: np.ndarray, hair_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """-> (1024x1024 crop, 3x3 matrix photo -> crop, framing info). Raises NoFaceError."""
        h, w = rgb.shape[:2]
        quad = self.quad(rgb)
        info = framing(quad, h, w, hair_mask)

        # where the square leaves the photo, extend the photo by its edge pixels (softened) instead of mirroring it
        pad = int(np.ceil(max(0.0, -quad.min(), quad[:, 0].max() - w, quad[:, 1].max() - h))) + 2
        padded = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        blur = cv2.GaussianBlur(padded, (0, 0), max(3.0, pad / 8))
        inside = np.zeros(padded.shape[:2], np.uint8)
        inside[pad:pad + h, pad:pad + w] = 1
        feather = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 2)[..., None]
        padded = (padded * feather + blur * (1 - feather)).astype(np.uint8)

        dst = np.array([[0, 0], [0, SIZE], [SIZE, SIZE], [SIZE, 0]], dtype=np.float32)
        m_pad = cv2.getPerspectiveTransform(quad + pad, dst)
        crop = cv2.warpPerspective(padded, m_pad, (SIZE, SIZE), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        shift = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], dtype=np.float64)
        return crop, m_pad @ shift, info


def framing(quad: np.ndarray, h: int, w: int, hair_mask: np.ndarray | None) -> dict:
    area = cv2.contourArea(quad)
    canvas = np.zeros((h, w), np.uint8)
    cv2.fillPoly(canvas, [np.round(quad).astype(np.int32)], 1)
    outside = max(0.0, 1 - canvas.sum() / max(area, 1.0))
    info = {"quad_side_px": round(float(np.sqrt(area))), "quad_outside": round(float(outside), 3),
            "hair_top_row": "", "flagged": False, "reason": ""}
    reasons = []
    if outside > MAX_OUTSIDE:
        reasons.append(f"crop {outside:.0%} outside photo")
    if hair_mask is not None and (hair_mask > 0.5).any():
        top = int(np.where((hair_mask > 0.5).any(axis=1))[0][0])
        info["hair_top_row"] = top
        if top < MIN_HEADROOM * h:
            reasons.append("hair touches the top edge")
    info["flagged"] = bool(reasons)
    info["reason"] = "; ".join(reasons)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--models", default="/work/models/hairfast")
    args = ap.parse_args()

    inputs = Path(args.inputs)
    out = inputs / "aligned"
    out.mkdir(exist_ok=True)
    aligner = Aligner(str(Path(args.models) / "pretrained_models" / "ShapeAdaptor" / "shape_predictor_68_face_landmarks.dat"))
    with open(inputs / "hairstyle_set.csv", encoding="utf-8") as f:
        ids = [r["id"] for r in csv.DictReader(f)]

    cols = ["id", "quad_side_px", "quad_outside", "hair_top_row", "flagged", "reason"]
    with open(out / "framing.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        for pid in ids:
            rgb = np.asarray(Image.open(inputs / f"{pid}.jpg").convert("RGB"))
            mask = np.asarray(Image.open(inputs / "masks" / f"{pid}.png"), dtype=np.float32) / 255
            try:
                crop, m, info = aligner.align(rgb, mask)
            except NoFaceError:
                wr.writerow({"id": pid, "flagged": True, "reason": "no face"})
                print(f"{pid}: no face")
                continue
            Image.fromarray(crop).save(out / f"{pid}.png")
            np.save(out / f"{pid}.npy", m)
            wr.writerow({"id": pid, **info})
            print(f"{pid}: outside={info['quad_outside']:.2f} {'FLAGGED: ' + info['reason'] if info['flagged'] else 'ok'}")


if __name__ == "__main__":
    main()
