"""MediaPipe hair segmentation wrapper. Returns a float32 hair confidence mask in [0, 1]."""
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
HAIR_MODEL = "hair_segmenter.tflite"
MULTICLASS_MODEL = "selfie_multiclass_256x256.tflite"
HAIR_CLASS_INDEX = 1  # both models: 0 = background, 1 = hair


class HairSegmenter:
    def __init__(self, model_name: str = HAIR_MODEL):
        options = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODELS_DIR / model_name)),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: uint8 HxWx3. Returns float32 HxW hair confidence at the input resolution."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._segmenter.segment(image)
        view = result.confidence_masks[HAIR_CLASS_INDEX].numpy_view()
        return np.squeeze(view).astype(np.float32).copy()

    def close(self) -> None:
        self._segmenter.close()


class SelfieParser:
    """MediaPipe selfie-multiclass: hair and face-skin confidence masks (0 = background, 1 = hair, 2 = body skin, 3 = face skin)."""

    HAIR, FACE = 1, 3

    def __init__(self) -> None:
        options = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODELS_DIR / MULTICLASS_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """rgb: uint8 HxWx3. Returns (hair, face_skin) float32 HxW at the input resolution."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        masks = self._segmenter.segment(image).confidence_masks
        get = lambda i: np.squeeze(masks[i].numpy_view()).astype(np.float32).copy()  # noqa: E731
        return get(self.HAIR), get(self.FACE)

    def close(self) -> None:
        self._segmenter.close()
