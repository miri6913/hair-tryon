"""HairFastGAN wrapper: an aligned 1024 face crop plus an aligned reference crop -> the face with the reference's hair shape.

Alignment lives in common/align.py and compositing back into the photo in common/finish.py, so every GAN model sees identical
inputs and goes through identical compositing. Shape comes from the reference, colour from the face itself (hair colour is kept).
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/opt/HairFastGAN")


class HairStyler:
    def __init__(self, models: str = "/work/models/hairfast", device: str = "cuda", smooth: int = 5, mixing: float = 0.95):
        # the repo reads its weights from ./pretrained_models and imports itself from the working directory
        link = REPO / "pretrained_models"
        if not link.exists():
            link.symlink_to(Path(models) / "pretrained_models")
        os.chdir(REPO)
        sys.path.insert(0, str(REPO))
        from hair_swap import HairFast, get_parser  # noqa: E402

        self.model = HairFast(get_parser().parse_args(["--device", device, "--smooth", str(smooth), "--mixing", str(mixing)]))

    @torch.no_grad()
    def swap(self, face: np.ndarray, shape: np.ndarray) -> np.ndarray:
        """face, shape: aligned 1024x1024 RGB uint8. Returns the 1024x1024 RGB uint8 result."""
        out = self.model(face, shape, face)  # face, shape, colour
        if isinstance(out, tuple):
            out = out[0]
        res = out.detach().float().cpu().squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()
        return (res * 255).round().astype(np.uint8)
