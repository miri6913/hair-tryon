# HairFastGAN weights and licences

Code: https://github.com/AIRI-Institute/HairFastGAN (MIT). Weights: Hugging Face `AIRI-Institute/HairFastGAN`, folder `pretrained_models/`.

**Not yet audited.** After the first `fetch-weights`, list the files that `fetch_weights.sh` prints and fill in the table. Known or
suspected restrictions, to settle before any commercial use:

| Component | Weight file (to confirm) | Licence concern |
| --- | --- | --- |
| StyleGAN2 FFHQ generator | `StyleGAN/ffhq.pt` | derived from NVIDIA StyleGAN2 / FFHQ: non-commercial terms (NVIDIA Source Code License-NC; FFHQ images carry their own licences) |
| dlib 68-point landmarks | `ShapeAdaptor/shape_predictor_68_face_landmarks.dat` | trained on iBUG 300-W: non-commercial |
| Encoders, rotate / blending / post-process nets | `FeatureStyleEncoder/`, `Rotate/`, `Blending/`, `PostProcess/` | trained by the authors on FFHQ/CelebA-HQ; check the HF model card |
| Face parser | `BiSeNet/` | CelebAMask-HQ: research use |

Fine for the staging comparison. If the model is chosen for production, either obtain written terms or swap in permissive parts
(e.g. MediaPipe landmarks instead of dlib).
