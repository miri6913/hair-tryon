# Hairstyle model comparison (Phase 2, step 1)

Runs on the GPU server `<host>` from this PC via `scripts\hs` (cmd wrapper for `scripts/hairstyle_sync.py`). The code here is the source of truth; the
server's `code/` folder is an overwritten copy (`push`). Server root: `<server root>`
(`code/ inputs/ outputs/ models/ hf-cache/`).

**Status:** Qwen (noisy) and Stable-Hair were judged unusable; Stable-Hair was never run. The GAN models (`hairfast`, `hairclipv2`) share
`common/`: `align.py` (FFHQ crops + framing check) -> model runner (raw 1024 crops) -> `finish.py` (BiSeNet parse, LaMa clean plate,
composite onto the original photo). HairFast runs end to end; HairCLIPv2 is built but not yet run.

| Output name | Model | What it gets | Code / weights license |
| --- | --- | --- | --- |
| `qwen_text` | Qwen-Image-Edit-2511 | photo + text prompt | Apache-2.0 |
| `qwen_ref` | Qwen-Image-Edit-2511 | photo + reference hairstyle photo (2 images) | Apache-2.0 |
| `stable_hair` | Stable-Hair (SD 1.5 + ControlNet) | photo + reference hairstyle photo | code Apache-2.0; SD 1.5 CreativeML OpenRAIL-M; checkpoints from the authors' Google Drive |
| `flux2_4b_text` / `flux2_4b_ref` | FLUX.2 [klein] 4B (4 steps) | photo + prompt / photo + reference photo | Apache-2.0 |
| `flux2_9b_text` / `flux2_9b_ref` | FLUX.2 [klein] 9B | same | FLUX Non-Commercial, gated on HF (accept the licence, set `HF_TOKEN` in the server shell): quality reference only |
| `hairfast` | HairFastGAN (StyleGAN2, 1024 px FFHQ crops) | photo + reference hairstyle photo (shape from the reference, colour kept from the photo) | code MIT; weights have non-commercial parts, see `hairfast/WEIGHTS.md` |
| `hairclipv2` | HairCLIPv2 (StyleGAN2, optimisation-based) | same inputs as hairfast | **no licence file (all rights reserved)**: research comparison only |

HairFusion was excluded: CC BY-NC-SA 4.0 (non-commercial), Python 3.8 / torch 2.0 pins, DensePose preprocessing.

Notes
- Qwen-Image-Edit is ~20B (transformer ~40 GB in bf16). Whole-model CPU offload alone ran out of memory on a 24 GB 3090 (first smoke test), so `run.py` defaults to a 4-bit NF4 transformer (~11 GB) plus `enable_model_cpu_offload()` for the text encoder. A small quality loss versus bf16 is possible; `--quant none` needs more GPU memory than one 24 GB card has.
- Stable-Hair resizes to 512x512: the runner edge-pads photos to a square and crops the result back (512 px long side).
- `runwayml/stable-diffusion-v1-5` named in Stable-Hair's config no longer exists; `fetch_weights.sh` uses the `stable-diffusion-v1-5/stable-diffusion-v1-5` mirror.
- Google Drive may refuse the Stable-Hair download (quota). Then download the folder by hand into `models/stable_hair/` (it must contain `stage1/` and `stage2/`).
- `run` needs an explicit `-g` (GPU number) and refuses a GPU that already holds more than 2 GB: the server is shared.

- GAN models only handle whole heads: `hs align` writes `inputs/aligned/framing.csv` and flags photos whose hair touches the top edge
  (`photo_c`); the runners skip them. The app's capture guide must keep the head in frame with headroom.
- The 1024 crop ends near the chin, so hair longer than that is cut off by a straight line (`long_straight`). Pick references whose hair fits.
- Docker builds: the context is `code/` (`-f <model>/Dockerfile`) so images can `COPY common`. Build gotchas fixed in the Dockerfiles:
  dlib must be built without CUDA, `CUDA_HOME=/usr/local/cuda` (torch finds conda's partial nvcc otherwise), ninja for StyleGAN2's ops.

Order (qwen / stable_hair): `push` -> `build` -> (`fetch-weights`) -> `run -l 2` (smoke test) -> `run` -> `pull` ->
`python scripts/hairstyle_composite.py` -> `python scripts/hairstyle_sheet.py`.

Order (GAN models): `push` -> `build -m hairfast` -> `fetch-weights` -> `align` -> `run -m <model> -g N` -> `finish -m <model> -g N` -> `pull`
-> `python scripts/hairstyle_sheet.py --models hairfast,hairclipv2`.
