# Phase 0 report (interim) — 2026-10-08

Test set so far: 150 frontal selfie-style photos (Kaggle TrainingData age-detection set, resized to 1024 px).
Not yet covered: curly/coily/braided hair, hats, glasses, strong dyes (Wikimedia gap-fill pending).

## Masks (MediaPipe, CPU, Windows)
- Hair-only model vs multiclass selfie model: median IoU 0.88, so they largely agree.
- Hair-only model: median 39 ms/image on desktop CPU at 1024 px (not a phone number).
- Auto-flags: 21 with hair area < 2% (mostly short/buzzed hair, likely OK, needs eye check), 1 with area > 45%.
- Eye review of 12 photos (sheets 00 and 08): edges follow long straight hair and flyaways well; guided-filter refinement tightens edges slightly. **This is not the full 1–5 scoring**: that is yours to do in `results/mask_scores.csv` (columns `score_hair`, `score_multi`).

## Hard cases (26 Wikimedia photos, sheets 25–29; Claude's eye review, please confirm)
Roughly 15 good, 11 with problems.

**Good (hair-only model):** afro and coily (very good: full volume, fine edges), braids and cornrows, high buns/updos, curls, long hair with glasses, long hair on a beach, a man with long wavy hair far from the camera.

**Problems:**
| Case | What happens |
| --- | --- |
| Hats (beret, fedora, brimmed hat) | Beret is painted as hair; fedora leaks pink onto the hat and background. Models have no "hat" concept |
| Bald head, white beard | Scalp skin and beard are painted as hair |
| Tiny or distant subjects (crowd, standing woman in a doorway, mother and child) | Mask lands on the wrong region or is missing |
| Heavy tint photos (sepia/red-toned portrait) | Mask spills onto face and beard |

**Model comparison:** the hair-only model beats the multiclass model on afro hair (multiclass clips the afro) and on dark high-contrast faces. No case where multiclass is clearly better, so keep hair-only as the default.

**Likely mitigations for Phase 1 (untested):** crop around the Face Landmarker box and reject frames without a face (kills the tiny-subject failures); subtract the multiclass face-skin class from the hair mask (helps bald/beard); hats need either a "hat" subtract via multiclass "others" class or a documented limitation.

## Recolor (LAB, luminance-preserving)
- Tints and mid/light shifts (blue, violet, emerald, burgundy, copper): look natural, strand texture survives.
- **Naive HSV hue shift fails on dark hair** (V near 0, so almost no visible change). Confirms the plan's risk.
- **Dark → golden blonde works only with a lightness lift.** Additive detail went flat and "painted"; ratio detail with gain 2.5 and eps 3 turned metallic (noise amplified). Current default (ratio detail, eps 12, gain 1.0, lift 0.8) is a believable blonde with texture.
- **Edge rim: fixed** (see `results/recolor/edge_crops.png`, columns before / guard / guard + contrast). Two causes:
  1. Bright metallic rim: at the boundary L mixes in background while the masked-blur base is hair-only, so the L/base ratio explodes. Fix: clamp the ratio to [0.4, 1.8] and fade it to 1.0 within ~5 px of the edge (`edge_guard`).
  2. Golden halo outside the hair: the guided-filter mask has a wide soft skirt. Fix: smoothstep(0.25, 0.75) on the mask weight (`mask_contrast`).
- Trade-off: edges are now crisper and can look slightly cut out on very wispy hair. The Phase 1 shader can expose the smoothstep range as a tunable.
- Remaining look issue: the blonde target (214,170,90) reads olive/flat gold at full intensity. A paler, less saturated target plus the intensity slider is probably the better UX; not tuned yet.

## On-device FPS (Samsung Galaxy Tab A8, SM-X200)
Device: Unisoc T618 (2×A75 + 6×A55, Mali-G52 MP2), Android 14, front camera, 4:3 analysis frames, landscape. Budget tablet, so a conservative baseline. Tablet battery read 0% while powered over USB; may have affected clocks.
Tool: MediaPipe image_segmentation sample (LIVE_STREAM, tasks-vision 1.0.0), patched for front camera and `VTO_FPS` logging (results/sec and capture→result latency, 2 s windows). Raw logs in `results/fps/`.

| Model | Delegate | Median FPS | Range | Latency | Result |
| --- | --- | --- | --- | --- | --- |
| DeepLabV3 (sample default, not ours) | CPU | ~4.3 | 3.5–4.7 | ~265 ms | runs |
| **hair_segmenter** | CPU | **6.9** | 5.4–7.9 | ~174 ms | runs, mask correct |
| hair_segmenter | GPU | — | — | — | **crashes** (SIGABRT, `image_frame.cc:298 Invalid format: UNKNOWN`) |
| selfie_multiclass | CPU | 1.2 | 1.1–1.7 | ~840 ms | runs (far too slow) |
| selfie_multiclass | GPU | — | — | — | init fails, sample falls back to CPU |

### Galaxy S20 5G (SM-G981N, Snapdragon 865 / Adreno 650, Android 13)
Same patched sample, tasks-vision 1.1.0, USB-charging at 90%.

| Model | Delegate | Median FPS | Range | Latency | Result |
| --- | --- | --- | --- | --- | --- |
| DeepLabV3 (sample default) | CPU | n/a | n/a | ~120 ms (on-screen) | runs |
| **hair_segmenter** | CPU | **15.1** | 7.4–16.4 | ~97 ms | runs |
| hair_segmenter | GPU | — | — | — | **same crash**: `image_frame.cc:298 Invalid format: UNKNOWN` |

S20 CPU is ~2.2× the tablet (15.1 vs 6.9 FPS). Still short of 25 FPS, and the GPU crash reproduces on a different SoC and GPU vendor, so it is not a Mali problem.

GPU delegate experiments (hair_segmenter, GPU), on the Tab A8:
| Variant | Result |
| --- | --- |
| tasks-vision 1.0.0, confidence masks | SIGABRT `image_frame.cc:298 Invalid format: UNKNOWN` |
| tasks-vision 1.1.0 (latest), confidence masks | same abort |
| tasks-vision 1.1.0, category-mask output only | SIGABRT `image_frame.cc:109 Check failed: ImageFormat::UNKNOWN != format_` |

So the failure is in MediaPipe's GPU output-image to ImageFrame conversion, independent of version, mask type, and (per the S20 run) device and GPU vendor. It points at the Tasks wrapper or how this sample uses it. It does not tell us whether the Mali-G52 can run the model itself. Next test is LiteRT's GPU delegate directly on the .tflite, bypassing the MediaPipe Tasks wrapper.

Caveats:
- The sample copies and rotates a full RGBA bitmap per frame and returns masks to Java, so these numbers include app overhead, not just model time. A real pipeline keeps frames on the GPU.
- A 25 FPS gate cannot be met by running the model every frame on this tablet's CPU, and the stock GPU delegate path crashes.

## Direct LiteRT benchmark (bypasses the MediaPipe Tasks wrapper)
App: `android/litert-bench` (LiteRT 2.3.0 `CompiledModel` API, Kotlin 2.4). Driver: `scripts/run_litert_bench.ps1`. Each model/config runs in its own process launch. Input: one fixed 512×512 photo (hair model: 512×512×4 float32, 4th channel = 0; multiclass: 256×256×3). Raw logs: `results/fps/<device>_litert_<model>_<config>.log`.

"e2e" = write input + run + read output back to CPU, median of 100 runs after 10 warm-up. (`run` alone is an async dispatch on GPU and understates the cost.) "Sustained" = back-to-back e2e iterations for 20 s, FPS per 5 s window.

**Galaxy S20 5G (Snapdragon 865 / Adreno 650)**
| Model | Config | e2e median | Sustained FPS (per 5 s) | Init + first run | vs CPU output |
| --- | --- | --- | --- | --- | --- |
| hair | CPU | 22.5 ms | 31 / 28 / 28 / 28 | 63 ms | — |
| **hair** | **GPU FP16** | **6.9 ms** | **127 / 120 / 115** | 1.8 s | class agree 100%, hair IoU 0.998 |
| hair | GPU FP32 | 9.9 ms | 85 / 83 / 78 | 1.3 s | identical (IoU 1.000) |
| multiclass | CPU | 360 ms | 2.8 | 0.3 s | — |
| multiclass | GPU FP16 | 19.0 ms | 41 / 41 / 40 | 1.5 s | class agree 99.97%, hair IoU 0.997 |
| multiclass | GPU FP32 | 51.5 ms | 19 | 1.1 s | identical |

Takeaways so far:
- **The GPU crash was the MediaPipe wrapper, not the hardware.** LiteRT's own GPU path runs the hair model in about 7 ms with FP16, with no accuracy loss on the mask.
- Even on CPU the hair model sustains about 28 FPS on the S20, so the MediaPipe sample's 15 FPS was mostly bitmap/wrapper overhead.
- Multiclass is usable on GPU (41 FPS) but not on CPU (2.8 FPS), so any multiclass use needs the GPU.
- No visible thermal drop in 20 s (hair FP16 127 → 115 FPS). A longer soak test is still worth doing.
- Caveat: this is segmentation only. The camera → GPU texture copy, guided filter and recolor shader are still to be added.

## Gate status
| Gate | Status |
| --- | --- |
| Masks acceptable on most photos | **Pass.** Kaggle selfies: all 4–5 (author's aggregate judgment). Hard set: good on curly/coily/braids/glasses; fails on hats, bald/white-beard and tiny subjects (see above). Front-camera use case is the Kaggle-like one |
| ≥ 25 FPS on device | **Met on Galaxy S20** with direct LiteRT (hair model: GPU FP16 ~120 FPS sustained, CPU ~28 FPS). The stock MediaPipe sample is not usable on GPU (crashes on Tab A8 and S20; 7 and 15 FPS on CPU). Galaxy Tab A8 (budget SoC) still to be run with the direct LiteRT app |
| Dark→blonde convincing | Mostly: rim and halo fixed; the blonde shade itself still needs tuning |

## Phase 0 verdict (2026-10-08)
**Go for Phase 1** on the hair-color path, with these decisions and known gaps:
- Segmentation: MediaPipe `hair_segmenter.tflite`, run through **LiteRT `CompiledModel` on GPU (FP16)**, not the MediaPipe Tasks wrapper (it crashes with the GPU delegate on both test devices).
- Recolor: LAB lift + ratio detail + edge guard + mask contrast (`src/vto_spike/recolor.py`) is the reference to port to a GLSL shader. Blonde shade still to tune.
- Deliberately skipped for now: hats, bald/white-beard mask failures; Galaxy Tab A8 run of the direct LiteRT app (it was 7 FPS through the MediaPipe sample, so expect the weakest result there); long thermal soak test.
- Not yet measured: full pipeline cost (camera texture, guided filter, recolor shader) on device.

## Reproduce
```
conda run -n vto python scripts/download_models.py
conda run -n vto python scripts/prepare_kaggle.py
conda run -n vto python scripts/run_masks.py      # results/masks/sheet_*.png, mask_scores.csv
conda run -n vto python scripts/run_recolor.py    # results/recolor/*.png
conda run -n vto python -m pytest
```
