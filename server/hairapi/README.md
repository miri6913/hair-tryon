# Hairstyle API

Image in, image out: `POST /v1/hairstyle` with a photo and a `style_id` returns the edited photo with the new hairstyle.

**How it works (default, `REF_MODE=text`, since 2026-11-02):** FLUX.2 [klein] 4B edits the photo from one fixed text prompt per style. No reference photo.
The person's own hair colour is measured on the photo (MediaPipe hair mask + `recolor2.measure_shade`), turned into a colour word (black, dark brown,
brown, light brown, blonde) and named in the prompt, so the new style keeps their colour. The model's output is returned as is (about 768×1024): the whole
image is a re-draw, so the face and background can drift slightly and skin gets smoothed (see "Known issues"). Live colour changing is a separate,
on-device feature and is not part of this API.

Prompt: `Change only the hairstyle of the person in the photo to {hair}, in {colour} hair colour. Keep the face, skin texture, expression, clothes, pose,
lighting and background exactly the same. Photorealistic.` (`{hair}` = `hair_prompt` of the style in `catalog.json`; one embedding per style × colour is
computed at startup, so the text encoder is unloaded.) Two things that did NOT work, kept here so nobody retries them: "keep the current hair colour" let
black hair come out with blond highlights; "no highlights, no dye" made it worse (negations prime the model). Naming the measured colour fixed both.

Other modes, kept for comparison: `REF_MODE=ref` (a reference photo of the style as second image + prompt; the hair colour then comes from the reference, and the
reference's framing, clothes and even face leak into the result unless it is a tight head crop, see `data/hairapi_refs/REFERENCE_GUIDE.md` and
`scripts/crop_reference.py`) and `COMPOSITE=1` (own colour + paste-back onto the original, `vto_spike/composite.py`: removes face drift but left pale
forehead patches and brow remnants; judged worse by the user).

| Method | Path | Notes |
|---|---|---|
| GET | `/v1/catalog?gender=female\|male` | styles grouped by length (`short`/`medium`/`long`): `id`, `name`, `thumbnail_url` (**null** in text mode: the app brings its own style icons) |
| GET | `/v1/catalog/thumbnails/{id}.jpg` | only in `REF_MODE=ref` (the reference photo, 320 px); 404 otherwise |
| POST | `/v1/hairstyle` | multipart `photo` (any image PIL reads, ≤ 12 MB) + `style_id` -> `200 image/jpeg` at the model's size (long side 1024 px, e.g. 768×1024). Headers: `X-Timings` (`seg_old;generate;post;total`, seconds), `X-Warnings` (`head_cut_off`) |
| GET | `/healthz` | no key; 200 once the model is loaded, 503 before (the Docker healthcheck uses it) |

All `/v1` routes need the `X-API-Key` header. Errors (body `{"detail": {"code", "message"}}` for photo problems):

| Status | When |
|---|---|
| 400 | `bad_image` (not an image), unknown `style_id`, unknown gender |
| 401 | bad or missing key |
| 413 | photo larger than 12 MB |
| 422 | `no_face`, `multiple_faces` (MediaPipe selfie-multiclass face-skin blobs: a heuristic, not a face detector) |
| 429 | more than `MAX_QUEUE` (12) requests in flight; `Retry-After: 10` |
| 503 | model still loading |

`head_cut_off` (hair touches the top edge) is only a warning: the model handled such photos in testing. The app's capture guide should still ask for headroom.

## Styles (16; `data/hairapi_refs/styles.json`, copied into `server/hairapi/catalog.json`)
Female: short pixie / short bob / Italian bob; medium layered / bob with bangs; long straight / low side ponytail / butterfly cut / long volume. Male: short buzz / textured crop / comma hair; medium quiff /
side part; long shoulder-length / man bun. The `reference` field of `catalog.json` is only used in `REF_MODE=ref`. To add a style: add it to `styles.json`
(id, gender, length, name, `hair_prompt`), regenerate `catalog.json` entries, rebuild.

## Time and resources (measured, one RTX 3090, 2026-11-02)
- **~4.3 s per image** end to end (hair measurement 0.1 s, generate ~4.2 s, no post step). Input photo fit to 1024 px long side, 4 steps. (`REF_MODE=ref`: ~6.5 s; `COMPOSITE=1`: ~7.6 s.)
- **~8.4 GB VRAM** for the live container (the transformer, VAE and 60 cached prompt embeddings; the text encoder sits in CPU memory, so budget ~16 GB host RAM).
- **One request at a time per GPU worker**, FIFO: ~800 images/hour per GPU. `MAX_QUEUE=12` (set by `hs api-start`): 12 × ~4.5 s ≈ 54 s worst-case wait, under Cloudflare's ~100 s request limit (error 524). The app's read timeout must cover its queue wait; add GPUs before raising `MAX_QUEUE`. (The 25-concurrent-request load test, 20 served / 5 rejected, was run in the 7 s reference mode.)
- Startup ~1.5 minutes (model load + 60 prompt embeddings).
- Privacy: photos and results exist in memory for the request only; nothing is written to disk (`docker diff` clean after load); logs hold the style id, timings and the colour word (a category).

## Run on <host> (Docker)
```
scripts\hs push                # server/, src/vto_spike, models/ -> code/ ; data/hairapi_refs -> refs/ ; data/hairstyle_set -> inputs/
scripts\hs api-build
scripts\hs api-start -g 0      # refuses a busy GPU; --restart unless-stopped; 127.0.0.1:8090 on the server only
scripts\hs api-logs            # "model loaded, 16 styles" + (healthy)
scripts\hs api-stop
```
- The FLUX weights are not in the image: `hf-cache` is mounted read-only with `HF_HUB_OFFLINE=1`. `refs/` is mounted read-only at `/inputs` (needed only in `REF_MODE=ref`; the personal test photos are not visible to the container).
- A second API for experiments (port 8091, no restart policy, holds a GPU while it runs): `hs api-test-start -g 1 --refs-dir refs [--test-env REF_MODE=ref] [--test-env COMPOSITE=1]`, `hs api-test-stop`. `scripts/ref_compare.py` compares its outputs with the stored baseline and prints a face-drift score.
- The API key is created once in `<server root>/hairapi.env` (mode 600) and never printed by `hs`; read it with `ssh <host> cat <server root>/hairapi.env`.
- Test from this PC through a tunnel: `ssh -N -L 8090:127.0.0.1:8090 <host>`, then `set HAIRAPI_KEY=...` and `python scripts/hairapi_client.py edit --photo <jpg> --style f_long_straight`.
- The container holds a GPU for as long as it runs: the server is shared, so stop it when not testing.

## Known issues
1. **Face re-drawn (identity drift).** Measured as the mean abs RGB difference over the original's face-skin region (`scripts/ref_compare.py`): prompt-only 21.2 on average over the 12 styles × 4 test photos vs 25.4 for the synthetic-reference route (lower = closer to the photo). Skin is smoothed (the tester's acne mostly disappears), some photos change slightly (eyes, cheeks); long hairstyles and the quiff/side part on one tester are the worst (25–29). Ideas if it matters: the 9B model (non-commercial licence), a paste-back limited to the face, fewer/more steps.
2. **Colour is a category, not an exact match.** Five words only; the second woman's dark brown hair (measured very dark, L 7.5) comes out near black; the first woman's light brown comes out brown. Add more words (grey/white, red) or a finer scale if needed.
3. **Style thumbnails:** none from the API in text mode; the app needs its own icons (or illustrations we make).
4. Not used by default but in the code (`COMPOSITE=1`): the paste-back defects (pale forehead patches on bright/shadowed foreheads, white streak on blonde→black recolour).

## Still to do before this is production
1. **Public HTTPS: done (2026-11-05) with Tailscale Funnel**, a free, permanent `https://<node>.<tailnet>.ts.net`. The API still listens only on the server's localhost; `hs funnel-start` runs the `tailscale/tailscale` container in userspace mode (it dials out: no open port, no sudo, no domain). Tailscale side, once: free account, DNS -> MagicDNS + Enable HTTPS, ACL `nodeAttrs` grants `funnel` to `autogroup:member`, a single-use auth key put on the server by hand (not through chat or scripts): `ssh <host> "umask 077; echo TS_AUTHKEY=<key> > <server root>/tailscale.env"`. Then `scripts\hs funnel-start` (prints the URL), `funnel-logs`, `funnel-stop`. The node identity lives in `tailscale/state/` on the server, so restarts keep the URL and the key is only used once. To re-register (new key): `funnel-stop`, delete `tailscale/state/`, write a new `tailscale.env`, `funnel-start`. The Android app reads the URL from `hairapi.url` in the git-ignored `local.properties`.
   - Caveats: Funnel is relayed and throttled by Tailscale (not published; fine for a few concurrent users) and Tailscale still labels it beta. Measured: one 490 KB photo -> 155 KB result took 11.0 s wall for 4.3 s of server time. Free, but with no uptime promise; if it becomes a bottleneck, the Cloudflare Tunnel route still works (`hs tunnel-start`; needs a domain on Cloudflare DNS, a tunnel token in `cloudflared.env`, and a Public Hostname -> `http://localhost:8090`; add a WAF rate-limit rule on `/v1/hairstyle`). Rotate the API key by deleting `hairapi.env` on the server and running `api-start`.
2. **Consent and privacy**: users upload face photos to our server; consent text, retention policy and local biometric/privacy rules need review.
3. **Text encoder licence**: the FLUX.2 klein 4B text encoder is Qwen3 (`Qwen3ForCausalLM`); BFL's repo does not state its licence. Check it in the repo's `text_encoder` folder. (Nothing else needs a licence check any more: there are no reference photos in the default mode.)
4. Throughput under real traffic (the default mode has not had the 25-request load test), and an async-job variant if the sync wait gets too long.
5. ~~The Android client~~ done: see `android/hairtryon/` (style picker, upload, result).
