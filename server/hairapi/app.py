"""Hairstyle API: image in, image out. See README.md.

  GET  /v1/catalog?gender=female|male      styles grouped by length
  GET  /v1/catalog/thumbnails/{id}.jpg     style thumbnail
  POST /v1/hairstyle                       multipart photo + style_id -> image/jpeg (own hair colour)
  GET  /healthz                            200 once the model is loaded (no key needed; for the Docker healthcheck)

Every /v1 route needs the X-API-Key header. GPU work runs on ONE worker thread, so requests are served one at a time in
arrival order; more than MAX_QUEUE in flight gets 429. Photos and results exist only in memory for the request.
"""
import asyncio
import hmac
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile

from pipeline import HairPipeline, PhotoError

API_KEY = os.environ.get("API_KEY", "")
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "20"))
MAX_UPLOAD = 12 * 1024 * 1024
CATALOG = Path(os.environ.get("CATALOG", Path(__file__).with_name("catalog.json")))
INPUTS = Path(os.environ.get("INPUTS_DIR", "/inputs"))
LENGTHS = ["short", "medium", "long"]

log = logging.getLogger("hairapi")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
worker = ThreadPoolExecutor(max_workers=1)   # the pipeline (and its MediaPipe/CUDA state) lives on this one thread
state: dict = {"pipe": None, "inflight": 0}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not API_KEY:
        raise RuntimeError("API_KEY is not set")
    loop = asyncio.get_running_loop()
    state["pipe"] = await loop.run_in_executor(worker, HairPipeline, CATALOG, INPUTS)
    log.info("model loaded, %d styles", len(state["pipe"].styles))
    yield


app = FastAPI(title="Hairstyle API", version="1", lifespan=lifespan)


def auth(x_api_key: str = Header(default="")) -> None:
    if not hmac.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(401, "bad or missing X-API-Key")


def pipe() -> HairPipeline:
    if state["pipe"] is None:
        raise HTTPException(503, "model is still loading")
    return state["pipe"]


@app.get("/healthz")
def healthz(response: Response) -> dict:
    ready = state["pipe"] is not None
    if not ready:
        response.status_code = 503
    return {"ready": ready, "inflight": state["inflight"], "max_queue": MAX_QUEUE}


@app.get("/v1/catalog", dependencies=[Depends(auth)])
def catalog(gender: str | None = None) -> dict:
    if gender not in (None, "male", "female"):
        raise HTTPException(400, "gender must be male or female")
    out = {}
    for g in ("female", "male"):
        if gender in (None, g):
            out[g] = [
                {"length": ln, "styles": [
                    # null in the default prompt-only mode: there is no reference photo to show (the app brings its own icons)
                    {"id": sid, "name": s["name"], "thumbnail_url": f"/v1/catalog/thumbnails/{sid}.jpg" if sid in pipe().thumbs else None}
                    for sid, s in pipe().styles.items() if s["gender"] == g and s["length"] == ln]}
                for ln in LENGTHS]
    return {"version": pipe().version, "catalog": out}


@app.get("/v1/catalog/thumbnails/{style_id}.jpg", dependencies=[Depends(auth)])
def thumbnail(style_id: str) -> Response:
    thumb = pipe().thumbs.get(style_id)
    if thumb is None:
        raise HTTPException(404, "unknown style")
    return Response(thumb, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.post("/v1/hairstyle", dependencies=[Depends(auth)])
async def hairstyle(photo: UploadFile = File(...), style_id: str = Form(...)) -> Response:
    p = pipe()
    if style_id not in p.styles:
        raise HTTPException(400, f"unknown style_id {style_id!r}")
    data = await photo.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"photo larger than {MAX_UPLOAD // 2**20} MB")
    if state["inflight"] >= MAX_QUEUE:
        raise HTTPException(429, "server is busy, try again shortly", headers={"Retry-After": "10"})

    state["inflight"] += 1
    try:
        loop = asyncio.get_running_loop()

        def job():
            rgb = p.decode(data)
            warnings = p.check(rgb)
            jpeg, t = p.run(rgb, style_id)
            return jpeg, t, warnings

        jpeg, t, warnings = await loop.run_in_executor(worker, job)
    except PhotoError as e:
        raise HTTPException(422 if e.code != "bad_image" else 400, {"code": e.code, "message": e.message}) from e
    finally:
        state["inflight"] -= 1

    t["total"] = sum(t.values())
    log.info("style=%s %s", style_id, " ".join(f"{k}={v:.2f}s" for k, v in t.items()))
    headers = {"X-Timings": ";".join(f"{k}={v:.2f}" for k, v in t.items())}
    if warnings:
        headers["X-Warnings"] = ",".join(warnings)
    return Response(jpeg, media_type="image/jpeg", headers=headers)
