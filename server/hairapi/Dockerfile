# Hairstyle API (FLUX.2 [klein] 4B, Apache-2.0). Build context is code/ on the server (scripts\hs push copies server/,
# src/vto_spike and models/ there). The FLUX weights are NOT in the image: mount hf-cache read-only (HF_HUB_OFFLINE=1).
# The style reference photos are not in the image either: mount the server's inputs/ folder at /inputs.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libegl1 libgles2 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    "git+https://github.com/huggingface/diffusers" \
    "transformers>=4.51" accelerate sentencepiece protobuf pillow numpy \
    "mediapipe==1.1.0" opencv-python-headless \
    fastapi "uvicorn[standard]" python-multipart
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/hf HF_HUB_OFFLINE=1 \
    PYTHONPATH=/app/src PYTHONUNBUFFERED=1

# vto_spike/segment.py finds the models at <package>/../../models
WORKDIR /app/server
COPY vto_spike /app/src/vto_spike
COPY models /app/models
COPY hairapi/app.py hairapi/pipeline.py hairapi/catalog.json /app/server/

EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8090/healthz', timeout=4)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
