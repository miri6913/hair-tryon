# Qwen-Image-Edit-2511 (Apache-2.0) runner. bf16 weights are ~40 GB+, so run.py loads the transformer in 4-bit (NF4) on one 24 GB card.
# Built and run on the server; weights download into /work/hf-cache (HF_HOME) on first use, never into the image.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    "git+https://github.com/huggingface/diffusers" \
    "transformers>=4.51" accelerate bitsandbytes sentencepiece protobuf pillow numpy
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /code
COPY qwen/run.py /code/run.py
