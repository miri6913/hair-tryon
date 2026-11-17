# FLUX.2 [klein] runner (4B Apache-2.0; 9B FLUX Non-Commercial, gated on Hugging Face: needs HF_TOKEN).
# Built and run on the server; weights download into /work/hf-cache (HF_HOME) on first use, never into the image.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    "git+https://github.com/huggingface/diffusers" \
    "transformers>=4.51" accelerate sentencepiece protobuf pillow numpy
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /code
COPY flux2/run.py flux2/gen_refs.py /code/
