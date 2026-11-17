# Stable-Hair (Apache-2.0 code; SD 1.5 weights are CreativeML OpenRAIL-M). Needs the older torch/diffusers stack, hence
# its own image. Weights (Google Drive) and SD 1.5 live under /work (see fetch_weights.sh), not in the image.
# Versions below follow the repo's requirements.txt for the pieces it needs; the rest of that file (gradio, keras, ...) is skipped.
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    diffusers==0.31.0 transformers==4.45.2 accelerate==1.0.1 huggingface-hub==0.25.2 \
    omegaconf==2.3.0 einops==0.4.1 numpy==1.26.4 pillow==10.3.0 opencv-python-headless==4.9.0.80 \
    safetensors scipy scikit-image kornia==0.7.3 peft==0.11.1 gdown==5.2.0 tqdm pyyaml

RUN git clone https://github.com/Xiaojiu-z/Stable-Hair.git /opt/Stable-Hair \
    && git -C /opt/Stable-Hair rev-parse HEAD > /opt/Stable-Hair.commit

WORKDIR /code
COPY stable_hair/run.py stable_hair/fetch_weights.sh /code/
RUN sed -i 's/\r$//' /code/fetch_weights.sh   # scp from Windows may leave CRLF endings
