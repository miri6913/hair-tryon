# HairCLIPv2 (ICCV 2023, github.com/wtybest/HairCLIPv2). Research comparison only: the repo has NO licence file (all rights reserved).
# Its README pins python 3.8 / torch 1.9.0+cu111 / dlib 19.22.1, hence its own image. Weights are fetched into /work (fetch_weights.sh).
# Build context is code/ (hs build), so common/ is shared with the other GAN models.
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

# the CUDA 11.1 base still lists NVIDIA apt repos whose signing key has expired, which fails apt-get update; drop them (not needed)
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update && apt-get install -y --no-install-recommends git cmake build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/wtybest/HairCLIPv2.git /opt/HairCLIPv2 \
    && git -C /opt/HairCLIPv2 rev-parse HEAD > /opt/HairCLIPv2.commit
# same build fixes as the hairfast image: torch finds conda's partial nvcc unless CUDA_HOME is set; dlib built without CUDA
ENV CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 TORCH_EXTENSIONS_DIR=/work/torch-ext-hcv2
RUN pip install --no-cache-dir dlib==19.22.1 --global-option=--no --global-option=DLIB_USE_CUDA
RUN pip install --no-cache-dir ftfy regex tqdm matplotlib opencv-python-headless scikit-image kornia==0.6.7 \
    face-alignment==1.3.5 ninja gdown git+https://github.com/openai/CLIP.git

WORKDIR /code
COPY common /code/common
COPY hairclipv2/run.py hairclipv2/fetch_weights.sh /code/
RUN sed -i 's/\r$//' /code/fetch_weights.sh
