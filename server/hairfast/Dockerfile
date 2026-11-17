# HairFastGAN (MIT code; StyleGAN2-FFHQ based, see WEIGHTS.md for the weight licences). Needs its old torch 1.13 / python 3.10 stack,
# hence its own image. dlib is built from source, so the -devel base (compilers). Weights live under /work (see fetch_weights.sh),
# not in the image. Build context is the code/ folder (hs build), so common/ can be shared with the other GAN models.
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/AIRI-Institute/HairFastGAN.git /opt/HairFastGAN \
    && git -C /opt/HairFastGAN rev-parse HEAD > /opt/HairFastGAN.commit
# dlib is only used for CPU face detection + landmarks; its default build finds CUDA here and fails (no cusolverDn.h), so build it
# without. Then the repo's own pins (torch 1.13.1 is already in the base image).
RUN pip install --no-cache-dir dlib==19.24.1 --global-option=--no --global-option=DLIB_USE_CUDA
RUN pip install --no-cache-dir -r /opt/HairFastGAN/requirements.txt huggingface-hub==0.25.2

# StyleGAN2's fused_act / upfirdn2d CUDA ops are JIT-compiled on first import: needs ninja, and the cache goes under /work
RUN pip install --no-cache-dir ninja
# CUDA_HOME: torch otherwise finds conda's partial nvcc (/opt/conda), which lacks cusolverDn.h
ENV CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 TORCH_EXTENSIONS_DIR=/work/torch-ext

WORKDIR /code
COPY common /code/common
COPY hairfast/run.py hairfast/pipeline.py hairfast/fetch_weights.sh /code/
RUN sed -i 's/\r$//' /code/fetch_weights.sh   # scp from Windows may leave CRLF endings
