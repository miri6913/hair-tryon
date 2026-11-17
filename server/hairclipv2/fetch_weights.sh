#!/bin/sh
# Download the HairCLIPv2 weights (Google Drive ids from the repo README) into /work/models/hairclipv2/pretrained_models. Idempotent.
# Google Drive can refuse with a quota error; then download the five files by hand from the README links into that folder.
set -e
D=/work/models/hairclipv2/pretrained_models
mkdir -p $D
cd $D
get() {  # id, expected file name (the README does not name them; the repo's options.py does)
    if [ ! -f "$2" ]; then gdown "https://drive.google.com/uc?id=$1" -O "$2"; fi
}
get 1g8S81ZybmrF86OjvjLYJzx-wx83ZOiIw ffhq.pt                 # FFHQ StyleGAN 1024
get 1OG6t7q4PpHOoYNdP-ipoxuqYbfMSgPta seg.pth                 # face parsing (from Barbershop)
get 1c-SgUUQj0X1mIl-W-_2sMboI2QS7GzfK shape_predictor_68_face_landmarks.dat
get 1sa732uBfX1739MFsvtRCKWCN54zYyltC bald_proxy.pt           # from HairMapper
get 1qk0ZIfA1VmrFUzDJ0g8mK8nx0WtF-5sY sketch_proxy.pt
ls -la $D
# options.py also wants ffhq_PCA.npz; HairFastGAN ships the same FFHQ PCA, reuse it when present
test -f ffhq_PCA.npz || cp /work/models/hairfast/pretrained_models/StyleGAN/ffhq_PCA.npz . 2>/dev/null || echo "WARNING: ffhq_PCA.npz missing"
