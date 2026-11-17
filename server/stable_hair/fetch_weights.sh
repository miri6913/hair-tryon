#!/bin/sh
# Download the Stable-Hair checkpoints (Google Drive) and SD 1.5 (Hugging Face) into /work/models. Idempotent.
# Google Drive can refuse with a quota error; then download the folder by hand from the repo README link into
# /work/models/stable_hair/ so it contains stage1/ and stage2/.
set -e
mkdir -p /work/models/stable_hair
if [ ! -f /work/models/stable_hair/stage2/pytorch_model.bin ]; then
    gdown --folder "https://drive.google.com/drive/folders/1E-8Udfw8S8IorCWhBgS4FajIbqlrWRbQ" -O /work/models/stable_hair
fi
python - <<'EOF'
from huggingface_hub import snapshot_download
# runwayml/stable-diffusion-v1-5 (named in the repo config) was taken down; this is the community mirror of it.
snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5", local_dir="/work/models/sd15",
                  allow_patterns=["model_index.json", "*/config.json", "*/*.json", "*/*.txt", "*/diffusion_pytorch_model.safetensors",
                                  "*/model.safetensors"])
EOF
find /work/models/stable_hair -maxdepth 2 | head -20
