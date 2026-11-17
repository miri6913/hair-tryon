#!/bin/sh
# Download the HairFastGAN weights (Hugging Face AIRI-Institute/HairFastGAN) into /work/models/hairfast. Idempotent.
# The repo expects them in a folder called pretrained_models; run.py links it into the code checkout.
set -e
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("AIRI-Institute/HairFastGAN", local_dir="/work/models/hairfast", allow_patterns=["pretrained_models/*"])
EOF
# big-lama (Apache-2.0), TorchScript build from the simple-lama-inpainting release; used by common/finish.py for the clean plate
mkdir -p /work/models/lama
test -f /work/models/lama/big-lama.pt || python - <<'EOF'
import urllib.request
urllib.request.urlretrieve("https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
                           "/work/models/lama/big-lama.pt")
EOF
ls -la /work/models/lama
find /work/models/hairfast/pretrained_models -maxdepth 2 | sort
du -sh /work/models/hairfast
test -f /work/models/hairfast/pretrained_models/ShapeAdaptor/shape_predictor_68_face_landmarks.dat \
    || echo "WARNING: dlib landmark file not in the HF repo; pipeline.py needs it at pretrained_models/ShapeAdaptor/"
