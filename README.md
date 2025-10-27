# optimal-charge-security-camera

## Quickstart
```bash
# 1) Install Python & create a project env (via uv)
uv python install 3.12
uv venv --python 3.12

# 2) Install project + deps
uv sync

# 3) Run the baseline pipeline
uv run ocs-run
# or (explicit src layout form)
# PYTHONPATH=src uv run python -m ocs_cam.main
# Ensure Python env (uv) if starting fresh
uv python install 3.12
uv venv --python 3.12
uv sync

# (Optional) create a sample image
uv run python - <<'PY'
from PIL import Image, ImageDraw
import numpy as np
img = (np.random.rand(480,640,3)*255).astype("uint8")
Image.fromarray(img).save("sample.jpg")
print("wrote sample.jpg")
PY

# Run baseline (choose one)
# Image:
OCS_SOURCE=image OCS_INPUT=sample.jpg OCS_INTERVAL_SEC=2 uv run ocs-run
# Webcam:
# OCS_SOURCE=webcam OCS_INTERVAL_SEC=1 uv run ocs-run
# 1) Get repo (or pull latest)
cd ~/optimal-charge-security-camera 2>/dev/null || \
git clone https://github.com/therealsamyak/optimal-charge-security-camera && \
cd optimal-charge-security-camera
git pull

# 2) Install uv + Python env
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.profile 2>/dev/null || source ~/.bashrc 2>/dev/null || true
uv python install 3.12 || uv python install 3.10
uv venv --python 3.12 || uv venv --python 3.10
uv sync

# 3) (If using webcam) OpenCV via apt
sudo apt update && sudo apt install -y python3-opencv libopencv-dev

# 4) Run baseline (pick one)
# Webcam:
OCS_SOURCE=webcam OCS_INTERVAL_SEC=1 uv run ocs-run
# Image:
# OCS_SOURCE=image OCS_INPUT=sample.jpg OCS_INTERVAL_SEC=2 uv run ocs-run

# 5) View metrics
tail -f runs/logs.jsonl

