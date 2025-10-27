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

