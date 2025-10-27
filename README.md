# Optimal Charge Security Camera

A security camera system with optimal charge management.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/therealsamyak/optimal-charge-security-camera
cd optimal-charge-security-camera
uv sync

# Run with webcam
./start.sh -w

# Or run with sample image
./start.sh -i

# View metrics
tail -f runs/logs.jsonl
```

## Usage

### Environment Variables
- `OCS_SOURCE`: Input source (`webcam` or `image`)
- `OCS_INPUT`: Image file path (when using image source)
- `OCS_INTERVAL_SEC`: Processing interval in seconds

### Commands
- `uv run ocs-run`: Run the main camera pipeline
- `uv run python -m ocs_cam.main`: Alternative explicit invocation

### Dependencies
- Python 3.12+ (managed via uv)
- OpenCV (for webcam support): `sudo apt install python3-opencv libopencv-dev`

