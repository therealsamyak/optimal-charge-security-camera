#!/bin/bash

# Optimal Charge Security Camera - Start Script
# Usage: ./start.sh [option]
# Options:
#   -w    Run with webcam
#   -i    Run with image (creates sample image if needed)
#   (no args) Default to image mode

set -e

# Default mode
MODE="image"

# Parse arguments
if [ "$1" = "-w" ]; then
    MODE="webcam"
elif [ "$1" = "-i" ]; then
    MODE="image"
elif [ -n "$1" ]; then
    echo "Usage: $0 [-w|-i]"
    echo "  -w  Run with webcam"
    echo "  -i  Run with image (default)"
    exit 1
fi

echo "Starting Optimal Charge Security Camera in $MODE mode..."

# Setup for image mode
if [ "$MODE" = "image" ]; then
    # Create sample image if it doesn't exist
    if [ ! -f "sample.jpg" ]; then
        echo "Creating sample image..."
        uv run python - <<'PY'
from PIL import Image
import numpy as np
img = (np.random.rand(480,640,3)*255).astype("uint8")
Image.fromarray(img).save("sample.jpg")
print("Created sample.jpg")
PY
    fi
    
    # Run with image
    OCS_SOURCE=image OCS_INPUT=sample.jpg OCS_INTERVAL_SEC=2 uv run ocs-run
else
    # Run with webcam
    OCS_SOURCE=webcam OCS_INTERVAL_SEC=1 uv run ocs-run
fi