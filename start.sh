#!/bin/bash

# Optimal Charge Security Camera - Start Script
# Usage: ./start.sh [mode] [model]
# Modes:
#   --webcam    Run with webcam
#   --image     Run with image (creates sample image if needed)
#   (no args)   Default to image mode with yolov8n
# Models: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

set -e

# Default mode and model
MODE="image"
MODEL="yolov8n"

# Parse arguments
if [ "$1" = "--webcam" ]; then
    MODE="webcam"
    MODEL="${2:-yolov8n}"
elif [ "$1" = "--image" ]; then
    MODE="image"
    MODEL="${2:-yolov8n}"
elif [ -z "$1" ]; then
    # Default
    :
else
    echo "Usage: $0 [--webcam|--image] [model]"
    echo "  --webcam  Run with webcam"
    echo "  --image   Run with image (default)"
    echo "  model     YOLOv8 model: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x (default: yolov8n)"
    exit 1
fi

echo "Starting Optimal Charge Security Camera in $MODE mode with model $MODEL..."

# Setup for image mode
if [ "$MODE" = "image" ]; then
    # Create sample image if it doesn't exist
    if [ ! -f "images/sample.jpg" ]; then
        echo "Creating sample image..."
        uv run python - <<'PY'
from PIL import Image
import numpy as np
img = (np.random.rand(480,640,3)*255).astype("uint8")
Image.fromarray(img).save("images/sample.jpg")
print("Created images/sample.jpg")
PY
    fi

    # Run with image
    OCS_SOURCE=image OCS_INPUT=images/sample.jpg OCS_INTERVAL_SEC=2 OCS_MODEL=$MODEL uv run ocs-run
else
    # Run with webcam
    OCS_SOURCE=webcam OCS_INTERVAL_SEC=1 OCS_MODEL=$MODEL uv run ocs-run
fi