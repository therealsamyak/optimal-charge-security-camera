"""Image processing with YOLO models for human detection."""

from pathlib import Path
from typing import Tuple
from loguru import logger

# Cache for loaded models to avoid reloading
_model_cache: dict = {}


def load_yolo_model(model_name: str):
    """Load YOLO model, using cache if available."""
    global _model_cache

    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        from ultralytics import YOLO

        # Map model names to ultralytics model paths
        model_map = {
            "YOLOv10-N": "yolov10n.pt",
            "YOLOv10-S": "yolov10s.pt",
            "YOLOv10-M": "yolov10m.pt",
            "YOLOv10-B": "yolov10b.pt",
            "YOLOv10-L": "yolov10l.pt",
            "YOLOv10-X": "yolov10x.pt",
        }

        model_file = model_map.get(model_name)
        if not model_file:
            raise ValueError(f"Unknown model name: {model_name}")

        logger.debug(f"Loading YOLO model: {model_name} ({model_file})")
        model = YOLO(model_file)
        _model_cache[model_name] = model
        return model
    except ImportError:
        logger.error("ultralytics not installed. Install with: uv add ultralytics")
        raise
    except Exception as e:
        logger.error(f"Failed to load YOLO model {model_name}: {e}")
        raise


def process_image_with_model(
    image_path: str, model_name: str, expected_humans: int = 1
) -> Tuple[bool, int, float]:
    """
    Process image with YOLO model and check if human count matches expected.

    Args:
        image_path: Path to image file
        model_name: YOLO model name (e.g., "YOLOv10-N")
        expected_humans: Expected number of humans (default: 1)

    Returns:
        Tuple of (is_correct, detected_count, confidence)
        - is_correct: True if detected count matches expected
        - detected_count: Number of humans detected
        - confidence: Average confidence of detections (0.0 if none)
    """
    import time

    if not Path(image_path).exists():
        logger.warning(f"Image not found: {image_path}, assuming incorrect")
        return False, 0, 0.0

    try:
        model = load_yolo_model(model_name)

        # Run inference
        start_time = time.time()
        results = model(image_path, verbose=False)
        inference_time = time.time() - start_time

        # Count humans (class 0 in COCO dataset is "person")
        detected_count = 0
        confidences = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Class 0 is "person" in COCO dataset
                    if int(box.cls) == 0:
                        detected_count += 1
                        confidences.append(float(box.conf))

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        is_correct = detected_count == expected_humans

        logger.debug(
            f"Model {model_name} on {Path(image_path).name}: "
            f"detected {detected_count} humans (expected {expected_humans}), "
            f"correct={is_correct}, confidence={avg_confidence:.3f}, "
            f"time={inference_time * 1000:.2f}ms"
        )

        return is_correct, detected_count, avg_confidence

    except Exception as e:
        logger.error(f"Error processing image {image_path} with {model_name}: {e}")
        return False, 0, 0.0


def get_image_path(image_quality: str) -> str:
    """Get image path based on quality setting."""
    if image_quality == "good":
        return "image1.png"
    elif image_quality == "bad":
        return "image2.jpeg"
    else:
        raise ValueError(f"Unknown image quality: {image_quality}")
