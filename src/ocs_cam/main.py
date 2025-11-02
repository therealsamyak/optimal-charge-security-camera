from datetime import datetime
from loguru import logger
import json
import os
import time
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _read_frame(source: str | None, mode: str):
    if mode == "image":
        if not source:
            raise ValueError("Set OCS_INPUT=<path-to-image> for image mode")
        if cv2:
            img_bgr = cv2.imread(source)
            if img_bgr is None:
                raise FileNotFoundError(source)
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if Image:
            return np.array(Image.open(source).convert("RGB"))
        raise RuntimeError("Need OpenCV or Pillow to read images")

    if mode in ("webcam", "video"):
        if not cv2:
            raise RuntimeError("OpenCV required for webcam/video")
        cap = cv2.VideoCapture(0 if mode == "webcam" else source)
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Failed to read from camera/video")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # synthetic
    return (np.random.rand(480, 640, 3) * 255).astype("uint8")


def main():
    mode = os.environ.get("OCS_SOURCE", "image")  # image|webcam|video|synthetic
    src = os.environ.get("OCS_INPUT", "data/input/sample.jpg")
    runs = os.environ.get("OCS_RUNS_DIR", "data/output/runs")
    interval = float(os.environ.get("OCS_INTERVAL_SEC", "2.0"))
    model_name = os.environ.get("OCS_MODEL", "yolov10n")

    if not YOLO:
        raise RuntimeError("ultralytics required for YOLO models")

    # Load model (ultralytics will auto-download if not found locally)
    model = YOLO(f"{model_name}.pt")

    os.makedirs(runs, exist_ok=True)
    log_path = os.path.join(runs, "logs.jsonl")

    logger.info(f"Pipeline start @ {datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"source={mode} input={src} interval={interval}s model={model_name}")
    logger.info(f"metrics -> {log_path} (Ctrl+C to stop)")

    try:
        while True:
            rgb = _read_frame(src, mode)

            # YOLO inference
            t0 = time.perf_counter()
            results = model(rgb, verbose=False)
            latency_ms = (time.perf_counter() - t0) * 1000

            # Extract top detection
            if results and len(results) > 0:
                result = results[0]
                if len(result.boxes) > 0:
                    # Get the detection with highest confidence
                    boxes = result.boxes
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)
                    max_idx = np.argmax(confs)
                    label = model.names[classes[max_idx]]
                    confidence = float(confs[max_idx])
                else:
                    label = "no detection"
                    confidence = 0.0
            else:
                label = "no detection"
                confidence = 0.0

            result = {
                "label": label,
                "confidence": round(confidence, 3),
                "latency_ms": round(latency_ms, 3),
            }

            payload = {
                "ts": datetime.now().isoformat(),
                "mode": mode,
                "input": src,
                "backend": f"yolov10-{model_name}",
                "result": result,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

            logger.info(f"Infer: {result}")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Stopped.")
