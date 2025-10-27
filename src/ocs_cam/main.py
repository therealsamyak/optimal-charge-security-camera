from datetime import datetime
from loguru import logger
import json, os, time
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None


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
    src = os.environ.get("OCS_INPUT")
    runs = os.environ.get("OCS_RUNS_DIR", "runs")
    interval = float(os.environ.get("OCS_INTERVAL_SEC", "2.0"))

    os.makedirs(runs, exist_ok=True)
    log_path = os.path.join(runs, "logs.jsonl")

    logger.info(f"Pipeline start @ {datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"source={mode} input={src} interval={interval}s")
    logger.info(f"metrics -> {log_path} (Ctrl+C to stop)")

    try:
        while True:
            rgb = _read_frame(src, mode)

            # fake compute baseline
            t0 = time.perf_counter()
            _ = float(np.mean(rgb))
            latency_ms = (time.perf_counter() - t0) * 1000
            result = {
                "label": "person",
                "confidence": 0.42,
                "latency_ms": round(latency_ms, 3),
            }

            payload = {
                "ts": datetime.now().isoformat(),
                "mode": mode,
                "input": src,
                "backend": "fake",
                "result": result,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

            logger.info(f"Infer: {result}")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Stopped.")
