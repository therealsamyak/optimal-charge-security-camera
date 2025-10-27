from datetime import datetime
from loguru import logger
import numpy as np, time, os
try:
    from PIL import Image
except ImportError:
    Image = None

def get_frame(path: str | None = None):
    # Use an image if provided; else synthesize a dummy frame
    if path and Image and os.path.exists(path):
        return Image.open(path)
    return (np.random.rand(480, 640, 3) * 255).astype("uint8")

def fake_model_infer(frame) -> dict:
    # Placeholder “model” so we can wire timing/metrics early
    start = time.perf_counter()
    _ = np.mean(frame)
    latency_ms = (time.perf_counter() - start) * 1000
    return {"label": "person", "confidence": 0.42, "latency_ms": round(latency_ms, 2)}

def main():
    logger.info(f"Pipeline start @ {datetime.now().isoformat(timespec='seconds')}")
    frame = get_frame(os.environ.get("OCS_INPUT"))
    logger.info(f"Infer result: {fake_model_infer(frame)}")
    logger.info("Done.")

if __name__ == "__main__":
    main()
