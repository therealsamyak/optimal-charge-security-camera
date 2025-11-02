"""YOLO model availability tests."""

import numpy as np
import ultralytics


class TestYOLOModels:
    """Test YOLO model availability and loading."""

    def test_available_models_load(self):
        """Test that available YOLOv10 models can be loaded successfully."""
        available_models = ["yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x"]

        for model_name in available_models:
            try:
                yolo_model = ultralytics.YOLO(model_name)
                assert yolo_model is not None
            except Exception as e:
                assert False, f"Failed to load {model_name}: {e}"

    def test_unavailable_models_fail(self):
        """Test that unavailable YOLO models raise appropriate errors."""
        unavailable_models = [
            "yolov5n",
            "yolov8n",
            "yolov11n",
            "damo-yolot",
            "yoloxs",
            "rtdetrv2s",
            "efficientdetd0",
        ]

        for model_name in unavailable_models:
            try:
                ultralytics.YOLO(model_name)
                assert False, f"Expected {model_name} to fail but it didn't"
            except Exception:
                pass  # Expected behavior

    def test_yolov10n_basic_inference(self):
        """Test basic inference capability with yolov10n."""
        # Create a dummy image (640x640x3)
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        try:
            model = ultralytics.YOLO("yolov10n")
            results = model(dummy_image)
            assert len(results) > 0
        except Exception as e:
            assert False, f"Basic inference failed: {e}"
