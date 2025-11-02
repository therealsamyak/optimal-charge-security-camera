"""
YOLOv10 model management and performance tracking.

This module provides a wrapper for YOLOv10 models with performance profiling
and dynamic model loading/unloading capabilities.
"""

import psutil
import gc
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from config import config
from utils import Timer, write_csv_log, format_timestamp


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""

    model_name: str
    accuracy: float
    latency_ms: float
    battery_consumption: float
    inference_count: int
    total_confidence: float
    successful_detections: int
    memory_usage_mb: float
    cpu_usage_percent: float


@dataclass
class ModelProfile:
    """Profile information for a YOLOv10 model."""

    name: str
    accuracy: float  # Expected accuracy percentage (0-100)
    latency_ms: float  # Expected latency in milliseconds
    battery_consumption: float  # Battery consumption per inference
    memory_usage_mb: float  # Expected memory usage in MB
    model_size_mb: float  # Model file size in MB
    parameters: int  # Number of parameters


class YOLOv10Manager:
    """Manager for YOLOv10 model family with performance tracking."""

    def __init__(self):
        """Initialize model manager."""
        if YOLO is None:
            raise RuntimeError("ultralytics package is required but not installed")

        self.available_models = config.get("models.available", ["yolov10n"])
        self.default_model = config.get("models.default", "yolov10n")
        self.current_model_name = None
        self.current_model = None
        self.performance_history: Dict[str, List[ModelPerformance]] = {}
        self.model_profiles = self._initialize_model_profiles()
        self.loaded_models: Dict[str, YOLO] = {}  # Cache for loaded models
        self.max_cached_models = 2  # Maximum models to keep in memory

        # Initialize performance tracking for all models
        for model_name in self.available_models:
            self.performance_history[model_name] = []

    def _initialize_model_profiles(self) -> Dict[str, ModelProfile]:
        """Initialize performance profiles from CSV data."""
        # Load from CSV using the model_loader
        try:
            from .model_loader import get_model_loader

            model_loader = get_model_loader()
            csv_profiles = model_loader.get_all_profiles()

            # Convert CSV profiles to YOLO manager profiles
            profiles = {}
            for name, csv_profile in csv_profiles.items():
                # Estimate memory usage and model size based on model size
                memory_mb = self._estimate_memory_usage(csv_profile.size_rank)
                model_size_mb = self._estimate_model_size(csv_profile.size_rank)
                parameters = self._estimate_parameters(csv_profile.size_rank)

                # Normalize accuracy to 60-100 range
                normalized_accuracy = self._normalize_accuracy(csv_profile.accuracy)

                profiles[name] = ModelProfile(
                    name=name,
                    accuracy=normalized_accuracy,
                    latency_ms=csv_profile.latency_ms,
                    battery_consumption=csv_profile.battery_consumption,
                    memory_usage_mb=memory_mb,
                    model_size_mb=model_size_mb,
                    parameters=parameters,
                )

            return profiles

        except Exception as e:
            logger.warning(f"Failed to load CSV profiles, using fallback: {e}")
            # YOLOv10 models only with normalized accuracy
            return {
                "yolov10n": ModelProfile(
                    "yolov10n",
                    accuracy=self._normalize_accuracy(39.5),
                    latency_ms=1.56,
                    battery_consumption=0.1,
                    memory_usage_mb=200.0,
                    model_size_mb=6.0,
                    parameters=2_300_000,
                ),
                "yolov10s": ModelProfile(
                    "yolov10s",
                    accuracy=self._normalize_accuracy(46.7),
                    latency_ms=2.66,
                    battery_consumption=0.2,
                    memory_usage_mb=350.0,
                    model_size_mb=12.0,
                    parameters=7_200_000,
                ),
                "yolov10m": ModelProfile(
                    "yolov10m",
                    accuracy=self._normalize_accuracy(51.3),
                    latency_ms=5.48,
                    battery_consumption=0.4,
                    memory_usage_mb=600.0,
                    model_size_mb=25.0,
                    parameters=25_900_000,
                ),
                "yolov10b": ModelProfile(
                    "yolov10b",
                    accuracy=self._normalize_accuracy(52.7),
                    latency_ms=6.54,
                    battery_consumption=0.6,
                    memory_usage_mb=900.0,
                    model_size_mb=45.0,
                    parameters=53_800_000,
                ),
                "yolov10l": ModelProfile(
                    "yolov10l",
                    accuracy=self._normalize_accuracy(53.3),
                    latency_ms=8.33,
                    battery_consumption=0.8,
                    memory_usage_mb=1400.0,
                    model_size_mb=85.0,
                    parameters=86_200_000,
                ),
                "yolov10x": ModelProfile(
                    "yolov10x",
                    accuracy=self._normalize_accuracy(54.4),
                    latency_ms=12.2,
                    battery_consumption=1.0,
                    memory_usage_mb=2200.0,
                    model_size_mb=160.0,
                    parameters=143_000_000,
                ),
            }

    def _estimate_memory_usage(self, size_rank: int) -> float:
        """Estimate memory usage based on size rank."""
        # Base memory usage increases exponentially with size
        base_memory = 150.0  # Base for smallest models
        return base_memory * (1.5 ** (size_rank - 1))

    def _estimate_model_size(self, size_rank: int) -> float:
        """Estimate model file size based on size rank."""
        # Base model size increases exponentially with size
        base_size = 5.0  # Base for smallest models in MB
        return base_size * (2.0 ** (size_rank - 1))

    def _estimate_parameters(self, size_rank: int) -> int:
        """Estimate parameter count based on size rank."""
        # Base parameters increase exponentially with size
        base_params = 2_000_000  # Base for smallest models
        return int(base_params * (3.0 ** (size_rank - 1)))

    def _normalize_accuracy(self, raw_accuracy: float) -> float:
        """Normalize accuracy from COCO mAP (0-100) to 60-95 range."""
        # YOLOv10 models range from ~39.5% to 54.4% COCO mAP
        # Map this range to 60-95 for user-facing accuracy
        min_raw = 39.5
        max_raw = 54.4
        min_normalized = 60.0
        max_normalized = 95.0

        # Linear normalization
        if raw_accuracy <= min_raw:
            return min_normalized
        elif raw_accuracy >= max_raw:
            return max_normalized
        else:
            # Scale to 60-100 range
            ratio = (raw_accuracy - min_raw) / (max_raw - min_raw)
            return min_normalized + ratio * (max_normalized - min_normalized)

    def load_model(self, model_name: str) -> bool:
        """
        Load a YOLOv10 model with caching and memory management.

        Args:
            model_name: Name of model to load (e.g., 'yolov10n')

        Returns:
            True if model loaded successfully
        """
        if model_name not in self.available_models:
            logger.error(
                f"Model {model_name} not in available models: {self.available_models}"
            )
            return False

        try:
            # Check if model is already cached
            if model_name in self.loaded_models:
                self.current_model = self.loaded_models[model_name]
                self.current_model_name = model_name
                logger.debug(f"Using cached model: {model_name}")
                return True

            # Manage memory by unloading least recently used models if cache is full
            if len(self.loaded_models) >= self.max_cached_models:
                self._unload_oldest_model()

            # Load new model
            logger.info(f"Loading model: {model_name}")
            models_dir = Path(config.get("models.models_dir", "src/models/cache"))
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / f"{model_name}.pt"
            model = YOLO(str(model_path))

            # Cache the model
            self.loaded_models[model_name] = model
            self.current_model = model
            self.current_model_name = model_name

            # Log model info
            profile = self.model_profiles.get(model_name)
            if profile:
                logger.info(
                    f"Model {model_name} loaded successfully - "
                    f"Size: {profile.model_size_mb:.1f}MB, "
                    f"Params: {profile.parameters:,}, "
                    f"Expected accuracy: {profile.accuracy:.1f}%"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def _unload_oldest_model(self) -> None:
        """Unload oldest cached model to free memory."""
        if not self.loaded_models:
            return

        # Simple strategy: remove first model (can be improved with LRU)
        oldest_model = next(iter(self.loaded_models))

        # Don't unload current model
        if oldest_model == self.current_model_name:
            # Find another model to unload
            for model_name in self.loaded_models:
                if model_name != self.current_model_name:
                    oldest_model = model_name
                    break

        if oldest_model != self.current_model_name:
            del self.loaded_models[oldest_model]
            logger.info(f"Unloaded model from cache: {oldest_model}")

            # Force garbage collection
            gc.collect()

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model from cache.

        Args:
            model_name: Name of model to unload

        Returns:
            True if model was unloaded
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]

            # Update current model if necessary
            if self.current_model_name == model_name:
                self.current_model = None
                self.current_model_name = None

            logger.info(f"Unloaded model: {model_name}")
            gc.collect()
            return True

        return False

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)

    def run_inference(
        self, image, model_name: Optional[str] = None
    ) -> Tuple[Optional[object], Dict]:
        """
        Run inference with specified model.

        Args:
            image: Input image for inference
            model_name: Model to use (if None, uses current model)

        Returns:
            Tuple of (results, performance_metrics)
        """
        if model_name is None:
            model_name = self.current_model_name

        if model_name is None:
            logger.error("No model specified and no current model loaded")
            return None, {}

        # Load model if needed
        if not self.load_model(model_name):
            return None, {}

        try:
            # Get system metrics before inference
            memory_before = self.get_memory_usage()
            cpu_before = self.get_cpu_usage()

            # Run inference with timing
            with Timer("inference") as timer:
                results = self.current_model(image, verbose=False)

            # Get system metrics after inference
            memory_after = self.get_memory_usage()
            cpu_after = self.get_cpu_usage()

            # Extract detection metrics
            detection_metrics = self._extract_detection_metrics(results)

            # Record performance
            performance = ModelPerformance(
                model_name=model_name,
                accuracy=detection_metrics.get("confidence", 0.0) * 100,
                latency_ms=timer.duration_ms,
                battery_consumption=self._get_battery_consumption(model_name),
                inference_count=1,
                total_confidence=detection_metrics.get("confidence", 0.0),
                successful_detections=1
                if detection_metrics.get("has_detection", False)
                else 0,
                memory_usage_mb=memory_after - memory_before,
                cpu_usage_percent=cpu_after - cpu_before,
            )

            self.performance_history[model_name].append(performance)

            # Prepare performance metrics dict
            metrics = {
                "model_name": model_name,
                "latency_ms": timer.duration_ms,
                "confidence": detection_metrics.get("confidence", 0.0),
                "has_detection": detection_metrics.get("has_detection", False),
                "label": detection_metrics.get("label", "no detection"),
                "battery_consumption": performance.battery_consumption,
                "memory_usage_mb": performance.memory_usage_mb,
                "cpu_usage_percent": performance.cpu_usage_percent,
            }

            return results, metrics

        except Exception as e:
            logger.error(f"Inference failed with model {model_name}: {e}")
            return None, {}

    def _extract_detection_metrics(self, results) -> Dict:
        """Extract metrics from YOLO results, focusing on human detection."""
        try:
            if results and len(results) > 0:
                result = results[0]
                if len(result.boxes) > 0:
                    boxes = result.boxes
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)

                    # Filter for human detections only (class 0 is 'person' in COCO dataset)
                    human_indices = [i for i, cls in enumerate(classes) if cls == 0]

                    if human_indices:
                        # Get the human detection with highest confidence
                        human_confidences = [confs[i] for i in human_indices]
                        best_human_idx = human_indices[
                            human_confidences.index(max(human_confidences))
                        ]
                        confidence = float(confs[best_human_idx])

                        return {
                            "has_detection": True,
                            "label": "person",
                            "confidence": confidence,
                            "detection_count": len(human_indices),
                        }
                    else:
                        # No humans detected, but other objects were detected
                        return {
                            "has_detection": False,
                            "label": "no human detected",
                            "confidence": 0.0,
                            "detection_count": 0,
                        }
                else:
                    return {
                        "has_detection": False,
                        "label": "no detection",
                        "confidence": 0.0,
                        "detection_count": 0,
                    }
            else:
                return {
                    "has_detection": False,
                    "label": "no detection",
                    "confidence": 0.0,
                    "detection_count": 0,
                }
        except Exception as e:
            logger.error(f"Failed to extract detection metrics: {e}")
            return {
                "has_detection": False,
                "label": "error",
                "confidence": 0.0,
                "detection_count": 0,
            }

    def _get_battery_consumption(self, model_name: str) -> float:
        """Get battery consumption rate for a model."""
        profile = self.model_profiles.get(model_name)
        return profile.battery_consumption if profile else 0.5

    def get_model_profile(self, model_name: str) -> Optional[ModelProfile]:
        """Get profile for a specific model."""
        return self.model_profiles.get(model_name)

    def get_model_performance_summary(self, model_name: str) -> Optional[Dict]:
        """Get performance summary for a specific model."""
        if model_name not in self.performance_history:
            return None

        history = self.performance_history[model_name]
        if not history:
            return None

        total_inferences = len(history)
        total_latency = sum(p.latency_ms for p in history)
        total_confidence = sum(p.total_confidence for p in history)
        successful_detections = sum(p.successful_detections for p in history)
        total_memory = sum(p.memory_usage_mb for p in history)
        total_cpu = sum(p.cpu_usage_percent for p in history)

        return {
            "model_name": model_name,
            "total_inferences": total_inferences,
            "avg_latency_ms": total_latency / total_inferences,
            "avg_confidence": total_confidence / total_inferences,
            "detection_rate": successful_detections / total_inferences,
            "total_battery_consumption": sum(p.battery_consumption for p in history),
            "avg_memory_usage_mb": total_memory / total_inferences,
            "avg_cpu_usage_percent": total_cpu / total_inferences,
        }

    def get_all_performance_summaries(self) -> Dict[str, Dict]:
        """Get performance summaries for all models."""
        summaries = {}
        for model_name in self.available_models:
            summary = self.get_model_performance_summary(model_name)
            if summary:
                summaries[model_name] = summary
        return summaries

    def export_performance_data(self, csv_path: str) -> None:
        """Export performance data to CSV file."""
        for model_name, history in self.performance_history.items():
            for performance in history:
                log_entry = {
                    "timestamp": format_timestamp(),
                    "model_name": performance.model_name,
                    "accuracy": performance.accuracy,
                    "latency_ms": performance.latency_ms,
                    "battery_consumption": performance.battery_consumption,
                    "confidence": performance.total_confidence,
                    "has_detection": performance.successful_detections > 0,
                    "memory_usage_mb": performance.memory_usage_mb,
                    "cpu_usage_percent": performance.cpu_usage_percent,
                }
                write_csv_log(csv_path, log_entry)

    def get_current_model(self) -> Optional[str]:
        """Get currently loaded model name."""
        return self.current_model_name

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return self.available_models.copy()

    def get_cached_models(self) -> List[str]:
        """Get list of currently cached models."""
        return list(self.loaded_models.keys())

    def clear_cache(self) -> None:
        """Clear all cached models."""
        self.loaded_models.clear()
        self.current_model = None
        self.current_model_name = None
        gc.collect()
        logger.info("Model cache cleared")
