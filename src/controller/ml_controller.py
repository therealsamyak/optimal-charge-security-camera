"""
ML-based controller interface for future training.

This module provides the interface and structure for a machine learning
based controller that can be trained on collected data.
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from .intelligent_controller import ControllerDecision
from .training_data import TrainingExample
from utils.helpers import clamp


@dataclass
class MLFeatures:
    """Features for ML model input."""

    # Battery and energy features
    battery_level: float
    energy_cleanliness: float
    is_charging: bool
    energy_source_encoded: float  # One-hot encoded

    # Temporal features
    time_of_day_sin: float  # Cyclical encoding
    time_of_day_cos: float
    day_of_week_sin: float
    day_of_week_cos: float

    # User requirements
    user_min_accuracy: float
    user_max_latency: float
    user_run_frequency: float

    # Context features
    battery_trend: float  # Rate of change
    energy_trend: float  # Rate of change

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model."""
        return np.array(
            [
                self.battery_level,
                self.energy_cleanliness,
                float(self.is_charging),
                self.energy_source_encoded,
                self.time_of_day_sin,
                self.time_of_day_cos,
                self.day_of_week_sin,
                self.day_of_week_cos,
                self.user_min_accuracy,
                self.user_max_latency,
                self.user_run_frequency,
                self.battery_trend,
                self.energy_trend,
            ]
        )


@dataclass
class MLTargets:
    """Targets for ML model training."""

    # Model selection (multi-class classification)
    selected_model_encoded: int  # Model index

    # Charging decision (binary classification)
    should_charge: bool

    # Performance prediction (regression)
    predicted_accuracy: float
    predicted_latency: float
    predicted_battery_consumption: float

    # User satisfaction (regression)
    user_satisfaction: float


class MLController:
    """
    Machine Learning based controller interface.

    This class provides the structure for training and using ML models
    for intelligent controller decisions. Currently implements placeholder
    methods that can be replaced with actual ML implementations.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ML controller.

        Args:
            model_path: Path to trained model file
        """
        self.model_path = model_path
        self.model = None
        self.feature_scaler = None
        self.model_encoder = {}  # Maps model names to indices

        # Available models (same as rule-based controller)
        self.available_models = [
            "yolov10n",
            "yolov10s",
            "yolov10m",
            "yolov10b",
            "yolov10l",
            "yolov10x",
        ]
        self._initialize_model_encoder()

        # History for trend calculation
        self.battery_history = []
        self.energy_history = []
        self.max_history_length = 10

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def _initialize_model_encoder(self) -> None:
        """Initialize model name to index encoding."""
        for i, model_name in enumerate(self.available_models):
            self.model_encoder[model_name] = i

    def extract_features(
        self,
        battery_level: float,
        energy_cleanliness: float,
        is_charging: bool,
        energy_source: str,
        time_of_day: float,
        day_of_week: int,
        user_requirements: Dict[str, float],
    ) -> MLFeatures:
        """
        Extract ML features from current state.

        Args:
            battery_level: Current battery level
            energy_cleanliness: Current energy cleanliness
            is_charging: Whether currently charging
            energy_source: Current energy source
            time_of_day: Time of day (0-24)
            day_of_week: Day of week (0-6)
            user_requirements: User requirements

        Returns:
            MLFeatures object
        """
        # Update history for trend calculation
        self.battery_history.append(battery_level)
        self.energy_history.append(energy_cleanliness)

        # Limit history length
        if len(self.battery_history) > self.max_history_length:
            self.battery_history.pop(0)
        if len(self.energy_history) > self.max_history_length:
            self.energy_history.pop(0)

        # Calculate trends
        battery_trend = self._calculate_trend(self.battery_history)
        energy_trend = self._calculate_trend(self.energy_history)

        # Encode energy source (simple one-hot)
        source_encoding = self._encode_energy_source(energy_source)

        # Cyclical encoding for temporal features
        time_sin = np.sin(2 * np.pi * time_of_day / 24)
        time_cos = np.cos(2 * np.pi * time_of_day / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        return MLFeatures(
            battery_level=battery_level,
            energy_cleanliness=energy_cleanliness,
            is_charging=is_charging,
            energy_source_encoded=source_encoding,
            time_of_day_sin=time_sin,
            time_of_day_cos=time_cos,
            day_of_week_sin=day_sin,
            day_of_week_cos=day_cos,
            user_min_accuracy=user_requirements["min_accuracy"],
            user_max_latency=user_requirements["max_latency_ms"],
            user_run_frequency=user_requirements["run_frequency_ms"],
            battery_trend=battery_trend,
            energy_trend=energy_trend,
        )

    def _calculate_trend(self, history: List[float]) -> float:
        """Calculate trend from history (positive = increasing)."""
        if len(history) < 2:
            return 0.0

        # Simple linear trend
        x = np.arange(len(history))
        y = np.array(history)

        # Calculate slope
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return clamp(slope, -1.0, 1.0)

        return 0.0

    def _encode_energy_source(self, energy_source: str) -> float:
        """Encode energy source as numeric value."""
        encoding_map = {"solar": 0.8, "wind": 0.7, "hydro": 0.6, "grid": 0.2}
        return encoding_map.get(energy_source, 0.0)

    def predict(self, features: MLFeatures) -> ControllerDecision:
        """
        Make prediction using ML model.

        Args:
            features: Extracted features

        Returns:
            Controller decision
        """
        if self.model is None:
            # Fallback to rule-based if no model trained
            return self._fallback_decision(features)

        try:
            # Convert features to array
            feature_array = features.to_array().reshape(1, -1)

            # Scale features if scaler available
            if self.feature_scaler:
                feature_array = self.feature_scaler.transform(feature_array)

            # Make predictions (placeholder - implement actual ML logic)
            model_pred, charging_pred, performance_pred = self._model_predict(
                feature_array
            )

            # Convert predictions back to decision
            selected_model = self._decode_model_prediction(model_pred)
            should_charge = bool(charging_pred > 0.5)

            # Calculate confidence score
            confidence = self._calculate_confidence(model_pred, charging_pred)

            # Generate reasoning
            reasoning = self._generate_ml_reasoning(
                features, selected_model, should_charge, confidence
            )

            return ControllerDecision(
                selected_model=selected_model,
                should_charge=should_charge,
                score=confidence,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._fallback_decision(features)

    def _model_predict(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Placeholder for actual ML model prediction.

        This should be replaced with trained model inference.
        """
        # Placeholder: return random predictions
        batch_size = features.shape[0]

        # Model selection (6 classes)
        model_pred = np.random.rand(batch_size, 6)

        # Charging decision (binary)
        charging_pred = np.random.rand(batch_size, 1)

        # Performance prediction (3 values)
        performance_pred = np.random.rand(batch_size, 3)

        return model_pred[0], charging_pred[0][0], performance_pred[0]

    def _decode_model_prediction(self, prediction: np.ndarray) -> str:
        """Decode model prediction to model name."""
        model_index = np.argmax(prediction)
        return self.available_models[model_index]

    def _calculate_confidence(
        self, model_pred: np.ndarray, charging_pred: float
    ) -> float:
        """Calculate confidence score from predictions."""
        # Model confidence (max probability)
        model_confidence = np.max(model_pred)

        # Charging confidence
        charging_confidence = max(charging_pred, 1 - charging_pred)

        # Combined confidence
        return (model_confidence + charging_confidence) / 2.0 * 100

    def _generate_ml_reasoning(
        self,
        features: MLFeatures,
        selected_model: str,
        should_charge: bool,
        confidence: float,
    ) -> str:
        """Generate reasoning for ML decision."""
        parts = [
            f"ML model selected {selected_model}",
            f"Confidence: {confidence:.1f}%",
            f"Charging: {'Yes' if should_charge else 'No'}",
            f"Battery: {features.battery_level:.1f}%, Energy: {features.energy_cleanliness:.1f}%",
        ]

        return " | ".join(parts)

    def _fallback_decision(self, features: MLFeatures) -> ControllerDecision:
        """Fallback decision when ML model is not available."""
        # Simple rule-based fallback
        if features.battery_level < 30:
            should_charge = True
            selected_model = "yolov10n"  # Most efficient
        elif features.energy_cleanliness > 80 and features.battery_level < 70:
            should_charge = True
            selected_model = "yolov10m"  # Balanced
        else:
            should_charge = False
            selected_model = "yolov10s"  # Default balanced

        reasoning = f"Fallback rule-based decision - Battery: {features.battery_level:.1f}%, Energy: {features.energy_cleanliness:.1f}%"

        return ControllerDecision(
            selected_model=selected_model,
            should_charge=should_charge,
            score=50.0,  # Medium confidence
            reasoning=reasoning,
        )

    def train(self, training_data: List[TrainingExample]) -> Dict[str, float]:
        """
        Train ML model on collected data.

        Args:
            training_data: List of training examples

        Returns:
            Training metrics
        """
        if len(training_data) < 100:
            logger.warning("Insufficient training data (need at least 100 examples)")
            return {"error": "Insufficient data"}

        try:
            # Convert training data to features and targets
            X, y = self._prepare_training_data(training_data)

            # Placeholder training logic
            metrics = self._train_model(X, y)

            logger.info(f"ML model trained with {len(training_data)} examples")
            return metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"error": str(e)}

    def _prepare_training_data(
        self, training_data: List[TrainingExample]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML model."""
        features = []
        targets = []

        for example in training_data:
            # Extract features
            feature_obj = self.extract_features(
                battery_level=example.battery_level,
                energy_cleanliness=example.energy_cleanliness,
                is_charging=example.is_charging,
                energy_source=example.energy_source,
                time_of_day=example.time_of_day,
                day_of_week=example.day_of_week,
                user_requirements={
                    "min_accuracy": example.user_min_accuracy,
                    "max_latency_ms": example.user_max_latency,
                    "run_frequency_ms": example.user_run_frequency,
                },
            )

            features.append(feature_obj.to_array())

            # Prepare targets
            target = MLTargets(
                selected_model_encoded=self.model_encoder[example.selected_model],
                should_charge=example.should_charge,
                predicted_accuracy=example.actual_accuracy,
                predicted_latency=example.actual_latency,
                predicted_battery_consumption=example.actual_battery_consumption,
                user_satisfaction=example.user_satisfaction,
            )

            targets.append(
                [
                    target.selected_model_encoded,
                    float(target.should_charge),
                    target.predicted_accuracy,
                    target.predicted_latency,
                    target.predicted_battery_consumption,
                    target.user_satisfaction,
                ]
            )

        return np.array(features), np.array(targets)

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Placeholder for actual model training.

        This should be replaced with actual ML training algorithm
        (e.g., neural network, random forest, etc.).
        """
        # Placeholder: create simple model
        logger.info("Training placeholder ML model...")

        # In a real implementation, this would train:
        # - Multi-class classifier for model selection
        # - Binary classifier for charging decision
        # - Regression models for performance prediction

        return {
            "training_examples": len(X),
            "feature_dimensions": X.shape[1],
            "training_loss": 0.5,  # Placeholder
            "validation_accuracy": 0.75,  # Placeholder
        }

    def save_model(self, path: str) -> bool:
        """Save trained model to file."""
        if self.model is None:
            logger.warning("No model to save")
            return False

        try:
            model_data = {
                "model": self.model,
                "feature_scaler": self.feature_scaler,
                "model_encoder": self.model_encoder,
                "available_models": self.available_models,
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load trained model from file."""
        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.feature_scaler = model_data.get("feature_scaler")
            self.model_encoder = model_data["model_encoder"]
            self.available_models = model_data["available_models"]

            logger.info(f"Model loaded from {path}")
            return True

        except FileNotFoundError:
            logger.warning(f"Model file not found at {path}, using fallback behavior")
            self.model = None
            self.feature_scaler = None
            self.model_encoder = None
            self.available_models = [
                "yolov10n",
                "yolov10s",
                "yolov10m",
                "yolov10b",
                "yolov10l",
                "yolov10x",
            ]
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_loaded": self.model is not None,
            "model_path": self.model_path,
            "available_models": self.available_models,
            "feature_count": 14,  # Number of features in MLFeatures
            "history_length": len(self.battery_history),
        }
