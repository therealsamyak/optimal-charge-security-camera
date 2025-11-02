"""
Hybrid controller supporting both rule-based and ML-based approaches.

This module provides a unified interface that can switch between rule-based
and ML-based controllers, enabling smooth transition to trained models.
"""

from typing import Dict, Optional, Union
from enum import Enum
from loguru import logger

from .intelligent_controller import (
    ModelController,
    ControllerDecision,
)

try:
    from .ml_controller import MLController, MLFeatures
except ImportError:
    MLController = None
    MLFeatures = None
try:
    from .training_data import TrainingDataCollector, TrainingExample
except ImportError:
    TrainingDataCollector = None
    TrainingExample = None
from config.manager import config


class ControllerType(Enum):
    """Types of controllers available."""

    RULE_BASED = "rule_based"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"


class HybridController:
    """
    Hybrid controller that can use rule-based or ML-based decision making.

    This controller provides a unified interface while supporting the transition
    from rule-based to ML-based control as training data becomes available.
    """

    def __init__(self, controller_type: ControllerType = ControllerType.RULE_BASED):
        """
        Initialize hybrid controller.

        Args:
            controller_type: Type of controller to use
        """
        self.controller_type = controller_type
        self.rule_controller = ModelController()
        self.ml_controller = MLController() if MLController else None
        self.training_collector = (
            TrainingDataCollector() if TrainingDataCollector else None
        )

        # Load ML model if available and requested
        if controller_type in [ControllerType.ML_BASED, ControllerType.HYBRID]:
            model_path = config.get("controller.ml_model_path", None)
            if model_path:
                self.ml_controller.load_model(model_path)
                logger.info(f"Loaded ML model from {model_path}")
            else:
                logger.warning("No ML model path specified, using rule-based fallback")

        logger.info(f"Initialized {controller_type.value} controller")

    def select_optimal_model(
        self,
        battery_level: float,
        energy_cleanliness: float,
        is_charging: bool,
        energy_source: str = "grid",
        user_requirements: Optional[Dict[str, float]] = None,
        record_training_data: bool = True,
    ) -> ControllerDecision:
        """
        Select optimal model using configured controller type.

        Args:
            battery_level: Current battery percentage (0-100)
            energy_cleanliness: Current energy cleanliness (0-100)
            is_charging: Whether battery is currently charging
            energy_source: Current energy source
            user_requirements: User requirements (uses config if None)
            record_training_data: Whether to record decision for training

        Returns:
            Controller decision with selected model and reasoning
        """
        # Use default user requirements if not provided
        if user_requirements is None:
            user_requirements = {
                "min_accuracy": config.get("requirements.min_accuracy", 80.0),
                "max_latency_ms": config.get("requirements.max_latency_ms", 100.0),
                "run_frequency_ms": config.get("requirements.run_frequency_ms", 2000.0),
            }

        # Get decision based on controller type
        if self.controller_type == ControllerType.RULE_BASED:
            decision = self.rule_controller.select_optimal_model(
                battery_level, energy_cleanliness, is_charging
            )

        elif self.controller_type == ControllerType.ML_BASED:
            # Extract features for ML controller
            import time

            time_struct = time.localtime()
            time_of_day = time_struct.tm_hour + time_struct.tm_min / 60.0
            day_of_week = time_struct.tm_wday

            features = self.ml_controller.extract_features(
                battery_level=battery_level,
                energy_cleanliness=energy_cleanliness,
                is_charging=is_charging,
                energy_source=energy_source,
                time_of_day=time_of_day,
                day_of_week=day_of_week,
                user_requirements=user_requirements,
            )

            decision = self.ml_controller.predict(features)

        elif self.controller_type == ControllerType.HYBRID:
            decision = self._hybrid_decision(
                battery_level,
                energy_cleanliness,
                is_charging,
                energy_source,
                user_requirements,
            )

        else:
            raise ValueError(f"Unknown controller type: {self.controller_type}")

        # Record training data if requested
        if record_training_data:
            self._record_decision_for_training(
                decision,
                battery_level,
                energy_cleanliness,
                is_charging,
                energy_source,
                user_requirements,
            )

        return decision

    def _hybrid_decision(
        self,
        battery_level: float,
        energy_cleanliness: float,
        is_charging: bool,
        energy_source: str,
        user_requirements: Dict[str, float],
    ) -> ControllerDecision:
        """
        Make hybrid decision combining rule-based and ML approaches.

        Args:
            battery_level: Current battery level
            energy_cleanliness: Current energy cleanliness
            is_charging: Whether currently charging
            energy_source: Current energy source
            user_requirements: User requirements

        Returns:
            Hybrid controller decision
        """
        # Get rule-based decision
        rule_decision = self.rule_controller.select_optimal_model(
            battery_level, energy_cleanliness, is_charging
        )

        # Get ML-based decision
        import time

        time_struct = time.localtime()
        time_of_day = time_struct.tm_hour + time_struct.tm_min / 60.0
        day_of_week = time_struct.tm_wday

        features = self.ml_controller.extract_features(
            battery_level=battery_level,
            energy_cleanliness=energy_cleanliness,
            is_charging=is_charging,
            energy_source=energy_source,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            user_requirements=user_requirements,
        )

        ml_decision = self.ml_controller.predict(features)

        # Combine decisions based on confidence and conditions
        if self.ml_controller.model is None:
            # No ML model available, use rule-based
            return rule_decision

        # Use ML decision if confidence is high, otherwise use rule-based
        ml_confidence = ml_decision.score
        rule_confidence = rule_decision.score

        # Weight decisions based on confidence
        if ml_confidence > rule_confidence + 10:  # ML significantly more confident
            selected_decision = ml_decision
            reasoning_prefix = "ML (high confidence)"
        elif rule_confidence > ml_confidence + 10:  # Rule significantly more confident
            selected_decision = rule_decision
            reasoning_prefix = "Rule (high confidence)"
        else:
            # Similar confidence, prefer rule-based for safety
            selected_decision = rule_decision
            reasoning_prefix = "Rule (safety preference)"

        # Update reasoning to indicate hybrid nature
        selected_decision.reasoning = (
            f"{reasoning_prefix} | {selected_decision.reasoning}"
        )

        return selected_decision

    def _record_decision_for_training(
        self,
        decision: ControllerDecision,
        battery_level: float,
        energy_cleanliness: float,
        is_charging: bool,
        energy_source: str,
        user_requirements: Dict[str, float],
    ) -> None:
        """Record decision for future ML training."""
        # Create placeholder performance outcomes
        # In real implementation, these would be actual measured values
        performance_outcomes = {
            "accuracy": 0.0,  # Will be updated after inference
            "latency_ms": 0.0,  # Will be updated after inference
            "battery_consumption": 0.0,  # Will be updated after inference
        }

        self.training_collector.record_decision(
            decision=decision,
            battery_level=battery_level,
            energy_cleanliness=energy_cleanliness,
            is_charging=is_charging,
            energy_source=energy_source,
            user_requirements=user_requirements,
            performance_outcomes=performance_outcomes,
            reasoning=decision.reasoning,
        )

    def update_performance_outcome(
        self, accuracy: float, latency_ms: float, battery_consumption: float
    ) -> None:
        """
        Update the most recent training example with actual performance outcomes.

        Args:
            accuracy: Actual inference accuracy
            latency_ms: Actual inference latency
            battery_consumption: Actual battery consumption
        """
        if self.training_collector.training_examples:
            last_example = self.training_collector.training_examples[-1]
            last_example.actual_accuracy = accuracy
            last_example.actual_latency = latency_ms
            last_example.actual_battery_consumption = battery_consumption

            # Recalculate user satisfaction
            user_requirements = {
                "min_accuracy": last_example.user_min_accuracy,
                "max_latency_ms": last_example.user_max_latency,
                "run_frequency_ms": last_example.user_run_frequency,
            }

            performance_outcomes = {
                "accuracy": accuracy,
                "latency_ms": latency_ms,
                "battery_consumption": battery_consumption,
            }

            last_example.user_satisfaction = (
                self.training_collector._calculate_user_satisfaction(
                    user_requirements, performance_outcomes
                )
            )

    def train_ml_model(self, min_examples: int = 1000) -> Dict[str, Union[float, str]]:
        """
        Train ML model on collected data.

        Args:
            min_examples: Minimum number of examples required for training

        Returns:
            Training metrics
        """
        if len(self.training_collector.training_examples) < min_examples:
            return {
                "error": f"Insufficient training data: {len(self.training_collector.training_examples)} < {min_examples}"
            }

        logger.info(
            f"Training ML model with {len(self.training_collector.training_examples)} examples"
        )

        # Train the model
        metrics = self.ml_controller.train(self.training_collector.training_examples)

        if "error" not in metrics:
            # Save the trained model
            model_path = config.get(
                "controller.ml_model_path", "src/data/trained_model.pkl"
            )
            self.ml_controller.save_model(model_path)

            # Optionally switch to ML-based controller
            if config.get("controller.auto_switch_to_ml", False):
                self.controller_type = ControllerType.ML_BASED
                logger.info("Switched to ML-based controller")

        return metrics

    def save_training_data(self) -> None:
        """Save collected training data."""
        self.training_collector.save_training_data()

    def get_training_summary(self) -> Dict[str, Union[int, float]]:
        """Get summary of collected training data."""
        return self.training_collector.get_training_summary()

    def switch_controller_type(self, controller_type: ControllerType) -> None:
        """
        Switch the active controller type.

        Args:
            controller_type: New controller type to use
        """
        old_type = self.controller_type
        self.controller_type = controller_type

        logger.info(
            f"Switched from {old_type.value} to {controller_type.value} controller"
        )

    def get_controller_info(self) -> Dict[str, Union[str, bool, int]]:
        """Get information about current controller state."""
        return {
            "controller_type": self.controller_type.value,
            "ml_model_loaded": self.ml_controller.model is not None,
            "training_examples_count": len(self.training_collector.training_examples),
            "rule_controller_available": self.rule_controller is not None,
            "ml_controller_available": self.ml_controller is not None,
        }

    def get_available_models(self) -> list:
        """Get list of available models."""
        return self.rule_controller.get_available_models()

    def get_model_profile(self, model_name: str):
        """Get profile for a specific model."""
        return self.rule_controller.get_model_profile(model_name)
