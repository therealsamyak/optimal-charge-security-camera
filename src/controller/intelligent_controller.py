"""
Intelligent controller for optimal model selection and battery management.

This module implements the rule-based controller that balances battery level,
energy cleanliness, and user requirements to select the optimal YOLOv10 model.
"""

from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
from loguru import logger
from config import config
from utils.helpers import validate_user_requirements

if TYPE_CHECKING:
    pass


@dataclass
class ControllerModelProfile:
    """Performance profile for a YOLO model variant."""

    name: str
    accuracy: float  # Expected accuracy percentage (0-100)
    latency_ms: float  # Expected latency in milliseconds
    battery_consumption: float  # Battery consumption per inference
    size_rank: int  # Size ranking (1=smallest, 6=largest)


@dataclass
class ControllerDecision:
    """Result of controller decision making."""

    selected_model: str
    should_charge: bool
    score: float
    reasoning: str


class ModelController:
    """Intelligent controller for model selection and battery management."""

    def __init__(self):
        """Initialize the controller with model profiles and configuration."""
        # Lazy import to avoid circular dependencies
        self.model_loader = None
        self.model_profiles: Dict[str, ControllerModelProfile] = {}
        self.user_requirements = self._load_user_requirements()
        self.controller_config = self._load_controller_config()
        self._load_model_profiles()

    def _load_model_profiles(self):
        """Load model profiles from CSV data."""
        if self.model_loader is None:
            try:
                from models.model_loader import get_model_loader

                self.model_loader = get_model_loader()
                # Convert ModelProfile to ControllerModelProfile
                loader_profiles = self.model_loader.get_all_profiles()
                self.model_profiles = {}
                for name, profile in loader_profiles.items():
                    self.model_profiles[name] = ControllerModelProfile(
                        name=profile.name,
                        accuracy=profile.accuracy,
                        latency_ms=profile.latency_ms,
                        battery_consumption=profile.battery_consumption,
                        size_rank=profile.size_rank,
                    )
            except ImportError:
                # Fallback to hardcoded profiles if CSV loading fails
                self.model_profiles = self._initialize_fallback_profiles()

    def _initialize_model_profiles(self) -> Dict[str, ControllerModelProfile]:
        """Initialize performance profiles from CSV data."""
        # This method is kept for compatibility but profiles are loaded from CSV
        return self.model_profiles

    def _initialize_fallback_profiles(self) -> Dict[str, ControllerModelProfile]:
        """Initialize fallback performance profiles for YOLOv10 variants with normalized accuracy (60-95%)."""
        return {
            "yolov10n": ControllerModelProfile(
                "yolov10n",
                accuracy=60.0,
                latency_ms=1.56,
                battery_consumption=0.1,
                size_rank=1,
            ),
            "yolov10s": ControllerModelProfile(
                "yolov10s",
                accuracy=76.9,
                latency_ms=2.66,
                battery_consumption=0.2,
                size_rank=2,
            ),
            "yolov10m": ControllerModelProfile(
                "yolov10m",
                accuracy=87.7,
                latency_ms=5.48,
                battery_consumption=0.4,
                size_rank=3,
            ),
            "yolov10b": ControllerModelProfile(
                "yolov10b",
                accuracy=91.0,
                latency_ms=6.54,
                battery_consumption=0.6,
                size_rank=4,
            ),
            "yolov10l": ControllerModelProfile(
                "yolov10l",
                accuracy=92.4,
                latency_ms=8.33,
                battery_consumption=0.8,
                size_rank=5,
            ),
            "yolov10x": ControllerModelProfile(
                "yolov10x",
                accuracy=95.0,
                latency_ms=12.2,
                battery_consumption=1.0,
                size_rank=6,
            ),
        }

    def _load_user_requirements(self) -> Dict[str, float]:
        """Load and validate user requirements from configuration."""
        raw_requirements = {
            "min_accuracy": config.get("requirements.min_accuracy", 80.0),
            "max_latency_ms": config.get("requirements.max_latency_ms", 100.0),
            "run_frequency_ms": config.get("requirements.run_frequency_ms", 2000.0),
        }

        try:
            return validate_user_requirements(raw_requirements)
        except ValueError as e:
            logger.warning(f"Invalid user requirements: {e}. Using defaults.")
            return validate_user_requirements(
                {
                    "min_accuracy": 80.0,
                    "max_latency_ms": 100.0,
                    "run_frequency_ms": 2000.0,
                }
            )

    def _load_controller_config(self) -> Dict[str, float]:
        """Load controller-specific configuration."""
        return {
            "enable_charging": config.get("controller.enable_charging", True),
            "min_battery_threshold": config.get(
                "controller.min_battery_threshold", 20.0
            ),
            "max_battery_threshold": config.get(
                "controller.max_battery_threshold", 90.0
            ),
        }

    def calculate_model_score(
        self,
        model: ControllerModelProfile,
        battery_level: float,
        energy_cleanliness: float,
        is_charging: bool = False,
    ) -> float:
        """
        Calculate score for a model based on current conditions and resource constraints.

        Args:
            model: Model profile to score
            battery_level: Current battery percentage (0-100)
            energy_cleanliness: Current energy cleanliness (0-100)

        Returns:
            Score for the model (higher is better)
        """
        score = 0.0

        # Resource availability factors
        energy_cleanliness_factor = energy_cleanliness / 100.0  # 0-1 scale

        # Base requirements check (must pass minimum thresholds)
        meets_accuracy = model.accuracy >= self.user_requirements["min_accuracy"]
        meets_latency = model.latency_ms <= self.user_requirements["max_latency_ms"]

        # Heavy penalty for not meeting requirements
        if not meets_accuracy:
            score -= 50.0
        if not meets_latency:
            score -= 30.0

        # Resource-aware scoring based on conditions
        if battery_level >= 80.0 and energy_cleanliness >= 80.0:
            # HIGH RESOURCES: Use most powerful models
            # Prioritize accuracy and capability
            score += model.accuracy * 0.4  # Accuracy heavily weighted
            score += (100 - model.latency_ms) * 0.2  # Latency less important
            score += model.size_rank * 5.0  # Bonus for larger models
            score += energy_cleanliness_factor * 10.0  # Clean energy bonus

        elif battery_level >= 50.0 and energy_cleanliness >= 60.0:
            # MEDIUM RESOURCES: Balanced approach
            # Balance between performance and efficiency
            score += model.accuracy * 0.3
            score += (100 - model.latency_ms) * 0.3
            score += (1.0 - model.battery_consumption) * 20.0  # Efficiency matters
            score += model.size_rank * 2.0  # Moderate bonus for larger models

        elif battery_level >= 30.0:
            # LOW BATTERY: Prioritize efficiency
            # Focus on battery conservation
            score += (1.0 - model.battery_consumption) * 40.0  # Efficiency critical
            score += model.accuracy * 0.2  # Accuracy still important
            score += (100 - model.latency_ms) * 0.2  # Latency matters
            score -= model.size_rank * 3.0  # Penalty for larger models

        else:
            # CRITICAL BATTERY: Maximum conservation
            # Use smallest, most efficient models
            score += (1.0 - model.battery_consumption) * 50.0  # Efficiency paramount
            score -= model.size_rank * 10.0  # Heavy penalty for large models
            score += model.accuracy * 0.1  # Minimal accuracy consideration

            # Only allow models that meet basic requirements
            if not meets_accuracy or not meets_latency:
                score -= 100.0

        # Clean energy opportunistic bonus
        if energy_cleanliness >= 90.0 and battery_level < 80.0:
            # When energy is very clean, allow more powerful models even with moderate battery
            opportunistic_bonus = energy_cleanliness_factor * model.size_rank * 2.0
            score += opportunistic_bonus

        # Charging state consideration
        if is_charging and energy_cleanliness >= 70.0:
            # When charging with clean energy, can afford more powerful models
            charging_bonus = energy_cleanliness_factor * model.size_rank * 3.0
            score += charging_bonus

        return score

    def should_start_charging(
        self, battery_level: float, energy_cleanliness: float
    ) -> bool:
        """
        Determine if charging should be started.

        Args:
            battery_level: Current battery percentage (0-100)
            energy_cleanliness: Current energy cleanliness (0-100)

        Returns:
            True if charging should be started
        """
        if not self.controller_config["enable_charging"]:
            return False

        # Force charging if battery is critically low
        if battery_level <= self.controller_config["min_battery_threshold"]:
            return True

        # Charge if battery is low and energy is clean
        if battery_level <= 50.0 and energy_cleanliness >= 80.0:
            return True

        # Opportunistic charging when energy is very clean
        if (
            energy_cleanliness >= 90.0
            and battery_level <= self.controller_config["max_battery_threshold"]
        ):
            return True

        return False

    def should_stop_charging(
        self, battery_level: float, energy_cleanliness: float
    ) -> bool:
        """
        Determine if charging should be stopped.

        Args:
            battery_level: Current battery percentage (0-100)
            energy_cleanliness: Current energy cleanliness (0-100)

        Returns:
            True if charging should be stopped
        """
        # Stop if battery is full
        if battery_level >= self.controller_config["max_battery_threshold"]:
            return True

        # Stop if energy becomes dirty and battery is reasonable
        if energy_cleanliness < 40.0 and battery_level >= 60.0:
            return True

        return False

    def select_optimal_model(
        self, battery_level: float, energy_cleanliness: float, is_charging: bool
    ) -> ControllerDecision:
        """
        Select the optimal model based on current conditions.

        Args:
            battery_level: Current battery percentage (0-100)
            energy_cleanliness: Current energy cleanliness (0-100)
            is_charging: Whether battery is currently charging

        Returns:
            Controller decision with selected model and reasoning
        """
        # Filter models that meet minimum requirements
        viable_models = []
        for model in self.model_profiles.values():
            if (
                model.accuracy >= self.user_requirements["min_accuracy"]
                and model.latency_ms <= self.user_requirements["max_latency_ms"]
            ):
                viable_models.append(model)

        # If no models meet requirements, relax constraints
        if not viable_models:
            logger.warning("No models meet user requirements, relaxing constraints")
            viable_models = list(self.model_profiles.values())

        # Score all viable models
        scored_models = []
        for model in viable_models:
            score = self.calculate_model_score(
                model, battery_level, energy_cleanliness, is_charging
            )
            scored_models.append((model, score))

        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[1], reverse=True)

        # Select best model
        best_model, best_score = scored_models[0]

        # Determine charging decision
        should_charge = False
        charging_reasoning = ""

        if is_charging:
            if self.should_stop_charging(battery_level, energy_cleanliness):
                should_charge = False
                charging_reasoning = "Stop charging: battery sufficient or energy dirty"
            else:
                should_charge = True
                charging_reasoning = "Continue charging: conditions favorable"
        else:
            if self.should_start_charging(battery_level, energy_cleanliness):
                should_charge = True
                charging_reasoning = "Start charging: battery low or energy clean"
            else:
                should_charge = False
                charging_reasoning = "No charging needed"

        # Generate reasoning
        reasoning_parts = [
            f"Selected {best_model.name} (accuracy: {best_model.accuracy}%, latency: {best_model.latency_ms}ms)",
            f"Score: {best_score:.2f}",
            charging_reasoning,
            f"Battery: {battery_level:.1f}%, Energy cleanliness: {energy_cleanliness:.1f}%",
        ]

        reasoning = " | ".join(reasoning_parts)

        return ControllerDecision(
            selected_model=best_model.name,
            should_charge=should_charge,
            score=best_score,
            reasoning=reasoning,
        )

    def get_model_profile(self, model_name: str) -> Optional[ControllerModelProfile]:
        """Get profile for a specific model."""
        return self.model_profiles.get(model_name)

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.model_profiles.keys())
