"""Controller implementations for simulation."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseController(ABC):
    """Base class for all controllers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.accuracy_threshold = config["accuracy_threshold"]
        self.latency_threshold = config["latency_threshold_ms"]

    @abstractmethod
    def select_model(
        self,
        battery_level: float,
        energy_cleanliness: float,
        model_data: Dict[str, Dict[str, float]],
    ) -> Optional[str]:
        """Select YOLO model based on current conditions."""
        pass


class CustomController(BaseController):
    """Weighted scoring controller balancing multiple factors."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        weights = config["custom_controller_weights"]
        self.accuracy_weight = weights["accuracy_weight"]
        self.latency_weight = weights["latency_weight"]
        self.energy_weight = weights["energy_cleanliness_weight"]
        self.battery_weight = weights["battery_conservation_weight"]

    def select_model(
        self,
        battery_level: float,
        energy_cleanliness: float,
        model_data: Dict[str, Dict[str, float]],
    ) -> Optional[str]:
        """Select model using weighted scoring algorithm."""
        best_model = None
        best_score = -float("inf")

        for model_name, data in model_data.items():
            # Skip if model doesn't meet performance thresholds
            if data["accuracy"] < self.accuracy_threshold:
                continue
            if data["latency_ms"] > self.latency_threshold:
                continue

            # Calculate weighted score
            accuracy_score = data["accuracy"] * self.accuracy_weight
            latency_score = (1.0 / data["latency_ms"]) * self.latency_weight
            energy_score = energy_cleanliness * self.energy_weight
            battery_score = battery_level * self.battery_weight

            total_score = accuracy_score + latency_score + energy_score + battery_score

            if total_score > best_score:
                best_score = total_score
                best_model = model_name

        return best_model


class OracleController(BaseController):
    """MILP-based oracle controller with full future knowledge."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.horizon_hours = config["oracle_controller"]["optimization_horizon_hours"]
        self.time_step_minutes = config["oracle_controller"]["time_step_minutes"]
        self.clean_energy_bonus = config["oracle_controller"][
            "clean_energy_bonus_factor"
        ]

    def select_model(
        self,
        battery_level: float,
        energy_cleanliness: float,
        model_data: Dict[str, Dict[str, float]],
    ) -> Optional[str]:
        """Select model using MILP optimization (simplified version)."""
        # For now, return the most efficient model that meets thresholds
        # Full MILP implementation will be in simulation runner
        valid_models = []

        for model_name, data in model_data.items():
            if (
                data["accuracy"] >= self.accuracy_threshold
                and data["latency_ms"] <= self.latency_threshold
            ):
                valid_models.append((model_name, data["energy_consumption"]))

        if not valid_models:
            return None

        # Select model with lowest energy consumption
        return min(valid_models, key=lambda x: x[1])[0]


class BenchmarkController(BaseController):
    """Performance-at-all-costs benchmark controller."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.prefer_largest = config["benchmark_controller"]["prefer_largest_model"]
        self.charge_threshold = config["benchmark_controller"]["charge_when_below"]

    def select_model(
        self,
        battery_level: float,
        energy_cleanliness: float,
        model_data: Dict[str, Dict[str, float]],
    ) -> Optional[str]:
        """Always select largest model or charge if battery low."""
        if battery_level < self.charge_threshold:
            return None  # Force charging

        if self.prefer_largest:
            # Find the largest model (YOLOv10-X if available)
            for model_name in [
                "YOLOv10-X",
                "YOLOv10-L",
                "YOLOv10-B",
                "YOLOv10-M",
                "YOLOv10-S",
                "YOLOv10-N",
            ]:
                if model_name in model_data:
                    return model_name

        return None
