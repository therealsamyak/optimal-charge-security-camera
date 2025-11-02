"""
Performance validation and constraint checking utilities.

This module provides validation functions to ensure model performance
meets user-defined requirements and constraints.
"""

from typing import Dict, List
from dataclasses import dataclass
from config import config


@dataclass
class ValidationResult:
    """Result of performance validation."""

    is_valid: bool
    violations: List[str]
    warnings: List[str]
    score: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for validation."""

    accuracy: float
    latency_ms: float
    battery_consumption: float
    confidence: float
    has_detection: bool


class PerformanceValidator:
    """Validates model performance against user constraints."""

    def __init__(self):
        """Initialize validator with user requirements."""
        self.requirements = self._load_requirements()
        self.tolerance_factor = 1.1  # 10% tolerance for warnings

    def _load_requirements(self) -> Dict[str, float]:
        """Load user requirements from configuration."""
        return {
            "min_accuracy": config.get("requirements.min_accuracy", 80.0),
            "max_latency_ms": config.get("requirements.max_latency_ms", 100.0),
            "run_frequency_ms": config.get("requirements.run_frequency_ms", 2000.0),
        }

    def validate_performance(
        self, metrics: PerformanceMetrics, model_name: str
    ) -> ValidationResult:
        """
        Validate performance metrics against user requirements.

        Args:
            metrics: Performance metrics to validate
            model_name: Name of the model being validated

        Returns:
            ValidationResult with validation status and details
        """
        violations = []
        warnings = []
        score = 100.0

        # Validate accuracy requirement
        if metrics.accuracy < self.requirements["min_accuracy"]:
            violations.append(
                f"Accuracy {metrics.accuracy:.1f}% below minimum {self.requirements['min_accuracy']:.1f}%"
            )
            score -= (self.requirements["min_accuracy"] - metrics.accuracy) * 2
        elif (
            metrics.accuracy < self.requirements["min_accuracy"] * self.tolerance_factor
        ):
            warnings.append(
                f"Accuracy {metrics.accuracy:.1f}% close to minimum {self.requirements['min_accuracy']:.1f}%"
            )
            score -= 10

        # Validate latency requirement
        if metrics.latency_ms > self.requirements["max_latency_ms"]:
            violations.append(
                f"Latency {metrics.latency_ms:.1f}ms exceeds maximum {self.requirements['max_latency_ms']:.1f}ms"
            )
            score -= (metrics.latency_ms - self.requirements["max_latency_ms"]) * 0.5
        elif (
            metrics.latency_ms
            > self.requirements["max_latency_ms"] * self.tolerance_factor
        ):
            warnings.append(
                f"Latency {metrics.latency_ms:.1f}ms close to maximum {self.requirements['max_latency_ms']:.1f}ms"
            )
            score -= 10

        # Validate battery consumption (soft constraint)
        if metrics.battery_consumption > 0.8:
            warnings.append(
                f"High battery consumption {metrics.battery_consumption:.2f} for model {model_name}"
            )
            score -= 15

        # Validate detection confidence
        if not metrics.has_detection and metrics.confidence < 0.5:
            warnings.append(
                f"Low detection confidence {metrics.confidence:.3f} - possible poor model performance"
            )
            score -= 5

        # Ensure score doesn't go negative
        score = max(0.0, score)

        is_valid = len(violations) == 0

        return ValidationResult(
            is_valid=is_valid, violations=violations, warnings=warnings, score=score
        )

    def validate_model_selection(
        self, model_name: str, battery_level: float, energy_cleanliness: float
    ) -> ValidationResult:
        """
        Validate if model selection is appropriate for current conditions.

        Args:
            model_name: Name of selected model
            battery_level: Current battery percentage
            energy_cleanliness: Current energy cleanliness percentage

        Returns:
            ValidationResult with validation status and details
        """
        violations = []
        warnings = []
        score = 100.0

        # Model-specific battery considerations
        high_power_models = ["yolov10l", "yolov10x"]
        medium_power_models = ["yolov10m", "yolov10b"]

        if model_name in high_power_models and battery_level < 30.0:
            violations.append(
                f"High-power model {model_name} selected with low battery {battery_level:.1f}%"
            )
            score -= 30
        elif model_name in high_power_models and battery_level < 50.0:
            warnings.append(
                f"High-power model {model_name} selected with moderate battery {battery_level:.1f}%"
            )
            score -= 15

        if model_name in medium_power_models and battery_level < 20.0:
            violations.append(
                f"Medium-power model {model_name} selected with very low battery {battery_level:.1f}%"
            )
            score -= 20

        # Energy cleanliness considerations
        if energy_cleanliness < 30.0 and battery_level < 40.0:
            warnings.append(
                f"Running on dirty energy ({energy_cleanliness:.1f}%) with low battery ({battery_level:.1f}%)"
            )
            score -= 10

        # Model efficiency scoring
        if (
            model_name in high_power_models
            and energy_cleanliness > 80.0
            and battery_level > 70.0
        ):
            score += 10  # Bonus for using powerful models when conditions are good

        is_valid = len(violations) == 0

        return ValidationResult(
            is_valid=is_valid, violations=violations, warnings=warnings, score=score
        )

    def check_run_frequency_feasibility(
        self, model_latency_ms: float
    ) -> ValidationResult:
        """
        Check if run frequency is feasible given model latency.

        Args:
            model_latency_ms: Model inference latency in milliseconds

        Returns:
            ValidationResult with feasibility check
        """
        violations = []
        warnings = []
        score = 100.0

        required_frequency = self.requirements["run_frequency_ms"]

        if model_latency_ms >= required_frequency:
            violations.append(
                f"Model latency {model_latency_ms:.1f}ms exceeds run frequency {required_frequency:.1f}ms"
            )
            score -= 50
        elif model_latency_ms >= required_frequency * 0.8:
            warnings.append(
                f"Model latency {model_latency_ms:.1f}ms close to run frequency {required_frequency:.1f}ms"
            )
            score -= 20

        # Calculate overhead percentage
        overhead_percentage = (model_latency_ms / required_frequency) * 100
        if overhead_percentage > 50:
            warnings.append(f"Model uses {overhead_percentage:.1f}% of run interval")
            score -= 10

        is_valid = len(violations) == 0

        return ValidationResult(
            is_valid=is_valid, violations=violations, warnings=warnings, score=score
        )

    def get_recommendations(
        self, validation_result: ValidationResult, current_model: str
    ) -> List[str]:
        """
        Get recommendations based on validation result.

        Args:
            validation_result: Result from validation
            current_model: Currently selected model

        Returns:
            List of recommendations
        """
        recommendations = []

        if not validation_result.is_valid:
            # Handle violations
            for violation in validation_result.violations:
                if "Accuracy" in violation:
                    recommendations.append(
                        "Consider using a more accurate model (yolov10m, yolov10b, yolov10l, or yolov10x)"
                    )
                elif "Latency" in violation:
                    recommendations.append(
                        "Consider using a faster model (yolov10n or yolov10s)"
                    )
                elif "High-power model" in violation and "low battery" in violation:
                    recommendations.append(
                        "Switch to a more efficient model (yolov10n, yolov10s, or yolov10m)"
                    )
                elif "exceeds run frequency" in violation:
                    recommendations.append("Reduce run frequency or use a faster model")

        # Handle warnings
        for warning in validation_result.warnings:
            if "close to minimum" in warning and "Accuracy" in warning:
                recommendations.append(
                    "Monitor accuracy closely, consider model upgrade if performance degrades"
                )
            elif "close to maximum" in warning and "Latency" in warning:
                recommendations.append(
                    "Monitor latency closely, consider model downgrade if performance degrades"
                )
            elif "High battery consumption" in warning:
                recommendations.append("Monitor battery level closely")
            elif "Low detection confidence" in warning:
                recommendations.append("Check input quality and model suitability")

        # General recommendations based on score
        if validation_result.score < 70:
            recommendations.append(
                "Overall performance is poor - review all constraints and model selection"
            )
        elif validation_result.score < 85:
            recommendations.append("Performance is acceptable but could be optimized")

        return list(set(recommendations))  # Remove duplicates
