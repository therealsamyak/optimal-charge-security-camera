"""
Training data collection and management for ML-based controller.

This module provides infrastructure to collect training data from the
rule-based controller for future ML model training.
"""

import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from loguru import logger

from .intelligent_controller import ControllerDecision
from utils.helpers import write_csv_log


@dataclass
class TrainingExample:
    """Single training example for ML controller."""

    # Input features
    battery_level: float
    energy_cleanliness: float
    is_charging: bool
    energy_source: str
    time_of_day: float  # Hour as float (0-24)
    day_of_week: int  # Day as int (0-6)

    # Context features
    user_min_accuracy: float
    user_max_latency: float
    user_run_frequency: float

    # Target outputs (what rule-based controller decided)
    selected_model: str
    should_charge: bool
    controller_score: float

    # Performance outcomes (for reinforcement learning)
    actual_accuracy: float
    actual_latency: float
    actual_battery_consumption: float
    user_satisfaction: float  # Calculated from requirements satisfaction

    # Metadata
    timestamp: float
    reasoning: str


class TrainingDataCollector:
    """Collects training data from rule-based controller decisions."""

    def __init__(self, data_dir: str = "src/data/training"):
        """Initialize training data collector."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.training_examples: List[TrainingExample] = []
        self.current_session_start = time.time()

        # File paths
        self.csv_path = self.data_dir / "training_data.csv"
        self.json_path = self.data_dir / "training_data.json"
        self.features_path = self.data_dir / "feature_analysis.json"

    def record_decision(
        self,
        decision: ControllerDecision,
        battery_level: float,
        energy_cleanliness: float,
        is_charging: bool,
        energy_source: str,
        user_requirements: Dict[str, float],
        performance_outcomes: Dict[str, float],
        reasoning: str,
    ) -> None:
        """
        Record a controller decision as training example.

        Args:
            decision: Controller decision
            battery_level: Current battery level
            energy_cleanliness: Current energy cleanliness
            is_charging: Whether currently charging
            energy_source: Current energy source
            user_requirements: User requirements dict
            performance_outcomes: Actual performance metrics
            reasoning: Controller reasoning
        """
        # Calculate time-based features
        current_time = time.time()
        time_struct = time.localtime(current_time)
        time_of_day = time_struct.tm_hour + time_struct.tm_min / 60.0
        day_of_week = time_struct.tm_wday

        # Calculate user satisfaction (how well requirements were met)
        user_satisfaction = self._calculate_user_satisfaction(
            user_requirements, performance_outcomes
        )

        # Create training example
        example = TrainingExample(
            # Input features
            battery_level=battery_level,
            energy_cleanliness=energy_cleanliness,
            is_charging=is_charging,
            energy_source=energy_source,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            # Context features
            user_min_accuracy=user_requirements["min_accuracy"],
            user_max_latency=user_requirements["max_latency_ms"],
            user_run_frequency=user_requirements["run_frequency_ms"],
            # Target outputs
            selected_model=decision.selected_model,
            should_charge=decision.should_charge,
            controller_score=decision.score,
            # Performance outcomes
            actual_accuracy=performance_outcomes.get("accuracy", 0.0),
            actual_latency=performance_outcomes.get("latency_ms", 0.0),
            actual_battery_consumption=performance_outcomes.get(
                "battery_consumption", 0.0
            ),
            user_satisfaction=user_satisfaction,
            # Metadata
            timestamp=current_time,
            reasoning=reasoning,
        )

        self.training_examples.append(example)

        # Log every 10 examples
        if len(self.training_examples) % 10 == 0:
            logger.info(f"Collected {len(self.training_examples)} training examples")

    def _calculate_user_satisfaction(
        self,
        user_requirements: Dict[str, float],
        performance_outcomes: Dict[str, float],
    ) -> float:
        """
        Calculate user satisfaction based on requirement fulfillment.

        Args:
            user_requirements: User requirements
            performance_outcomes: Actual performance

        Returns:
            Satisfaction score (0-1, higher is better)
        """
        satisfaction = 0.0
        total_weight = 0.0

        # Accuracy satisfaction
        if "accuracy" in performance_outcomes:
            accuracy_req = user_requirements["min_accuracy"]
            actual_acc = performance_outcomes["accuracy"]
            if actual_acc >= accuracy_req:
                satisfaction += 1.0
            else:
                satisfaction += actual_acc / accuracy_req  # Partial credit
            total_weight += 1.0

        # Latency satisfaction
        if "latency_ms" in performance_outcomes:
            latency_req = user_requirements["max_latency_ms"]
            actual_latency = performance_outcomes["latency_ms"]
            if actual_latency <= latency_req:
                satisfaction += 1.0
            else:
                satisfaction += latency_req / actual_latency  # Partial credit
            total_weight += 1.0

        # Battery efficiency satisfaction
        if "battery_consumption" in performance_outcomes:
            # Lower consumption is better
            consumption = performance_outcomes["battery_consumption"]
            # Normalize to 0-1 scale (assuming 1.0 is max reasonable consumption)
            efficiency_score = max(0.0, 1.0 - consumption)
            satisfaction += efficiency_score
            total_weight += 1.0

        return satisfaction / total_weight if total_weight > 0 else 0.0

    def save_training_data(self) -> None:
        """Save collected training data to files."""
        if not self.training_examples:
            logger.warning("No training data to save")
            return

        # Save as CSV for easy analysis
        self._save_csv()

        # Save as JSON for ML training
        self._save_json()

        # Save feature analysis
        self._save_feature_analysis()

        logger.info(
            f"Saved {len(self.training_examples)} training examples to {self.data_dir}"
        )

    def _save_csv(self) -> None:
        """Save training data as CSV."""
        if not self.training_examples:
            return

        # Convert examples to dict format
        headers = [
            "battery_level",
            "energy_cleanliness",
            "is_charging",
            "energy_source",
            "time_of_day",
            "day_of_week",
            "user_min_accuracy",
            "user_max_latency",
            "user_run_frequency",
            "selected_model",
            "should_charge",
            "controller_score",
            "actual_accuracy",
            "actual_latency",
            "actual_battery_consumption",
            "user_satisfaction",
            "timestamp",
            "reasoning",
        ]

        for example in self.training_examples:
            row = asdict(example)
            write_csv_log(str(self.csv_path), row, headers)
            headers = None  # Only write headers once

    def _save_json(self) -> None:
        """Save training data as JSON for ML training."""
        data = {
            "metadata": {
                "total_examples": len(self.training_examples),
                "collection_start": self.current_session_start,
                "collection_end": time.time(),
                "version": "1.0",
            },
            "examples": [asdict(example) for example in self.training_examples],
        }

        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_feature_analysis(self) -> None:
        """Save feature analysis for ML model development."""
        if not self.training_examples:
            return

        # Analyze feature distributions
        analysis = {
            "feature_statistics": self._analyze_features(),
            "model_distribution": self._analyze_model_distribution(),
            "charging_patterns": self._analyze_charging_patterns(),
            "satisfaction_analysis": self._analyze_satisfaction(),
        }

        with open(self.features_path, "w") as f:
            json.dump(analysis, f, indent=2)

    def _analyze_features(self) -> Dict[str, Any]:
        """Analyze feature distributions."""
        features = [
            "battery_level",
            "energy_cleanliness",
            "time_of_day",
            "user_satisfaction",
        ]
        stats = {}

        for feature in features:
            values = [getattr(example, feature) for example in self.training_examples]
            if values:
                stats[feature] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "count": len(values),
                }

        return stats

    def _analyze_model_distribution(self) -> Dict[str, Any]:
        """Analyze model selection distribution."""
        model_counts = {}
        for example in self.training_examples:
            model = example.selected_model
            model_counts[model] = model_counts.get(model, 0) + 1

        total = len(self.training_examples)
        distribution = {}
        for model, count in model_counts.items():
            distribution[model] = {"count": count, "percentage": (count / total) * 100}

        return distribution

    def _analyze_charging_patterns(self) -> Dict[str, Any]:
        """Analyze charging decision patterns."""
        charging_scenarios = {
            "low_battery": {"charge": 0, "no_charge": 0},
            "clean_energy": {"charge": 0, "no_charge": 0},
            "normal": {"charge": 0, "no_charge": 0},
        }

        for example in self.training_examples:
            # Categorize scenario
            if example.battery_level < 30:
                scenario = "low_battery"
            elif example.energy_cleanliness > 80:
                scenario = "clean_energy"
            else:
                scenario = "normal"

            # Count decision
            if example.should_charge:
                charging_scenarios[scenario]["charge"] += 1
            else:
                charging_scenarios[scenario]["no_charge"] += 1

        return charging_scenarios

    def _analyze_satisfaction(self) -> Dict[str, Any]:
        """Analyze user satisfaction patterns."""
        satisfaction_levels = [
            example.user_satisfaction for example in self.training_examples
        ]

        if not satisfaction_levels:
            return {}

        return {
            "mean_satisfaction": sum(satisfaction_levels) / len(satisfaction_levels),
            "min_satisfaction": min(satisfaction_levels),
            "max_satisfaction": max(satisfaction_levels),
            "satisfaction_distribution": {
                "high (>0.8)": sum(1 for s in satisfaction_levels if s > 0.8),
                "medium (0.5-0.8)": sum(
                    1 for s in satisfaction_levels if 0.5 <= s <= 0.8
                ),
                "low (<0.5)": sum(1 for s in satisfaction_levels if s < 0.5),
            },
        }

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of collected training data."""
        if not self.training_examples:
            return {"total_examples": 0}

        return {
            "total_examples": len(self.training_examples),
            "collection_duration_hours": (time.time() - self.current_session_start)
            / 3600,
            "examples_per_hour": len(self.training_examples)
            / ((time.time() - self.current_session_start) / 3600),
            "unique_models": len(
                set(ex.selected_model for ex in self.training_examples)
            ),
            "charging_decisions": sum(
                1 for ex in self.training_examples if ex.should_charge
            ),
            "average_satisfaction": sum(
                ex.user_satisfaction for ex in self.training_examples
            )
            / len(self.training_examples),
        }

    def clear_data(self) -> None:
        """Clear collected training data."""
        self.training_examples.clear()
        self.current_session_start = time.time()
        logger.info("Training data cleared")
