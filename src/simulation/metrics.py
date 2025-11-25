"""Performance metrics tracking for simulation."""

from typing import Dict, Any
from collections import defaultdict


class MetricsTracker:
    """Tracks simulation performance metrics."""

    def __init__(self):
        self.results = []
        self.metrics = {
            "total_inferences": 0,
            "small_misses": 0,
            "large_misses": 0,
            "charging_periods": 0,
            "total_energy_consumed": 0.0,
            "clean_energy_consumed": 0.0,
            "model_usage": defaultdict(int),
        }

    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a simulation result to track."""
        self.results.append(result)

        # Update counters
        miss_type = result["miss_type"]
        if miss_type == "small_miss":
            self.metrics["small_misses"] += 1
        elif miss_type == "large_miss":
            self.metrics["large_misses"] += 1
        elif miss_type == "charging":
            self.metrics["charging_periods"] += 1

        # Track energy consumption
        energy_consumed = result["energy_consumed"]
        clean_energy_consumed = result["clean_energy_consumed"]

        if energy_consumed > 0:
            self.metrics["total_inferences"] += 1
            self.metrics["total_energy_consumed"] += energy_consumed
            self.metrics["clean_energy_consumed"] += clean_energy_consumed

        # Track model usage
        model_selected = result["model_selected"]
        if model_selected:
            self.metrics["model_usage"][model_selected] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        total = self.metrics["total_inferences"]

        if total == 0:
            return {
                "total_inferences": 0,
                "small_misses": 0,
                "large_misses": 0,
                "small_miss_rate": 0.0,
                "large_miss_rate": 0.0,
                "charging_rate": 0.0,
                "total_energy_used": 0.0,
                "clean_energy_used": 0.0,
                "clean_energy_percentage": 0.0,
                "model_usage_distribution": {},
            }

        small_miss_rate = (self.metrics["small_misses"] / total) * 100
        large_miss_rate = (self.metrics["large_misses"] / total) * 100
        charging_rate = (self.metrics["charging_periods"] / len(self.results)) * 100

        clean_energy_percentage = 0.0
        if self.metrics["total_energy_consumed"] > 0:
            clean_energy_percentage = (
                self.metrics["clean_energy_consumed"]
                / self.metrics["total_energy_consumed"]
            ) * 100

        # Model usage distribution
        model_usage_dist = {}
        if total > 0:
            for model, count in self.metrics["model_usage"].items():
                model_usage_dist[model] = (count / total) * 100

        return {
            "total_inferences": total,
            "small_misses": self.metrics["small_misses"],
            "large_misses": self.metrics["large_misses"],
            "small_miss_rate": small_miss_rate,
            "large_miss_rate": large_miss_rate,
            "charging_rate": charging_rate,
            "total_energy_used": self.metrics["total_energy_consumed"],
            "clean_energy_used": self.metrics["clean_energy_consumed"],
            "clean_energy_percentage": clean_energy_percentage,
            "model_usage_distribution": model_usage_dist,
        }
