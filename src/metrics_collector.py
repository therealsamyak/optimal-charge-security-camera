import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetricsCollector:
    """Collects and tracks simulation metrics in real-time."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "missed_deadlines": 0,
            "small_model_tasks": 0,
            "large_model_tasks": 0,
            "small_model_misses": 0,
            "large_model_misses": 0,
            "total_energy_wh": 0.0,
            "clean_energy_wh": 0.0,
            "battery_levels": [],
            "model_selections": {},
            "task_completion_rate": 0.0,
            "small_model_miss_rate": 0.0,
            "large_model_miss_rate": 0.0,
            "clean_energy_percentage": 0.0,
        }
        self.start_time = None
        self.end_time = None

    def start_simulation(self):
        """Mark the start of simulation."""
        self.start_time = datetime.now()
        self.logger.info("Metrics collection started")

    def end_simulation(self):
        """Mark the end of simulation."""
        self.end_time = datetime.now()
        self.logger.info("Metrics collection ended")

    def update_task_metrics(self, task_data: Dict[str, Any]):
        """Update metrics with task completion data."""
        self.metrics["total_tasks"] += 1

        if task_data.get("completed", False):
            self.metrics["completed_tasks"] += 1
            self.metrics["total_energy_wh"] += task_data.get("energy_used_wh", 0.0)
            self.metrics["clean_energy_wh"] += task_data.get(
                "clean_energy_used_wh", 0.0
            )

            # Track model usage
            model_name = task_data.get("model_used", "unknown")
            if model_name not in self.metrics["model_selections"]:
                self.metrics["model_selections"][model_name] = 0
            self.metrics["model_selections"][model_name] += 1

            # Categorize model size
            if model_name in ["YOLOv10_N", "YOLOv10_S"]:
                self.metrics["small_model_tasks"] += 1
            else:
                self.metrics["large_model_tasks"] += 1
        else:
            self.metrics["missed_deadlines"] += 1

            # Track misses by model size
            model_name = task_data.get("model_used", "unknown")
            if model_name in ["YOLOv10_N", "YOLOv10_S"]:
                self.metrics["small_model_misses"] += 1
            else:
                self.metrics["large_model_misses"] += 1

    def update_battery_level(self, timestamp: float, level: float):
        """Update battery level tracking."""
        self.metrics["battery_levels"].append({"timestamp": timestamp, "level": level})

    def calculate_final_metrics(self):
        """Calculate final derived metrics."""
        # Task completion rate
        if self.metrics["total_tasks"] > 0:
            self.metrics["task_completion_rate"] = (
                self.metrics["completed_tasks"] / self.metrics["total_tasks"] * 100
            )

        # Small model miss rate
        if self.metrics["small_model_tasks"] > 0:
            self.metrics["small_model_miss_rate"] = (
                self.metrics["small_model_misses"]
                / self.metrics["small_model_tasks"]
                * 100
            )

        # Large model miss rate
        if self.metrics["large_model_tasks"] > 0:
            self.metrics["large_model_miss_rate"] = (
                self.metrics["large_model_misses"]
                / self.metrics["large_model_tasks"]
                * 100
            )

        # Clean energy percentage
        if self.metrics["total_energy_wh"] > 0:
            self.metrics["clean_energy_percentage"] = (
                self.metrics["clean_energy_wh"] / self.metrics["total_energy_wh"] * 100
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset all metrics to initial state."""
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "missed_deadlines": 0,
            "small_model_tasks": 0,
            "large_model_tasks": 0,
            "small_model_misses": 0,
            "large_model_misses": 0,
            "total_energy_wh": 0.0,
            "clean_energy_wh": 0.0,
            "battery_levels": [],
            "model_selections": {},
            "task_completion_rate": 0.0,
            "small_model_miss_rate": 0.0,
            "large_model_miss_rate": 0.0,
            "clean_energy_percentage": 0.0,
        }
        self.start_time = None
        self.end_time = None


class CSVExporter:
    """Exports simulation results to CSV format."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def export_summary(
        self, metrics: Dict[str, Any], simulation_info: Dict[str, Any]
    ) -> str:
        """Export simulation summary to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_summary_{timestamp}.csv"
        filepath = self.output_dir / filename

        # Prepare summary data
        summary_data = {
            "simulation_id": simulation_info.get("id", "unknown"),
            "location": simulation_info.get("location", "unknown"),
            "season": simulation_info.get("season", "unknown"),
            "week": simulation_info.get("week", 0),
            "controller": simulation_info.get("controller", "unknown"),
            "total_tasks": metrics.get("total_tasks", 0),
            "completed_tasks": metrics.get("completed_tasks", 0),
            "missed_deadlines": metrics.get("missed_deadlines", 0),
            "task_completion_rate": metrics.get("task_completion_rate", 0.0),
            "small_model_tasks": metrics.get("small_model_tasks", 0),
            "large_model_tasks": metrics.get("large_model_tasks", 0),
            "small_model_miss_rate": metrics.get("small_model_miss_rate", 0.0),
            "large_model_miss_rate": metrics.get("large_model_miss_rate", 0.0),
            "total_energy_wh": metrics.get("total_energy_wh", 0.0),
            "clean_energy_wh": metrics.get("clean_energy_wh", 0.0),
            "clean_energy_percentage": metrics.get("clean_energy_percentage", 0.0),
            "final_battery_level": metrics.get("final_battery_level", 0.0),
            "avg_battery_level": metrics.get("avg_battery_level", 0.0),
        }

        # Add model selection counts
        model_selections = metrics.get("model_selections", {})
        for model in [
            "YOLOv10_N",
            "YOLOv10_S",
            "YOLOv10_M",
            "YOLOv10_B",
            "YOLOv10_L",
            "YOLOv10_X",
        ]:
            summary_data[f"{model}_count"] = model_selections.get(model, 0)

        try:
            with open(filepath, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=summary_data.keys())
                writer.writeheader()
                writer.writerow(summary_data)

            self.logger.info(f"Summary exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to export summary: {e}")
            return ""

    def export_detailed_timeseries(
        self, metrics: Dict[str, Any], simulation_info: Dict[str, Any]
    ) -> str:
        """Export detailed time-series data to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"timeseries_{simulation_info.get('id', 'unknown')}_{timestamp}.csv"
        filepath = self.output_dir / filename

        battery_levels = metrics.get("battery_levels", [])

        if not battery_levels:
            self.logger.warning("No battery level data to export")
            return ""

        try:
            with open(filepath, "w", newline="") as csvfile:
                fieldnames = [
                    "timestamp",
                    "battery_level",
                    "simulation_id",
                    "location",
                    "season",
                    "controller",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for entry in battery_levels:
                    writer.writerow(
                        {
                            "timestamp": entry["timestamp"],
                            "battery_level": entry["level"],
                            "simulation_id": simulation_info.get("id", "unknown"),
                            "location": simulation_info.get("location", "unknown"),
                            "season": simulation_info.get("season", "unknown"),
                            "controller": simulation_info.get("controller", "unknown"),
                        }
                    )

            self.logger.info(f"Time-series data exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to export time-series data: {e}")
            return ""

    def export_aggregated_results(
        self, all_simulations: List[Dict[str, Any]], filename: Optional[str] = None
    ) -> str:
        """Export aggregated results across multiple simulations."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aggregated_results_{timestamp}.csv"
        filepath = self.output_dir / filename

        if not all_simulations:
            self.logger.warning("No simulation data to aggregate")
            return ""

        try:
            # Collect all unique fieldnames
            all_fieldnames = set()
            for sim in all_simulations:
                all_fieldnames.update(sim.keys())

            all_fieldnames = sorted(list(all_fieldnames))

            with open(filepath, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=all_fieldnames)
                writer.writeheader()
                writer.writerows(all_simulations)

            self.logger.info(f"Aggregated results exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to export aggregated results: {e}")
            return ""

    def export_detailed_results(
        self, all_simulations: List[Dict[str, Any]], filename: Optional[str] = None
    ) -> str:
        """Export detailed results with all parameters for batch simulations."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_results_{timestamp}.csv"
        filepath = self.output_dir / filename

        if not all_simulations:
            self.logger.warning("No simulation data to export")
            return ""

        try:
            # Define detailed fieldnames including parameter variations
            detailed_fieldnames = [
                "simulation_id",
                "variation_id",
                "location",
                "season",
                "week",
                "controller",
                "accuracy_requirement",
                "latency_requirement",
                "battery_capacity_wh",
                "charge_rate_watts",
                "total_tasks",
                "completed_tasks",
                "missed_deadlines",
                "task_completion_rate",
                "small_model_tasks",
                "large_model_tasks",
                "small_model_miss_rate",
                "large_model_miss_rate",
                "total_energy_wh",
                "clean_energy_wh",
                "clean_energy_percentage",
                "final_battery_level",
                "avg_battery_level",
                "timestamp",
                "success",
            ]

            # Add model selection counts
            model_fields = [
                f"{model}_count"
                for model in [
                    "YOLOv10_N",
                    "YOLOv10_S",
                    "YOLOv10_M",
                    "YOLOv10_B",
                    "YOLOv10_L",
                    "YOLOv10_X",
                ]
            ]
            detailed_fieldnames.extend(model_fields)

            # Collect any additional fields from simulations
            all_fieldnames = set(detailed_fieldnames)
            for sim in all_simulations:
                all_fieldnames.update(sim.keys())

            all_fieldnames = sorted(list(all_fieldnames))

            with open(filepath, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=all_fieldnames)
                writer.writeheader()
                writer.writerows(all_simulations)

            self.logger.info(f"Detailed results exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to export detailed results: {e}")
            return ""

    def export_json(self, data: Dict[str, Any], filename: str) -> str:
        """Export data to JSON format."""
        filepath = self.output_dir / filename

        try:
            with open(filepath, "w") as jsonfile:
                json.dump(data, jsonfile, indent=2, default=str)

            self.logger.info(f"JSON data exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")
            return ""
