import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class CSVExporter:
    """Export simulation results to CSV format."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def export_aggregated_results(
        self, aggregated_data: List[Dict[str, Any]], filename: Optional[str] = None
    ) -> str:
        """Export aggregated results to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aggregated_results_{timestamp}.csv"
        filepath = self.output_dir / filename

        if not aggregated_data:
            self.logger.warning("No aggregated data to export")
            return ""

        try:
            fieldnames = [
                "controller",
                "location",
                "season",
                "week",
                "total_simulations",
                "successful_simulations",
                "success_rate",
                "avg_task_completion_rate",
                "avg_clean_energy_percentage",
                "avg_battery_efficiency",
                "total_energy_consumed_wh",
                "total_clean_energy_wh",
                "total_tasks_completed",
                "total_missed_deadlines",
                "timestamp",
            ]

            with open(filepath, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(aggregated_data)

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
                # Energy metrics
                "total_energy_wh",
                "clean_energy_wh",
                "clean_energy_mwh",
                "dirty_energy_mwh",
                "clean_energy_percentage",
                "energy_per_task_wh",
                "clean_energy_per_task_wh",
                "peak_power_mw",
                "average_power_mw",
                # Battery metrics
                "final_battery_level",
                "avg_battery_level",
                "battery_depletion_rate_per_hour",
                "battery_efficiency_score",
                "time_below_20_percent",
                "time_above_80_percent",
                "charging_events_count",
                "total_simulation_time",
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

            # Add model energy breakdown fields
            model_energy_fields = [
                f"{model}_energy_wh"
                for model in [
                    "YOLOv10_N",
                    "YOLOv10_S",
                    "YOLOv10_M",
                    "YOLOv10_B",
                    "YOLOv10_L",
                    "YOLOv10_X",
                ]
            ]
            detailed_fieldnames.extend(model_energy_fields)

            # Collect any additional fields from simulations
            all_fieldnames = set(detailed_fieldnames)
            for sim in all_simulations:
                all_fieldnames.update(sim.keys())

            # Remove fields that contain complex data (JSON arrays, etc.)
            exclude_fields = {"battery_levels", "model_selections"}
            all_fieldnames = [f for f in all_fieldnames if f not in exclude_fields]
            all_fieldnames = sorted(list(all_fieldnames))

            # Validate export data before exporting
            self._validate_export_data(all_simulations)

            # Extract model selection counts and energy breakdown into individual fields before filtering
            for sim in all_simulations:
                model_selections = sim.get("model_selections", {})
                model_energy_breakdown = sim.get("model_energy_breakdown", {})

                for model in [
                    "YOLOv10_N",
                    "YOLOv10_S",
                    "YOLOv10_M",
                    "YOLOv10_B",
                    "YOLOv10_L",
                    "YOLOv10_X",
                ]:
                    sim[f"{model}_count"] = model_selections.get(model, 0)
                    # Convert MWh to Wh for consistency
                    sim[f"{model}_energy_wh"] = (
                        model_energy_breakdown.get(model, 0.0) * 1000
                    )

            # Filter simulation data to exclude problematic fields
            exclude_fields = {"battery_levels", "model_selections"}
            filtered_simulations = []
            for sim in all_simulations:
                filtered_sim = {k: v for k, v in sim.items() if k not in exclude_fields}
                filtered_simulations.append(filtered_sim)

            with open(filepath, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=all_fieldnames)
                writer.writeheader()
                writer.writerows(filtered_simulations)

            # Export time series data to JSON files
            self._export_time_series_data(all_simulations)

            self.logger.info(f"Detailed results exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to export detailed results: {e}")
            return ""

    def _validate_export_data(self, all_simulations: List[Dict[str, Any]]):
        """Validate data before export."""
        for i, sim in enumerate(all_simulations):
            # Check for missing critical fields
            required_fields = ["total_tasks", "completed_tasks", "total_energy_wh"]
            for field in required_fields:
                if field not in sim or sim[field] is None:
                    self.logger.error(f"Simulation {i} missing required field: {field}")

            # Check for logical inconsistencies
            if sim.get("completed_tasks", 0) > sim.get("total_tasks", 0):
                self.logger.error(f"Simulation {i}: completed_tasks > total_tasks")

            # Check energy balance
            total_energy = sim.get("total_energy_wh", 0)
            clean_energy = sim.get("clean_energy_wh", 0)
            dirty_energy = sim.get("dirty_energy_mwh", 0) * 1000  # Convert to Wh

            if abs(total_energy - (clean_energy + dirty_energy)) > 0.001:
                self.logger.warning(f"Simulation {i}: Energy balance mismatch")

            # Check for negative values where they shouldn't exist
            negative_fields = [
                "total_energy_wh",
                "clean_energy_wh",
                "completed_tasks",
                "total_tasks",
            ]
            for field in negative_fields:
                if sim.get(field, 0) < 0:
                    self.logger.error(
                        f"Simulation {i}: Negative value for {field}: {sim[field]}"
                    )

    def _export_time_series_data(self, all_simulations: List[Dict[str, Any]]):
        """Export detailed time series data to separate JSON files."""
        try:
            # Battery time series data
            battery_data = {}
            for sim in all_simulations:
                sim_id = sim.get("simulation_id", f"sim_{len(battery_data)}")
                if "battery_levels" in sim and sim["battery_levels"]:
                    battery_data[sim_id] = {
                        "location": sim.get("location"),
                        "season": sim.get("season"),
                        "week": sim.get("week"),
                        "controller": sim.get("controller"),
                        "battery_levels": sim["battery_levels"],
                    }

            if battery_data:
                battery_filepath = self.output_dir / "battery_time_series.json"
                with open(battery_filepath, "w") as f:
                    json.dump(battery_data, f, indent=2, default=str)
                self.logger.info(f"Battery time series exported to {battery_filepath}")

            # Model selection timeline data
            model_timeline = {}
            for sim in all_simulations:
                sim_id = sim.get("simulation_id", f"sim_{len(model_timeline)}")
                model_timeline[sim_id] = {
                    "location": sim.get("location"),
                    "season": sim.get("season"),
                    "week": sim.get("week"),
                    "controller": sim.get("controller"),
                    "model_selections": sim.get("model_selections", {}),
                    "model_energy_breakdown": sim.get("model_energy_breakdown", {}),
                }

            if model_timeline:
                model_filepath = self.output_dir / "model_selection_timeline.json"
                with open(model_filepath, "w") as f:
                    json.dump(model_timeline, f, indent=2, default=str)
                self.logger.info(
                    f"Model selection timeline exported to {model_filepath}"
                )

            # Energy usage summary data
            energy_summary = {}
            for sim in all_simulations:
                sim_id = sim.get("simulation_id", f"sim_{len(energy_summary)}")
                energy_summary[sim_id] = {
                    "location": sim.get("location"),
                    "season": sim.get("season"),
                    "week": sim.get("week"),
                    "controller": sim.get("controller"),
                    "total_energy_wh": sim.get("total_energy_wh", 0),
                    "clean_energy_wh": sim.get("clean_energy_wh", 0),
                    "clean_energy_percentage": sim.get("clean_energy_percentage", 0),
                    "dirty_energy_mwh": sim.get("dirty_energy_mwh", 0),
                    "peak_power_mw": sim.get("peak_power_mw", 0),
                    "average_power_mw": sim.get("average_power_mw", 0),
                    "energy_per_task_wh": sim.get("energy_per_task_wh", 0),
                    "model_energy_breakdown": sim.get("model_energy_breakdown", {}),
                }

            if energy_summary:
                energy_filepath = self.output_dir / "energy_usage_summary.json"
                with open(energy_filepath, "w") as f:
                    json.dump(energy_summary, f, indent=2, default=str)
                self.logger.info(f"Energy usage summary exported to {energy_filepath}")

        except Exception as e:
            self.logger.error(f"Failed to export time series data: {e}")

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
