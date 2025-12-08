import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class JSONExporter:
    """Export simulation results to hierarchical JSON format."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def export_results(
        self,
        all_simulations: List[Dict[str, Any]],
        aggregated_data: List[Dict[str, Any]],
        filename: Optional[str] = None,
    ) -> str:
        """Export all results to hierarchical JSON structure."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp}.json"
        filepath = self.output_dir / filename

        if not all_simulations and not aggregated_data:
            self.logger.warning("No data to export")
            return ""

        try:
            # Build hierarchical JSON structure
            json_data = {
                "metadata": self._build_metadata(),
                "aggregated_metrics": aggregated_data,
                "detailed_metrics": self._process_detailed_metrics(all_simulations),
                "time_series": self._extract_time_series_data(all_simulations),
            }

            # Validate JSON structure
            self._validate_json_structure(json_data)

            with open(filepath, "w") as f:
                json.dump(json_data, f, indent=2, default=str)

            self.logger.info(f"Results exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            return ""

    def _build_metadata(self) -> Dict[str, Any]:
        """Build metadata section."""
        return {
            "export_timestamp": datetime.now().isoformat(),
            "export_version": "1.0",
            "schema": {
                "aggregated_metrics": "Summary statistics per controller/location/season",
                "detailed_metrics": "Individual simulation results with all parameters",
                "time_series": "Battery levels and model selections over time",
            },
        }

    def _process_detailed_metrics(
        self, all_simulations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process and clean detailed metrics."""
        if not all_simulations:
            return []

        processed_simulations = []
        for sim in all_simulations:
            # Create a clean copy without time series data
            clean_sim = {
                k: v
                for k, v in sim.items()
                if k not in {"battery_levels", "model_selections"}
            }

            # Add model selection counts as individual fields
            model_selections = sim.get("model_selections", {})
            for model in [
                "YOLOv10_N",
                "YOLOv10_S",
                "YOLOv10_M",
                "YOLOv10_B",
                "YOLOv10_L",
                "YOLOv10_X",
            ]:
                clean_sim[f"{model}_count"] = model_selections.get(model, 0)

            # Convert model energy breakdown from MWh to Wh for consistency
            model_energy_breakdown = sim.get("model_energy_breakdown", {})
            for model in [
                "YOLOv10_N",
                "YOLOv10_S",
                "YOLOv10_M",
                "YOLOv10_B",
                "YOLOv10_L",
                "YOLOv10_X",
            ]:
                clean_sim[f"{model}_energy_wh"] = (
                    model_energy_breakdown.get(model, 0.0) * 1000
                )

            processed_simulations.append(clean_sim)

        return processed_simulations

    def _extract_time_series_data(
        self, all_simulations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract time series data from simulations."""
        time_series = {
            "battery_levels": {},
            "model_selections": {},
            "energy_breakdown": {},
        }

        for sim in all_simulations:
            sim_id = sim.get(
                "simulation_id", f"sim_{len(time_series['battery_levels'])}"
            )

            # Battery levels over time
            if "battery_levels" in sim and sim["battery_levels"]:
                time_series["battery_levels"][sim_id] = {
                    "location": sim.get("location"),
                    "season": sim.get("season"),
                    "week": sim.get("week"),
                    "controller": sim.get("controller"),
                    "levels": sim["battery_levels"],
                }

            # Model selections over time
            if "model_selections" in sim and sim["model_selections"]:
                time_series["model_selections"][sim_id] = {
                    "location": sim.get("location"),
                    "season": sim.get("season"),
                    "week": sim.get("week"),
                    "controller": sim.get("controller"),
                    "selections": sim["model_selections"],
                }

            # Energy breakdown over time
            if "model_energy_breakdown" in sim and sim["model_energy_breakdown"]:
                time_series["energy_breakdown"][sim_id] = {
                    "location": sim.get("location"),
                    "season": sim.get("season"),
                    "week": sim.get("week"),
                    "controller": sim.get("controller"),
                    "breakdown": sim["model_energy_breakdown"],
                }

        return time_series

    def _validate_json_structure(self, json_data: Dict[str, Any]):
        """Validate the JSON structure before export."""
        required_sections = [
            "metadata",
            "aggregated_metrics",
            "detailed_metrics",
            "time_series",
        ]
        for section in required_sections:
            if section not in json_data:
                raise ValueError(f"Missing required section: {section}")

        # Validate metadata
        metadata = json_data["metadata"]
        required_metadata = ["export_timestamp", "export_version", "schema"]
        for field in required_metadata:
            if field not in metadata:
                raise ValueError(f"Missing metadata field: {field}")

        # Validate detailed metrics
        if json_data["detailed_metrics"]:
            required_sim_fields = ["simulation_id", "controller", "location"]
            for i, sim in enumerate(json_data["detailed_metrics"]):
                for field in required_sim_fields:
                    if field not in sim:
                        self.logger.warning(f"Simulation {i} missing field: {field}")

    def export_json(self, data: Dict[str, Any], filename: str) -> str:
        """Export arbitrary data to JSON format."""
        filepath = self.output_dir / filename

        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            self.logger.info(f"JSON data exported to {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")
            return ""
