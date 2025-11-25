"""Main simulation runner."""

import csv
from datetime import datetime, timedelta
from typing import Dict, Any, List
from loguru import logger

from src.data.energy_loader import EnergyLoader
from src.data.model_data import ModelDataLoader
from src.sensors.simulation_sensors import SimulationSensors
from src.simulation.controllers import (
    CustomController,
    OracleController,
    BenchmarkController,
)
from src.simulation.metrics import MetricsTracker


class SimulationRunner:
    """Main simulation orchestrator."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sim_config = config["simulation"]

        # Initialize components
        self.energy_loader = EnergyLoader()
        self.model_loader = ModelDataLoader(config=config)
        self.sensors = SimulationSensors(config)
        self.metrics = MetricsTracker()

        # Initialize controller
        controller_type = self.sim_config["controller_type"]
        if controller_type == "custom":
            self.controller = CustomController(config)
        elif controller_type == "oracle":
            self.controller = OracleController(config)
        elif controller_type == "benchmark":
            self.controller = BenchmarkController(config)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

        # Simulation parameters
        self.sim_date = datetime.strptime(self.sim_config["date"], "%Y-%m-%d")
        self.output_interval = self.sim_config["output_interval_seconds"]
        self.image_quality = self.sim_config["image_quality"]

        # Get model data
        self.model_data = self.model_loader.get_model_data()

    def run_simulation(self) -> List[Dict[str, Any]]:
        """Run 24-hour simulation."""
        logger.info(
            f"Starting simulation for {self.sim_config['date']} with {self.sim_config['controller_type']} controller"
        )

        results = []
        current_time = self.sim_date
        end_time = current_time + timedelta(hours=24)

        while current_time < end_time:
            # Update energy cleanliness based on time
            energy_cleanliness = self.energy_loader.get_clean_energy_percentage(
                current_time
            )
            self.sensors.update_energy_cleanliness(energy_cleanliness)

            # Get current battery level
            battery_level = self.sensors.get_battery_level()

            # Check if charging is forced
            if self.sensors.is_charging():
                model_selected = None
                accuracy = 0.0
                latency = 0.0
                miss_type = "charging"
                energy_consumed = 0.0
                clean_energy_consumed = 0.0

                # Charge battery
                self.sensors.charge_battery(self.output_interval)
            else:
                # Select model
                model_selected = self.controller.select_model(
                    battery_level, energy_cleanliness, self.model_data
                )

                if model_selected is None:
                    # No suitable model found
                    accuracy = 0.0
                    latency = 0.0
                    miss_type = "no_model"
                    energy_consumed = 0.0
                    clean_energy_consumed = 0.0
                else:
                    # Get model performance
                    model_info = self.model_data[model_selected]
                    accuracy = model_info["accuracy"]
                    latency = model_info["latency_ms"]
                    energy_consumed = model_info["energy_consumption"]

                    # Check if model meets thresholds
                    if (
                        accuracy < self.config["accuracy_threshold"]
                        or latency > self.config["latency_threshold_ms"]
                    ):
                        miss_type = "small_miss"
                    else:
                        miss_type = "none"

                    # Consume energy
                    if self.sensors.consume_energy(energy_consumed):
                        clean_energy_consumed = energy_consumed * energy_cleanliness
                    else:
                        # Battery dead
                        miss_type = "large_miss"
                        clean_energy_consumed = 0.0

            # Record results
            result = {
                "timestamp": current_time.isoformat(),
                "battery_level": battery_level,
                "energy_cleanliness": energy_cleanliness,
                "model_selected": model_selected or "",
                "accuracy": accuracy,
                "latency": latency,
                "miss_type": miss_type,
                "energy_consumed": energy_consumed,
                "clean_energy_consumed": clean_energy_consumed,
            }

            results.append(result)
            self.metrics.add_result(result)

            # Advance time
            current_time += timedelta(seconds=self.output_interval)

        logger.info(f"Simulation completed. Total results: {len(results)}")
        return results

    def save_results(
        self, results: List[Dict[str, Any]], filename: str = "simulation_results.csv"
    ) -> None:
        """Save simulation results to CSV."""
        if not results:
            logger.warning("No results to save")
            return

        fieldnames = [
            "timestamp",
            "battery_level",
            "energy_cleanliness",
            "model_selected",
            "accuracy",
            "latency",
            "miss_type",
            "energy_consumed",
            "clean_energy_consumed",
        ]

        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"Results saved to {filename}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of simulation metrics."""
        return self.metrics.get_summary()
