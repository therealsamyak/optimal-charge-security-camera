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

        logger.info("=" * 60)
        logger.info("Initializing Simulation Runner")
        logger.info("=" * 60)
        logger.info("Configuration:")
        logger.info(f"  Date: {self.sim_config['date']}")
        logger.info(f"  Controller: {self.sim_config['controller_type']}")
        logger.info(f"  Image Quality: {self.sim_config['image_quality']}")
        logger.info(f"  Output Interval: {self.sim_config['output_interval_seconds']}s")
        logger.info(f"  Accuracy Threshold: {config['accuracy_threshold']}")
        logger.info(f"  Latency Threshold: {config['latency_threshold_ms']}ms")
        logger.info(f"  Initial Battery: {config['battery']['initial_capacity']}%")
        logger.info(f"  Max Battery: {config['battery']['max_capacity']}%")

        # Initialize components
        logger.info("Loading energy data...")
        self.energy_loader = EnergyLoader()
        logger.info("Loading model data...")
        self.model_loader = ModelDataLoader(config=config)
        self.sensors = SimulationSensors(config)
        self.metrics = MetricsTracker()

        # Simulation parameters
        self.sim_date = datetime.strptime(self.sim_config["date"], "%Y-%m-%d")
        self.output_interval = self.sim_config["output_interval_seconds"]
        self.image_quality = self.sim_config["image_quality"]

        # Get model data
        self.model_data = self.model_loader.get_model_data()
        logger.info(
            f"Loaded {len(self.model_data)} models: {list(self.model_data.keys())}"
        )

        # Initialize controller
        controller_type = self.sim_config["controller_type"]
        logger.info(f"Initializing {controller_type} controller...")
        if controller_type == "custom":
            self.controller = CustomController(config)
            logger.info("Custom controller initialized")
        elif controller_type == "oracle":
            self.controller = OracleController(config)
            # Initialize Oracle with full day data
            energy_data = self.energy_loader.get_seasonal_day_data(
                self.sim_config["date"]
            )
            initial_battery = config["battery"]["initial_capacity"]
            self.controller.initialize(
                self.sim_date, energy_data, self.model_data, initial_battery
            )
        elif controller_type == "benchmark":
            self.controller = BenchmarkController(config)
            logger.info("Benchmark controller initialized")
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

        logger.info("=" * 60)

    def run_simulation(self) -> List[Dict[str, Any]]:
        """Run 24-hour simulation."""
        import time

        logger.info("=" * 60)
        logger.info("Starting 24-Hour Simulation")
        logger.info("=" * 60)
        logger.info(
            f"Date: {self.sim_config['date']}, "
            f"Controller: {self.sim_config['controller_type']}, "
            f"Interval: {self.output_interval}s"
        )

        start_time = time.time()
        results = []
        current_time = self.sim_date
        end_time = current_time + timedelta(hours=24)

        # Calculate total timesteps for progress tracking
        total_seconds = 24 * 3600
        total_timesteps = total_seconds // self.output_interval
        logger.info(
            f"Total timesteps: {total_timesteps} (24 hours / {self.output_interval}s interval)"
        )

        # Progress tracking
        last_log_time = current_time
        timestep_count = 0
        last_battery_level = self.sensors.get_battery_level()
        last_energy_cleanliness = 0.5

        logger.info("Simulation loop starting...")
        logger.info("-" * 60)

        while current_time < end_time:
            # Update energy cleanliness based on time
            energy_cleanliness = self.energy_loader.get_clean_energy_percentage(
                current_time
            )
            self.sensors.update_energy_cleanliness(energy_cleanliness)

            # Log energy cleanliness changes
            if abs(energy_cleanliness - last_energy_cleanliness) > 0.1:
                logger.debug(
                    f"[{current_time.strftime('%H:%M:%S')}] Energy cleanliness changed: "
                    f"{last_energy_cleanliness:.3f} -> {energy_cleanliness:.3f}"
                )
                last_energy_cleanliness = energy_cleanliness

            # Get current battery level
            battery_level = self.sensors.get_battery_level()

            # Log significant battery level changes
            if abs(battery_level - last_battery_level) > 5.0:
                logger.debug(
                    f"[{current_time.strftime('%H:%M:%S')}] Battery level changed: "
                    f"{last_battery_level:.2f}% -> {battery_level:.2f}%"
                )
                last_battery_level = battery_level

            # Check if charging is forced
            if self.sensors.is_charging():
                model_selected = None
                accuracy = 0.0
                latency = 0.0
                miss_type = "charging"
                energy_consumed = 0.0
                clean_energy_consumed = 0.0

                # Charge battery
                old_battery = battery_level
                self.sensors.charge_battery(self.output_interval)
                new_battery = self.sensors.get_battery_level()
                logger.debug(
                    f"[{current_time.strftime('%H:%M:%S')}] Charging: "
                    f"battery {old_battery:.2f}% -> {new_battery:.2f}%"
                )
            else:
                # Select model (Oracle uses get_decision_for_time, others use select_model)
                if isinstance(self.controller, OracleController):
                    model_selected = self.controller.get_decision_for_time(current_time)
                else:
                    model_selected = self.controller.select_model(
                        battery_level, energy_cleanliness, self.model_data
                    )

                if model_selected is None:
                    # No suitable model found or charging decision
                    accuracy = 0.0
                    latency = 0.0
                    miss_type = "no_model"
                    energy_consumed = 0.0
                    clean_energy_consumed = 0.0
                    logger.debug(
                        f"[{current_time.strftime('%H:%M:%S')}] No model selected "
                        f"(battery: {battery_level:.2f}%, energy: {energy_cleanliness:.3f})"
                    )
                else:
                    # Get model performance
                    model_info = self.model_data[model_selected]
                    accuracy = model_info["accuracy"]
                    latency = model_info["latency_ms"]
                    energy_consumed = model_info["energy_consumption"]

                    # Apply image quality modifier
                    if self.image_quality == "bad":
                        accuracy *= 0.9  # Reduce accuracy by 10% for bad quality

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
                        logger.debug(
                            f"[{current_time.strftime('%H:%M:%S')}] Model {model_selected} selected: "
                            f"accuracy={accuracy:.3f}, latency={latency:.2f}ms, "
                            f"energy={energy_consumed:.4f}%, miss={miss_type}"
                        )
                    else:
                        # Battery dead
                        miss_type = "large_miss"
                        clean_energy_consumed = 0.0
                        logger.warning(
                            f"[{current_time.strftime('%H:%M:%S')}] Battery insufficient for {model_selected}: "
                            f"required {energy_consumed:.4f}%, available {battery_level:.2f}%"
                        )

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
            timestep_count += 1

            # Progress logging (every hour or every 1000 timesteps)
            if (
                current_time - last_log_time
            ).total_seconds() >= 3600 or timestep_count % 1000 == 0:
                progress_pct = (timestep_count / total_timesteps) * 100
                elapsed_time = time.time() - start_time
                avg_time_per_step = (
                    elapsed_time / timestep_count if timestep_count > 0 else 0
                )
                remaining_steps = total_timesteps - timestep_count
                eta_seconds = remaining_steps * avg_time_per_step
                eta_minutes = eta_seconds / 60

                logger.info(
                    f"Progress: {timestep_count}/{total_timesteps} timesteps ({progress_pct:.1f}%) | "
                    f"Time: {current_time.strftime('%H:%M:%S')} | "
                    f"Battery: {battery_level:.2f}% | "
                    f"Elapsed: {elapsed_time:.1f}s | "
                    f"ETA: {eta_minutes:.1f}min"
                )
                last_log_time = current_time

        total_time = time.time() - start_time
        logger.info("-" * 60)
        logger.info(f"Simulation completed in {total_time:.2f}s")
        logger.info(f"Total results: {len(results)} timesteps")
        logger.info(
            f"Average time per timestep: {total_time / len(results) * 1000:.2f}ms"
        )
        logger.info("=" * 60)
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
