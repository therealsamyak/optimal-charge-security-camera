"""Main simulation runner."""

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger

from src.data.energy_loader import EnergyLoader
from src.data.model_data import ModelDataLoader
from src.sensors.simulation_sensors import SimulationSensors
from src.sensors.model_inference import run_yolo_inference
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
        battery_cfg = config.get("battery", {})
        initial_battery_pct = battery_cfg.get(
            "initial_capacity_pct", 
            battery_cfg.get("initial_capacity", 100.0)
        )
        logger.info(f"  Initial Battery: {initial_battery_pct}%")
        logger.info(f"  Max Battery: 100.0%")

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

        # Determine image path based on quality
        if self.image_quality == "good":
            self.image_path = "image1.png"
        elif self.image_quality == "bad":
            self.image_path = "image2.jpeg"
        else:
            logger.warning(
                f"Unknown image quality {self.image_quality}, defaulting to image1.png"
            )
            self.image_path = "image1.png"

        # Verify image exists
        if not Path(self.image_path).exists():
            logger.error(f"Image file not found: {self.image_path}")
            raise FileNotFoundError(f"Image file not found: {self.image_path}")

        logger.info(f"Using image: {self.image_path} for quality: {self.image_quality}")

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
            battery_cfg = config.get("battery", {})
            initial_battery = battery_cfg.get(
                "initial_capacity_pct",
                battery_cfg.get("initial_capacity", 100.0)
            )
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
                        battery_level, energy_cleanliness, self.model_data, self.image_path
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
                    # Run actual YOLO inference on the image
                    # Expected answer: 1 human in both images
                    expected_humans = 1
                    accuracy, latency = run_yolo_inference(
                        model_selected, self.image_path, expected_humans
                    )

                    # Get energy consumption from model data
                    model_info = self.model_data[model_selected]
                    energy_consumed = model_info["energy_consumption"]

                    # Note: accuracy is now 1.0 if model correctly identifies 1 human, 0.0 otherwise
                    # This is binary: correct (1.0) or wrong (0.0)
                    # If accuracy is 0.0, the model failed to correctly identify 1 human - this counts against accuracy

                    # Check if model meets thresholds (configured in config.jsonc)
                    # Accuracy threshold: if accuracy > threshold = success, if accuracy < threshold = failure
                    # Latency threshold: if latency > threshold = failure
                    accuracy_threshold = self.config["accuracy_threshold"]
                    latency_threshold = self.config["latency_threshold_ms"]

                    if accuracy < accuracy_threshold or latency > latency_threshold:
                        miss_type = "small_miss"
                        if accuracy < accuracy_threshold:
                            logger.debug(
                                f"Model {model_selected} failed accuracy check: "
                                f"accuracy {accuracy:.1f} < threshold {accuracy_threshold} "
                                f"(detected wrong number of humans)"
                            )
                        if latency > latency_threshold:
                            logger.debug(
                                f"Model {model_selected} failed latency check: "
                                f"{latency:.2f}ms > threshold {latency_threshold}ms"
                            )
                    else:
                        miss_type = "none"
                        logger.debug(
                            f"Model {model_selected} passed thresholds: "
                            f"accuracy {accuracy:.1f} >= {accuracy_threshold}, "
                            f"latency {latency:.2f}ms <= {latency_threshold}ms"
                        )

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
                "model_ran": model_selected
                or "",  # Model that was actually run during inference
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
        """Save simulation results to CSV in outputs/ folder."""
        if not results:
            logger.warning("No results to save")
            return

        # Ensure outputs directory exists
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)

        # If filename doesn't include outputs/, add it
        output_path = Path(filename)
        if not str(output_path).startswith("outputs/"):
            output_path = outputs_dir / output_path.name

        fieldnames = [
            "timestamp",
            "battery_level",
            "energy_cleanliness",
            "model_selected",
            "model_ran",
            "accuracy",
            "latency",
            "miss_type",
            "energy_consumed",
            "clean_energy_consumed",
        ]

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"Results saved to {output_path}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of simulation metrics."""
        return self.metrics.get_summary()
