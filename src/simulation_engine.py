import logging
import random
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.battery import Battery
from src.controller import Controller, OracleController
from src.energy_data import EnergyData
from src.config_loader import SimulationConfig


@dataclass
class Task:
    """Represents a single inference task."""

    timestamp: float
    accuracy_requirement: float
    latency_requirement: float
    completed: bool = False
    model_used: Optional[str] = None
    energy_used_mwh: float = 0.0
    clean_energy_used_mwh: float = 0.0
    missed_deadline: bool = False


class TaskGenerator:
    """Generates realistic security camera workload."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate_task(
        self, timestamp: float, accuracy_req: float, latency_req: float
    ) -> Optional[Task]:
        """Generate a single task with fixed requirements."""
        # Security cameras have periodic activity with some randomness
        if self.rng.random() < 0.1:  # 10% chance of no task
            return None

        return Task(
            timestamp=timestamp,
            accuracy_requirement=accuracy_req,
            latency_requirement=latency_req,
        )


class SimulationEngine:
    """Core simulation engine for security camera operations."""

    def __init__(
        self,
        config: SimulationConfig,
        controller: Controller,
        location: str,
        season: str,
        week: int,
        power_profiles: Dict,
        energy_data: Optional[EnergyData] = None,
    ):
        self.config = config
        self.controller = controller
        self.location = location
        self.season = season
        self.week = week
        self.power_profiles = power_profiles

        # Initialize components
        self.battery = Battery(
            capacity_wh=config.battery_capacity_wh,
            charge_rate_watts=config.charge_rate_watts,
        )

        self.energy_data = energy_data if energy_data is not None else EnergyData()
        self.task_generator = TaskGenerator()
        self.logger = logging.getLogger(__name__)

        # Simulation state
        self.current_time = 0.0
        self.tasks: List[Task] = []
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "missed_deadlines": 0,
            "small_model_tasks": 0,
            "large_model_tasks": 0,
            "small_model_misses": 0,
            "large_model_misses": 0,
            "total_energy_wh": 0.0,
            "clean_energy_mwh": 0.0,
            "dirty_energy_mwh": 0.0,
            "battery_levels": [],
            "clean_energy_wh": 0.0,
            "model_selections": {model: 0 for model in power_profiles.keys()},
            # New comprehensive metrics
            "peak_power_mw": 0.0,
            "charging_events_count": 0,
            "model_energy_breakdown": {model: 0.0 for model in power_profiles.keys()},
            "time_below_20_percent": 0,
            "time_above_80_percent": 0,
        }

        # Oracle controller will be prepared after method definitions

        # Load energy data for location and season
        self.clean_energy_data = self._load_clean_energy_data()

        # For oracle controller: prepare full-day data
        if isinstance(controller, OracleController):
            self.prepare_oracle_future_data()

    def _load_clean_energy_data(self) -> Dict[int, float]:
        """Load clean energy data for specific location and season."""
        try:
            # Map location to filename
            location_files = {
                "CA": "US-CAL-LDWP_2024_5_minute.csv",
                "FL": "US-FLA-FPL_2024_5_minute.csv",
                "NW": "US-NW-PSEI_2024_5_minute.csv",
                "NY": "US-NY-NYIS_2024_5_minute.csv",
            }

            filename = location_files.get(self.location)
            if not filename:
                raise ValueError(f"Unknown location: {self.location}")

            # Load energy data using existing EnergyData class
            region_name = filename.replace(".csv", "")
            if region_name not in self.energy_data.data:
                raise ValueError(f"Region {region_name} not found in energy data")
            data = self.energy_data.data[region_name].to_dict("records")

            # Filter by season and interpolate to 5-second intervals
            season_data = self._filter_by_season(data, self.season)
            return self._interpolate_energy_data(season_data)

        except Exception as e:
            self.logger.error(f"Failed to load energy data: {e}")
            raise RuntimeError(
                f"Failed to load energy data for {self.location} {self.season}: {e}"
            )

    def _filter_by_season(self, data: List[Dict], season: str) -> List[Dict]:
        """Filter energy data by season."""
        # Simple season mapping - would need more sophisticated logic for real data
        season_months = {
            "winter": [12, 1, 2],
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "fall": [9, 10, 11],
        }

        months = season_months.get(season, [1, 2, 3])
        filtered = []

        for entry in data:
            try:
                month = datetime.fromisoformat(entry["Datetime (UTC)"]).month
                if month in months:
                    filtered.append(entry)
            except (KeyError, ValueError):
                continue

        return filtered

    def _interpolate_energy_data(self, data: List[Dict]) -> Dict[int, float]:
        """Interpolate 5-minute energy data to 5-second intervals."""
        if not data:
            raise RuntimeError(
                f"No energy data available for {self.location} {self.season}"
            )

        # Create mapping from timestamp (seconds) to clean energy percentage
        energy_map = {}

        # Sort data by timestamp
        data.sort(key=lambda x: x["Datetime (UTC)"])

        # Convert to seconds and interpolate
        for i in range(len(data) - 1):
            current_time = datetime.fromisoformat(data[i]["Datetime (UTC)"])
            next_time = datetime.fromisoformat(data[i + 1]["Datetime (UTC)"])

            current_seconds = (
                current_time - current_time.replace(hour=0, minute=0, second=0)
            ).total_seconds()
            next_seconds = (
                next_time - next_time.replace(hour=0, minute=0, second=0)
            ).total_seconds()

            current_clean = float(
                data[i].get("Carbon-free energy percentage (CFE%)", 50.0)
            )
            next_clean = float(
                data[i + 1].get("Carbon-free energy percentage (CFE%)", 50.0)
            )

            # Interpolate for each 5-second interval
            steps = int((next_seconds - current_seconds) / 5)
            for step in range(steps):
                t = step / steps
                interpolated_clean = current_clean + t * (next_clean - current_clean)
                timestamp = current_seconds + step * 5
                energy_map[int(timestamp)] = interpolated_clean

        return energy_map

    def _create_default_energy_profile(self) -> Dict[int, float]:
        """Create default clean energy profile when data loading fails."""
        # Simple sinusoidal pattern for demonstration
        energy_map = {}
        for seconds in range(0, 24 * 3600, 5):  # 24 hours in 5-second steps
            hour = seconds / 3600
            # Peak solar at noon, lowest at night
            clean_percentage = max(0, 50 + 40 * ((hour - 12) / 12) ** 2)
            energy_map[seconds] = clean_percentage
        return energy_map

    def _get_clean_energy_percentage(self, timestamp: float) -> float:
        """Get clean energy percentage for given timestamp."""
        # Convert timestamp to seconds since midnight
        seconds_in_day = int(timestamp % (24 * 3600))

        # Find closest timestamp in energy data
        available_times = sorted(self.clean_energy_data.keys())
        if not available_times:
            raise RuntimeError("No clean energy data available")

        # Find the closest time point
        closest_time = min(available_times, key=lambda x: abs(x - seconds_in_day))
        return self.clean_energy_data[closest_time]

    def _get_available_models(self) -> Dict[str, Dict[str, float]]:
        """Get available models with their specs."""
        models = {}
        for name, profile in self.power_profiles.items():
            models[name] = {
                "accuracy": profile["accuracy"],
                "latency": profile["avg_inference_time_seconds"],  # Keep in seconds
                "power_cost": profile[
                    "model_power_mw"
                ],  # Power in mW from PowerProfiler
            }
        return models

    def _execute_task(self, task: Task) -> bool:
        """Execute a single task and return success status."""
        sim_id = f"{self.location}_{self.season}_week{self.week}"
        battery_level = self.battery.get_percentage()
        clean_energy_pct = self._get_clean_energy_percentage(task.timestamp)
        available_models = self._get_available_models()

        self.logger.debug(
            f"[{sim_id}] Executing task at t={task.timestamp}s, "
            f"battery={battery_level:.1f}%, clean_energy={clean_energy_pct:.1f}%"
        )

        # Get controller decision
        choice = self.controller.select_model(
            battery_level=battery_level,
            clean_energy_percentage=clean_energy_pct,
            user_accuracy_requirement=task.accuracy_requirement,
            user_latency_requirement=task.latency_requirement,
            available_models=available_models,
        )

        # Advance oracle timestep if this is an oracle controller
        if isinstance(self.controller, OracleController):
            self.controller.advance_timestep()

        self.logger.debug(
            f"[{sim_id}] Controller selected model: {choice.model_name}, charge: {choice.should_charge}"
        )

        # Check if selected model meets requirements
        model_specs = available_models[choice.model_name]
        meets_accuracy = model_specs["accuracy"] >= task.accuracy_requirement
        meets_latency = model_specs["latency"] <= task.latency_requirement

        if not (meets_accuracy and meets_latency):
            task.missed_deadline = True
            self.metrics["missed_deadlines"] += 1
            self.logger.debug(
                f"[{sim_id}] Task missed deadline: model {choice.model_name} "
                f"accuracy={model_specs['accuracy']:.3f} (need {task.accuracy_requirement:.3f}), "
                f"latency={model_specs['latency']:.3f}s (need {task.latency_requirement:.3f}s)"
            )

            # Track model-specific misses
            if choice.model_name in ["YOLOv10_N", "YOLOv10_S"]:
                self.metrics["small_model_misses"] += 1
            else:
                self.metrics["large_model_misses"] += 1

            return False

        # Execute inference
        power_mw = model_specs["power_cost"]
        duration_seconds = model_specs["latency"]  # Already in seconds

        self.logger.debug(
            f"[{sim_id}] Executing inference: {choice.model_name}, "
            f"power={power_mw}mW, duration={duration_seconds:.3f}s"
        )

        # Try to discharge battery
        success = self.battery.discharge(
            power_mw=power_mw,
            duration_seconds=duration_seconds,
            clean_energy_percentage=clean_energy_pct,
        )

        if not success:
            task.missed_deadline = True
            self.metrics["missed_deadlines"] += 1
            self.logger.debug(f"[{sim_id}] Task failed: insufficient battery")
            return False

        # Update task and metrics
        task.completed = True
        task.model_used = choice.model_name
        task.energy_used_mwh = power_mw * (duration_seconds / 3600)  # mW * hours = mWh
        task.clean_energy_used_mwh = task.energy_used_mwh * (clean_energy_pct / 100)

        self.metrics["completed_tasks"] += 1
        self.metrics["total_energy_wh"] += (
            task.energy_used_mwh / 1000
        )  # Convert mWh to Wh
        self.metrics["clean_energy_wh"] += (
            task.clean_energy_used_mwh / 1000
        )  # Convert mWh to Wh
        self.metrics["clean_energy_mwh"] += (
            task.clean_energy_used_mwh
        )  # Fix: Add missing MWh tracking

        # Track dirty energy (non-clean energy)
        dirty_energy_mwh = task.energy_used_mwh - task.clean_energy_used_mwh
        self.metrics["dirty_energy_mwh"] += dirty_energy_mwh

        # Track peak power usage
        if power_mw > self.metrics["peak_power_mw"]:
            self.metrics["peak_power_mw"] = power_mw

        # Track model-specific energy usage
        self.metrics["model_energy_breakdown"][choice.model_name] += (
            task.energy_used_mwh
        )

        self.metrics["model_selections"][choice.model_name] += 1

        # Track model usage categories
        if choice.model_name in ["YOLOv10_N", "YOLOv10_S"]:
            self.metrics["small_model_tasks"] += 1
        else:
            self.metrics["large_model_tasks"] += 1

        self.logger.debug(
            f"[{sim_id}] Task completed: battery now {self.battery.get_percentage():.1f}%"
        )

        # Handle charging decision
        if choice.should_charge:
            self.battery.charge(self.config.task_interval_seconds)
            self.metrics["charging_events_count"] += 1
            self.logger.debug(
                f"[{sim_id}] Charging battery for {self.config.task_interval_seconds}s"
            )

        return True

    def prepare_oracle_future_data(self):
        """Extract full day's clean energy and task requirements for oracle"""
        # Get clean energy for entire day
        clean_energy_series = self.get_full_day_clean_energy()

        # Generate task requirements for entire day
        task_requirements = self.generate_full_day_task_requirements()

        # Re-initialize oracle with full data
        self.controller = OracleController(
            clean_energy_series=clean_energy_series,
            task_requirements=task_requirements,
            config=self.config,
        )

    def get_full_day_clean_energy(self) -> List[float]:
        """Get clean energy percentages for entire day"""
        from datetime import datetime, timedelta

        # Get representative day for the season
        day = self.get_representative_day_for_season()

        clean_energy_series = []
        duration_seconds = self.config.duration_days * 24 * 3600
        task_interval = self.config.task_interval_seconds

        for timestamp in range(0, int(duration_seconds), task_interval):
            dt = datetime(2024, day["month"], day["day"]) + timedelta(seconds=timestamp)
            clean_energy_pct = self.energy_data.get_clean_energy_percentage(
                self.get_location_filename(), dt.strftime("%Y-%m-%d %H:%M:%S")
            )
            clean_energy_series.append(clean_energy_pct or 0.0)

        return clean_energy_series

    def generate_full_day_task_requirements(self) -> List[Dict]:
        """Generate task requirements for entire day"""
        task_requirements = []
        duration_seconds = self.config.duration_days * 24 * 3600
        task_interval = self.config.task_interval_seconds

        for timestamp in range(0, int(duration_seconds), task_interval):
            task_requirements.append(
                {
                    "accuracy": self.config.user_accuracy_requirement
                    / 100.0,  # Convert to decimal
                    "latency": self.config.user_latency_requirement,
                }
            )

        return task_requirements

    def get_representative_day_for_season(self) -> Dict[str, int]:
        """Get representative day for current season"""
        season_days = {
            "winter": {"month": 1, "day": 15},  # January 15
            "spring": {"month": 4, "day": 15},  # April 15
            "summer": {"month": 7, "day": 15},  # July 15
            "fall": {"month": 10, "day": 15},  # October 15
        }
        return season_days.get(self.season, {"month": 1, "day": 15})

    def get_location_filename(self) -> str:
        """Get energy data filename for location"""
        location_mapping = {
            "CA": "US-CAL-LDWP_2024_5_minute",
            "FL": "US-FLA-FPL_2024_5_minute",
            "NW": "US-NW-PSEI_2024_5_minute",
            "NY": "US-NY-NYIS_2024_5_minute",
        }
        return location_mapping.get(self.location, "US-CAL-LDWP_2024_5_minute")

    def run(self) -> Dict:
        """Run the complete simulation."""
        sim_id = f"{self.location}_{self.season}_week{self.week}"
        self.logger.info(f"[{sim_id}] Starting simulation")
        self.logger.debug(
            f"[{sim_id}] Duration: {self.config.duration_days} days, "
            f"task interval: {self.config.task_interval_seconds}s, "
            f"time acceleration: {self.config.time_acceleration}x"
        )

        duration_seconds = self.config.duration_days * 24 * 3600
        task_interval = self.config.task_interval_seconds
        total_iterations = int(duration_seconds / task_interval)

        start_time = time.time()
        last_progress_time = start_time
        progress_interval = 30.0  # Log progress every 30 seconds

        for iteration, timestamp in enumerate(
            range(0, int(duration_seconds), task_interval)
        ):
            self.current_time = timestamp

            # Log progress periodically
            current_time = time.time()
            if current_time - last_progress_time >= progress_interval:
                progress = (iteration / total_iterations) * 100
                elapsed = current_time - start_time
                eta = (
                    (elapsed / iteration * (total_iterations - iteration))
                    if iteration > 0
                    else 0
                )
                self.logger.info(
                    f"[{sim_id}] Progress: {progress:.1f}% ({iteration}/{total_iterations}), "
                    f"elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s"
                )
                self.logger.debug(
                    f"[{sim_id}] Current metrics: {self.metrics['total_tasks']} tasks, "
                    f"{self.metrics['completed_tasks']} completed, "
                    f"battery: {self.battery.get_percentage():.1f}%"
                )
                last_progress_time = current_time

            # Generate task
            task = self.task_generator.generate_task(
                timestamp,
                self.config.user_accuracy_requirement
                / 100.0,  # Convert percentage to decimal
                self.config.user_latency_requirement,
            )
            if task is not None:
                self.tasks.append(task)
                self.metrics["total_tasks"] += 1
                self._execute_task(task)

                # Track battery level and time spent at different levels
                battery_level = self.battery.get_percentage()
                self.metrics["battery_levels"].append(
                    {"timestamp": timestamp, "level": battery_level}
                )

                # Track time spent at different battery levels
                if battery_level < 20.0:
                    self.metrics["time_below_20_percent"] += (
                        self.config.task_interval_seconds
                    )
                elif battery_level > 80.0:
                    self.metrics["time_above_80_percent"] += (
                        self.config.task_interval_seconds
                    )

            # Apply time acceleration
            if self.config.time_acceleration > 1:
                time.sleep(
                    0.001 / self.config.time_acceleration
                )  # Minimal delay for acceleration

        elapsed = time.time() - start_time
        self.logger.info(f"[{sim_id}] Simulation completed in {elapsed:.2f} seconds")
        self.logger.debug(
            f"[{sim_id}] Final state: {self.metrics['total_tasks']} total tasks, "
            f"{self.metrics['completed_tasks']} completed, "
            f"{self.metrics['missed_deadlines']} missed deadlines"
        )

        # Calculate final metrics
        self._calculate_final_metrics()

        # Validate metrics consistency
        self._validate_metrics_consistency()

        return self.metrics

    def _calculate_final_metrics(self):
        """Calculate final simulation metrics."""
        # Always calculate metrics, even if no tasks were generated
        if self.metrics["total_tasks"] == 0:
            self.metrics["task_completion_rate"] = 0.0
            self.metrics["clean_energy_percentage"] = 0.0
            self.metrics["small_model_miss_rate"] = 0.0
            self.metrics["large_model_miss_rate"] = 0.0
            return

        # Calculate miss rates
        if self.metrics["small_model_tasks"] > 0:
            self.metrics["small_model_miss_rate"] = (
                self.metrics["small_model_misses"]
                / self.metrics["small_model_tasks"]
                * 100
            )
        else:
            self.metrics["small_model_miss_rate"] = 0.0

        if self.metrics["large_model_tasks"] > 0:
            self.metrics["large_model_miss_rate"] = (
                self.metrics["large_model_misses"]
                / self.metrics["large_model_tasks"]
                * 100
            )
        else:
            self.metrics["large_model_miss_rate"] = 0.0

        # Calculate clean energy percentage
        if self.metrics["total_energy_wh"] > 0:
            self.metrics["clean_energy_percentage"] = (
                self.metrics.get("clean_energy_wh", 0.0)
                / self.metrics["total_energy_wh"]
                * 100
            )
        else:
            self.metrics["clean_energy_percentage"] = 0.0

        # Calculate task completion rate
        self.metrics["task_completion_rate"] = (
            self.metrics["completed_tasks"] / self.metrics["total_tasks"] * 100
        )

        # Calculate additional comprehensive metrics
        duration_seconds = self.config.duration_days * 24 * 3600
        self.metrics["total_simulation_time"] = duration_seconds

        # Energy efficiency metrics
        if self.metrics["total_tasks"] > 0:
            self.metrics["energy_per_task_wh"] = (
                self.metrics["total_energy_wh"] / self.metrics["total_tasks"]
            )
        else:
            self.metrics["energy_per_task_wh"] = 0.0

        if self.metrics["completed_tasks"] > 0:
            self.metrics["clean_energy_per_task_wh"] = (
                self.metrics["clean_energy_wh"] / self.metrics["completed_tasks"]
            )
        else:
            self.metrics["clean_energy_per_task_wh"] = 0.0

        # Average power usage
        if duration_seconds > 0:
            self.metrics["average_power_mw"] = (
                self.metrics["total_energy_wh"] * 1000
            ) / duration_seconds  # Convert Wh back to mW
        else:
            self.metrics["average_power_mw"] = 0.0

        # Battery efficiency score (tasks per percent battery depletion)
        initial_battery = 0.0
        final_battery = 0.0
        if self.metrics["battery_levels"]:
            initial_battery = self.metrics["battery_levels"][0]["level"]
            final_battery = self.metrics["battery_levels"][-1]["level"]
            battery_depletion = initial_battery - final_battery
            if battery_depletion > 0:
                self.metrics["battery_efficiency_score"] = (
                    self.metrics["completed_tasks"] / battery_depletion
                )
            else:
                self.metrics["battery_efficiency_score"] = (
                    float("inf") if self.metrics["completed_tasks"] > 0 else 0.0
                )
        else:
            self.metrics["battery_efficiency_score"] = 0.0

        # Battery depletion rate per hour
        if self.config.duration_days > 0:
            self.metrics["battery_depletion_rate_per_hour"] = (
                initial_battery - final_battery
            ) / (self.config.duration_days * 24)
        else:
            self.metrics["battery_depletion_rate_per_hour"] = 0.0

    def _validate_metrics_consistency(self):
        """Validate that metrics are logically consistent."""
        # Energy balance check
        calculated_total_mwh = self.metrics.get(
            "clean_energy_mwh", 0.0
        ) + self.metrics.get("dirty_energy_mwh", 0.0)
        calculated_total_wh = calculated_total_mwh * 1000
        if abs(calculated_total_wh - self.metrics.get("total_energy_wh", 0.0)) > 0.001:
            self.logger.warning(
                f"Energy balance mismatch: calculated {calculated_total_wh:.3f}Wh vs recorded {self.metrics.get('total_energy_wh', 0.0):.3f}Wh"
            )

        # Task completion consistency
        if self.metrics.get("completed_tasks", 0) > self.metrics.get("total_tasks", 0):
            self.logger.error(
                f"Task completion inconsistency: completed_tasks ({self.metrics.get('completed_tasks', 0)}) > total_tasks ({self.metrics.get('total_tasks', 0)})"
            )

        # Model selection consistency
        total_model_selections = sum(self.metrics.get("model_selections", {}).values())
        if total_model_selections != self.metrics.get("completed_tasks", 0):
            self.logger.warning(
                f"Model selection inconsistency: sum of selections ({total_model_selections}) != completed_tasks ({self.metrics.get('completed_tasks', 0)})"
            )

        # Battery level sanity checks
        battery_levels = self.metrics.get("battery_levels", [])
        if battery_levels and len(battery_levels) > 1:
            battery_levels[0]["level"]
            final = battery_levels[-1]["level"]
            current = self.battery.get_percentage()
            if abs(final - current) > 1.0:
                self.logger.warning(
                    f"Battery level inconsistency: final recorded {final:.1f}% vs current {current:.1f}%"
                )

        # Clean energy percentage sanity check
        if self.metrics.get("total_energy_wh", 0) > 0:
            calculated_clean_pct = (
                self.metrics.get("clean_energy_wh", 0)
                / self.metrics.get("total_energy_wh", 1)
            ) * 100
            recorded_clean_pct = self.metrics.get("clean_energy_percentage", 0)
            if abs(calculated_clean_pct - recorded_clean_pct) > 0.1:
                self.logger.warning(
                    f"Clean energy percentage inconsistency: calculated {calculated_clean_pct:.2f}% vs recorded {recorded_clean_pct:.2f}%"
                )

        # Model selection consistency
        total_model_selections = sum(self.metrics.get("model_selections", {}).values())
        if total_model_selections != self.metrics.get("completed_tasks", 0):
            self.logger.warning(
                f"Model selection inconsistency: sum of selections ({total_model_selections}) != completed_tasks ({self.metrics.get('completed_tasks', 0)})"
            )

        # Battery level sanity checks
        battery_levels = self.metrics.get("battery_levels", [])
        if battery_levels and len(battery_levels) > 1:
            battery_levels[0]["level"]
            final = battery_levels[-1]["level"]
            current = self.battery.get_percentage()
            if abs(final - current) > 1.0:
                self.logger.warning(
                    f"Battery level inconsistency: final recorded {final:.1f}% vs current {current:.1f}%"
                )

        # Clean energy percentage sanity check
        if self.metrics.get("total_energy_wh", 0) > 0:
            calculated_clean_pct = (
                self.metrics.get("clean_energy_wh", 0)
                / self.metrics.get("total_energy_wh", 1)
            ) * 100
            recorded_clean_pct = self.metrics.get("clean_energy_percentage", 0)
            if abs(calculated_clean_pct - recorded_clean_pct) > 0.1:
                self.logger.warning(
                    f"Clean energy percentage inconsistency: calculated {calculated_clean_pct:.2f}% vs recorded {recorded_clean_pct:.2f}%"
                )
