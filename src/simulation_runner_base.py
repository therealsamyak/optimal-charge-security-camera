"""
Shared simulation logic for both basic and batch simulation runners.

This module contains common functionality for running simulations,
including controller creation, simulation execution, and result handling.
"""

import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config_loader import BatchConfig, ConfigLoader, SimulationConfig
from src.controller import (
    CustomController,
    NaiveStrongController,
    NaiveWeakController,
    OracleController,
)
from src.energy_data import EnergyData
from src.simulation_engine import SimulationEngine


class SimulationRunnerBase:
    """Base class for simulation runners with shared functionality."""

    def __init__(self, config_path: str = "config.jsonc", max_workers: int = 4):
        self.config_loader = ConfigLoader(config_path)
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

        # Load power profiles
        self.power_profiles = self._load_power_profiles()

        # Load energy data once and reuse
        self.energy_data = EnergyData()

        # Results storage
        self.all_results = []
        self.failed_simulations = []

        # Validate configuration
        if not self.config_loader.validate_config():
            raise ValueError("Invalid configuration")

    def _load_power_profiles(self):
        """Load power profiles using PowerProfiler."""
        from src.power_profiler import PowerProfiler

        profiler = PowerProfiler()
        profiler.load_profiles()  # Load profiles from file
        return profiler.get_all_models_data()

    def _create_controller(self, controller_type: str):
        """Create controller instance based on type."""
        controllers = {
            "naive_weak": NaiveWeakController(),
            "naive_strong": NaiveStrongController(),
            "custom": CustomController(),
        }

        if controller_type == "oracle":
            # Oracle controller needs future data - simplified for now
            return OracleController({}, 0)

        if controller_type not in controllers:
            raise ValueError(f"Unknown controller type: {controller_type}")

        return controllers[controller_type]

    def _create_simulation_config_with_overrides(
        self,
        base_config: SimulationConfig,
        accuracy_override: Optional[float] = None,
        latency_override: Optional[float] = None,
        battery_capacity_override: Optional[float] = None,
        charge_rate_override: Optional[float] = None,
    ) -> SimulationConfig:
        """Create simulation config with parameter overrides."""
        return SimulationConfig(
            duration_days=base_config.duration_days,
            task_interval_seconds=base_config.task_interval_seconds,
            time_acceleration=base_config.time_acceleration,
            user_accuracy_requirement=accuracy_override
            or base_config.user_accuracy_requirement,
            user_latency_requirement=latency_override
            or base_config.user_latency_requirement,
            battery_capacity_wh=battery_capacity_override
            or base_config.battery_capacity_wh,
            charge_rate_watts=charge_rate_override or base_config.charge_rate_watts,
            locations=base_config.locations,
            seasons=base_config.seasons,
        )

    def _run_single_simulation(
        self,
        simulation_params: Dict[str, Any],
        sim_config: Optional[SimulationConfig] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run a single simulation and return results."""
        simulation_start_time = datetime.now()
        simulation_id = simulation_params.get("simulation_id", "unknown")

        try:
            # Extract parameters
            location = simulation_params["location"]
            season = simulation_params["season"]
            week = simulation_params["week"]
            controller_type = simulation_params["controller"]

            self.logger.info(
                f"[{simulation_id}] Starting simulation: {location} {season} week {week} with {controller_type}"
            )
            self.logger.debug(
                f"[{simulation_id}] Parameters: accuracy={sim_config.user_accuracy_requirement if sim_config else 'default'}, "
                f"latency={sim_config.user_latency_requirement if sim_config else 'default'}, "
                f"battery={sim_config.battery_capacity_wh if sim_config else 'default'}Wh, "
                f"charge_rate={sim_config.charge_rate_watts if sim_config else 'default'}W"
            )

            # Create controller
            self.logger.debug(
                f"[{simulation_id}] Creating controller: {controller_type}"
            )
            controller = self._create_controller(controller_type)

            # Get simulation config (use provided override or default)
            if sim_config is None:
                self.logger.debug(f"[{simulation_id}] Using default simulation config")
                sim_config = self.config_loader.get_simulation_config()
            else:
                self.logger.debug(
                    f"[{simulation_id}] Using overridden simulation config"
                )

            # Create and run simulation
            self.logger.debug(f"[{simulation_id}] Creating simulation engine")
            engine = SimulationEngine(
                config=sim_config,
                controller=controller,
                location=location,
                season=season,
                week=week,
                power_profiles=self.power_profiles,
                energy_data=self.energy_data,
            )

            self.logger.debug(f"[{simulation_id}] Starting simulation execution")
            # Run simulation
            metrics = engine.run()

            # Calculate execution time
            execution_time = (datetime.now() - simulation_start_time).total_seconds()
            self.logger.info(
                f"[{simulation_id}] Simulation completed in {execution_time:.2f} seconds"
            )
            print(f"✓ [{simulation_id}] Completed in {execution_time:.2f}s")
            self.logger.debug(
                f"[{simulation_id}] Metrics: {metrics.get('total_tasks', 0)} tasks, "
                f"{metrics.get('completed_tasks', 0)} completed, "
                f"{metrics.get('task_completion_rate', 0):.1f}% completion rate"
            )

            # Add simulation metadata
            result = {
                **metrics,
                "location": location,
                "season": season,
                "week": week,
                "controller": controller_type,
                "simulation_id": simulation_id,
                "timestamp": simulation_start_time.isoformat(),
                "execution_time_seconds": execution_time,
                "success": True,
            }

            # Add parameter overrides if present
            if (
                sim_config.user_accuracy_requirement
                != self.config_loader.get_simulation_config().user_accuracy_requirement
            ):
                result["accuracy_requirement"] = sim_config.user_accuracy_requirement
            if (
                sim_config.user_latency_requirement
                != self.config_loader.get_simulation_config().user_latency_requirement
            ):
                result["latency_requirement"] = sim_config.user_latency_requirement
            if (
                sim_config.battery_capacity_wh
                != self.config_loader.get_simulation_config().battery_capacity_wh
            ):
                result["battery_capacity_wh"] = sim_config.battery_capacity_wh
            if (
                sim_config.charge_rate_watts
                != self.config_loader.get_simulation_config().charge_rate_watts
            ):
                result["charge_rate_watts"] = sim_config.charge_rate_watts

            self.logger.info(f"Completed simulation: {result['simulation_id']}")
            return result

        except Exception as e:
            import traceback

            execution_time = (datetime.now() - simulation_start_time).total_seconds()
            error_msg = f"[{simulation_id}] Simulation failed after {execution_time:.2f}s: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"[{simulation_id}] Traceback: {traceback.format_exc()}")

            # Record failure
            failure_record = {
                **simulation_params,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": simulation_start_time.isoformat(),
                "execution_time_seconds": execution_time,
                "success": False,
            }
            self.failed_simulations.append(failure_record)

            return None

    def _generate_simulation_list(
        self,
        locations: List[str],
        seasons: List[str],
        controllers: List[str],
        weeks: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate list of simulations to run."""
        if weeks is None:
            weeks = [1, 2, 3]  # Default to 3 weeks

        simulations = []
        for location in locations:
            for season in seasons:
                for controller in controllers:
                    for week in weeks:
                        sim_params = {
                            "location": location,
                            "season": season,
                            "week": week,
                            "controller": controller,
                            "simulation_id": f"{location}_{season}_week{week}_{controller}",
                        }
                        simulations.append(sim_params)

        self.logger.info(f"Generated {len(simulations)} simulations to run")
        print(f"📋 Generated {len(simulations)} simulations to run")
        return simulations

    def generate_parameter_variations(
        self, batch_config: BatchConfig
    ) -> List[Dict[str, Any]]:
        """Generate parameter variations for batch simulation."""
        variations = []

        # Set random seed if provided
        if batch_config.random_seed is not None:
            random.seed(batch_config.random_seed)
            self.logger.info(f"Using random seed: {batch_config.random_seed}")

        for i in range(batch_config.num_variations):
            accuracy = random.uniform(
                batch_config.accuracy_range["min"], batch_config.accuracy_range["max"]
            )
            latency = random.uniform(
                batch_config.latency_range["min"], batch_config.latency_range["max"]
            )
            battery_capacity = random.uniform(
                batch_config.battery_capacity_range["min"],
                batch_config.battery_capacity_range["max"],
            )
            charge_rate = random.uniform(
                batch_config.charge_rate_range["min"],
                batch_config.charge_rate_range["max"],
            )

            variation = {
                "variation_id": i + 1,
                "accuracy_requirement": round(accuracy, 2),
                "latency_requirement": round(latency, 2),
                "battery_capacity_wh": round(battery_capacity, 2),
                "charge_rate_watts": round(charge_rate, 2),
            }
            variations.append(variation)

        self.logger.info(f"Generated {len(variations)} parameter variations")
        return variations

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all simulations."""
        if not self.all_results:
            return {}

        # Calculate aggregate statistics
        total_tasks = sum(r.get("total_tasks", 0) for r in self.all_results)
        total_completed = sum(r.get("completed_tasks", 0) for r in self.all_results)
        total_energy = sum(r.get("total_energy_wh", 0.0) for r in self.all_results)
        total_clean_energy = sum(
            r.get("clean_energy_wh", 0.0) for r in self.all_results
        )

        # Group by controller
        controller_stats = {}
        for result in self.all_results:
            controller = result.get("controller", "unknown")
            if controller not in controller_stats:
                controller_stats[controller] = {
                    "count": 0,
                    "avg_completion_rate": 0.0,
                    "avg_clean_energy_pct": 0.0,
                    "total_energy": 0.0,
                }

            stats = controller_stats[controller]
            stats["count"] += 1
            stats["avg_completion_rate"] += result.get("task_completion_rate", 0.0)
            stats["avg_clean_energy_pct"] += result.get("clean_energy_percentage", 0.0)
            stats["total_energy"] += result.get("total_energy_wh", 0.0)

        # Calculate averages
        for controller, stats in controller_stats.items():
            if stats["count"] > 0:
                stats["avg_completion_rate"] /= stats["count"]
                stats["avg_clean_energy_pct"] /= stats["count"]

        return {
            "total_simulations": len(self.all_results),
            "total_tasks": total_tasks,
            "total_completed_tasks": total_completed,
            "overall_completion_rate": (
                (total_completed / total_tasks * 100) if total_tasks > 0 else 0
            ),
            "total_energy_wh": total_energy,
            "total_clean_energy_wh": total_clean_energy,
            "overall_clean_energy_percentage": (
                (total_clean_energy / total_energy * 100) if total_energy > 0 else 0
            ),
            "controller_performance": controller_stats,
            "failed_simulations": len(self.failed_simulations),
        }
