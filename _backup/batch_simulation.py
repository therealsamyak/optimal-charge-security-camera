#!/usr/bin/env python3
"""
Batch Security Camera Simulation Runner

This file runs comprehensive batch simulations with parameter variations:
- Generates multiple parameter sets based on config.jsonc batch_run section
- For each parameter set, runs full 192 simulation matrix (4 locations × 4 seasons × 4 controllers × 3 weeks)
- Outputs detailed CSV files with batch-run prefix and timestamps
- Terminates on any failure with clear error messages

Usage:
    python batch_simulation.py [--parallel] [--workers N] [--config config.jsonc]
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.logging_config import setup_logging
from src.metrics_collector import JSONExporter
from src.simulation_runner_base import SimulationRunnerBase

# Initialize logging
logger = setup_logging()


class BatchSimulationRunner(SimulationRunnerBase):
    """Orchestrates execution of batch simulations with parameter variations."""

    def __init__(self, config_path: str = "config.jsonc", max_workers: int = 100):
        logger.info("Initializing BatchSimulationRunner")
        logger.debug(f"Config path: {config_path}, Max workers: {max_workers}")

        super().__init__(config_path, max_workers)

        # Get batch configuration
        self.batch_config = self.config_loader.get_batch_config()
        logger.info(f"Batch configuration loaded: {self.batch_config}")

        # Initialize exporter
        output_dir = self.config_loader.get_output_dir()
        self.exporter = JSONExporter(output_dir)
        logger.info(f"Output directory: {output_dir}")

        # Generate timestamp for filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Timestamp for this run: {self.timestamp}")

        logger.info("BatchSimulationRunner initialization complete")

    def run_batch_simulations(self, parallel: bool = True) -> bool:
        """
        Run batch simulations with parameter variations.

        Args:
            parallel: Whether to run simulations in parallel

        Returns:
            True if all simulations succeeded, False if any failed
        """
        logger.info("=" * 80)
        logger.info("STARTING BATCH SIMULATION RUN")
        logger.info("=" * 80)

        start_time = datetime.now()
        logger.info(f"Batch simulation start time: {start_time}")
        logger.info(f"Parallel execution: {parallel}")

        try:
            # Generate parameter variations
            logger.info("Generating parameter variations...")
            variations = self.generate_parameter_variations(self.batch_config)
            logger.info(f"Generated {len(variations)} parameter variations")
            logger.debug(
                f"Variation details: {variations[:2] if len(variations) > 2 else variations}"
            )

            # Get configuration for simulation matrix
            locations = self.config_loader.get_locations()
            seasons = self.config_loader.get_seasons()
            controllers = self.config_loader.get_controllers()

            logger.info("Configuration loaded:")
            logger.info(f"  Locations: {locations}")
            logger.info(f"  Seasons: {seasons}")
            logger.info(f"  Controllers: {controllers}")

            # Generate base simulation list (full matrix: 4 locations × 4 seasons × 4 controllers × 3 weeks = 192)
            base_simulations = self._generate_simulation_list(
                locations=locations,
                seasons=seasons,
                controllers=controllers,
                weeks=[1, 2, 3],  # All 3 weeks
            )

            total_simulations = len(variations) * len(base_simulations)
            logger.info(
                f"Base simulation matrix: {len(base_simulations)} simulations per variation"
            )
            logger.info(f"Total simulations to run: {total_simulations:,}")

            successful_results = []

            # Run simulations for each parameter variation
            for i, variation in enumerate(variations):
                variation_start_time = datetime.now()
                logger.info(
                    f"Running variation {variation['variation_id']}/{len(variations)}"
                )
                logger.debug(f"Variation parameters: {variation}")

                # Create simulation config with parameter overrides
                base_config = self.config_loader.get_simulation_config()
                sim_config = self._create_simulation_config_with_overrides(
                    base_config=base_config,
                    accuracy_override=variation["accuracy_requirement"],
                    latency_override=variation["latency_requirement"],
                    battery_capacity_override=variation["battery_capacity_wh"],
                    charge_rate_override=variation["charge_rate_watts"],
                )
                logger.debug("Simulation config created with overrides")

                # Run all simulations for this variation
                variation_results = self._run_variation_simulations(
                    base_simulations=base_simulations,
                    sim_config=sim_config,
                    variation=variation,
                    parallel=parallel,
                )

                successful_results.extend(variation_results)

                variation_time = (datetime.now() - variation_start_time).total_seconds()
                logger.info(
                    f"Variation {variation['variation_id']} completed in {variation_time:.2f}s, "
                    f"{len(variation_results)} successful results"
                )

            # Check if any simulations failed
            if len(self.failed_simulations) > 0:
                logger.error(f"{len(self.failed_simulations)} simulations failed")
                logger.error(
                    "TERMINATING DUE TO FAILURES - NO OUTPUT WILL BE GENERATED"
                )

                # Print clear error information
                for failure in self.failed_simulations[:5]:  # Show first 5 failures
                    logger.error(
                        f"FAILED: {failure.get('simulation_id', 'unknown')} - {failure.get('error', 'unknown error')}"
                    )

                if len(self.failed_simulations) > 5:
                    logger.error(
                        f"... and {len(self.failed_simulations) - 5} more failures"
                    )

                return False

            # Store successful results
            self.all_results = successful_results
            logger.info(f"Collected {len(successful_results)} successful results")

            # Export results
            logger.info("Exporting batch results...")
            self._export_batch_results()

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"All {len(successful_results)} batch simulations completed successfully in {duration:.2f} seconds"
            )
            return True

        except Exception as e:
            logger.error(f"Batch simulation failed with exception: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def _run_variation_simulations(
        self, base_simulations: list, sim_config, variation: dict, parallel: bool = True
    ) -> list:
        """Run all simulations for a single parameter variation."""
        successful_results = []
        variation_start_time = datetime.now()

        logger.info(
            f"[Variation {variation['variation_id']}] Starting {len(base_simulations)} simulations"
        )
        logger.debug(f"[Variation {variation['variation_id']}] Parameters: {variation}")

        if parallel:
            logger.debug(
                f"[Variation {variation['variation_id']}] Using parallel execution with {self.max_workers} workers"
            )
            import concurrent.futures

            # Run simulations in parallel
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # Submit all simulations with variation config
                future_to_sim = {
                    executor.submit(self._run_single_simulation, sim, sim_config): sim
                    for sim in base_simulations
                }
                logger.info(
                    f"[Variation {variation['variation_id']}] Submitted all {len(base_simulations)} simulations to process pool"
                )

                completed_count = 0
                error_count = 0
                print(
                    f"[Variation {variation['variation_id']}] Processing {len(base_simulations)} simulations..."
                )
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_sim):
                    sim = future_to_sim[future]
                    completed_count += 1
                    try:
                        result = future.result()
                        if result:
                            # Add variation metadata
                            result["variation_id"] = variation["variation_id"]
                            successful_results.append(result)
                            logger.debug(
                                f"[Variation {variation['variation_id']}] ✓ Simulation {sim['simulation_id']} completed"
                            )
                        else:
                            error_count += 1
                            logger.warning(
                                f"[Variation {variation['variation_id']}] ✗ Simulation {sim['simulation_id']} returned no result"
                            )

                        if completed_count % 10 == 0:  # Log every 10 completions
                            progress = (completed_count / len(base_simulations)) * 100
                            print(
                                f"[Variation {variation['variation_id']}] Progress: {completed_count}/{len(base_simulations)} ({progress:.1f}%) - Errors: {error_count}"
                            )
                            logger.info(
                                f"[Variation {variation['variation_id']}] Progress: {progress:.1f}% ({completed_count}/{len(base_simulations)}) - Errors: {error_count}"
                            )

                    except Exception as e:
                        error_count += 1
                        logger.error(
                            f"[Variation {variation['variation_id']}] Simulation {sim['simulation_id']} raised exception: {e}"
                        )
                        logger.exception(
                            f"[Variation {variation['variation_id']}] Full traceback for simulation {sim['simulation_id']}"
                        )
                        # Failure already recorded in _run_single_simulation
        else:
            logger.debug(
                f"[Variation {variation['variation_id']}] Using sequential execution"
            )
            for i, sim in enumerate(base_simulations):
                if (i + 1) % 10 == 0:  # Log every 10 simulations
                    progress = ((i + 1) / len(base_simulations)) * 100
                    logger.info(
                        f"[Variation {variation['variation_id']}] Progress: {progress:.1f}% ({i + 1}/{len(base_simulations)})"
                    )

                result = self._run_single_simulation(sim, sim_config)
                if result:
                    # Add variation metadata
                    result["variation_id"] = variation["variation_id"]
                    successful_results.append(result)
                    logger.debug(
                        f"[Variation {variation['variation_id']}] ✓ Simulation {sim['simulation_id']} completed"
                    )
                else:
                    logger.warning(
                        f"[Variation {variation['variation_id']}] ✗ Simulation {sim['simulation_id']} returned no result"
                    )

        variation_time = (datetime.now() - variation_start_time).total_seconds()
        success_rate = (len(successful_results) / len(base_simulations)) * 100
        logger.info(
            f"[Variation {variation['variation_id']}] Completed in {variation_time:.2f}s, "
            f"{len(successful_results)} successful, {len(base_simulations) - len(successful_results)} failed ({success_rate:.1f}% success rate)"
        )

        return successful_results

    def _generate_aggregated_results(
        self, all_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate aggregated results for JSON export."""
        if not all_results:
            return []

        # Group by controller, location, season
        aggregated = {}

        for result in all_results:
            key = (
                result.get("controller", "unknown"),
                result.get("location", "unknown"),
                result.get("season", "unknown"),
            )

            if key not in aggregated:
                aggregated[key] = {
                    "controller": key[0],
                    "location": key[1],
                    "season": key[2],
                    "total_simulations": 0,
                    "successful_simulations": 0,
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "total_energy_wh": 0.0,
                    "clean_energy_wh": 0.0,
                    "missed_deadlines": 0,
                }

            agg = aggregated[key]
            agg["total_simulations"] += 1

            if result.get("success", True):
                agg["successful_simulations"] += 1
                agg["total_tasks"] += result.get("total_tasks", 0)
                agg["completed_tasks"] += result.get("completed_tasks", 0)
                agg["total_energy_wh"] += result.get("total_energy_wh", 0.0)
                agg["clean_energy_wh"] += result.get("clean_energy_wh", 0.0)
                agg["missed_deadlines"] += result.get("missed_deadlines", 0)

        # Calculate derived metrics
        for agg in aggregated.values():
            agg["success_rate"] = (
                agg["successful_simulations"] / agg["total_simulations"]
            ) * 100
            agg["avg_task_completion_rate"] = (
                (agg["completed_tasks"] / agg["total_tasks"] * 100)
                if agg["total_tasks"] > 0
                else 0
            )
            agg["avg_clean_energy_percentage"] = (
                (agg["clean_energy_wh"] / agg["total_energy_wh"] * 100)
                if agg["total_energy_wh"] > 0
                else 0
            )
            agg["avg_battery_efficiency"] = agg["avg_task_completion_rate"] * (
                agg["avg_clean_energy_percentage"] / 100
            )
            agg["total_clean_energy_mwh"] = agg["clean_energy_wh"] / 1000
            agg["total_dirty_energy_mwh"] = (
                agg["total_energy_wh"] - agg["clean_energy_wh"]
            ) / 1000
            agg["energy_per_task_wh"] = (
                agg["total_energy_wh"] / agg["completed_tasks"]
                if agg["completed_tasks"] > 0
                else 0
            )
            agg["clean_energy_per_task_wh"] = (
                agg["clean_energy_wh"] / agg["completed_tasks"]
                if agg["completed_tasks"] > 0
                else 0
            )
            agg["timestamp"] = datetime.now().isoformat()

        return list(aggregated.values())

    def _export_batch_results(self):
        """Export batch simulation results to JSON files with timestamps."""
        logger.info("Starting batch results export process")

        if not self.all_results:
            logger.warning("No results to export")
            return

        try:
            # Generate aggregated data for export
            logger.info("Generating aggregated data for export...")
            aggregated_data = self._generate_aggregated_results(self.all_results)
            logger.info(f"Generated {len(aggregated_data)} aggregated data records")

            # Export all results to hierarchical JSON
            results_filename = f"batch-run-{self.timestamp}-results.json"
            logger.info(f"Exporting results to {results_filename}")
            results_file = self.exporter.export_results(
                all_simulations=self.all_results,
                aggregated_data=aggregated_data,
                filename=results_filename,
            )
            if results_file:
                logger.info(f"Results exported to {results_file}")

                # Log file size
                if Path(results_file).exists():
                    file_size = Path(results_file).stat().st_size
                    file_size_mb = file_size / (1024 * 1024)
                    logger.info(f"Results file size: {file_size_mb:.2f} MB")

            # Export batch metadata separately
            logger.info("Creating batch metadata...")
            metadata = {
                "total_simulations": len(self.all_results),
                "total_variations": len(
                    set(r["variation_id"] for r in self.all_results)
                ),
                "locations": list(set(r["location"] for r in self.all_results)),
                "seasons": list(set(r["season"] for r in self.all_results)),
                "controllers": list(set(r["controller"] for r in self.all_results)),
                "weeks_simulated": [1, 2, 3],
                "export_timestamp": self.timestamp,
                "config_file": str(self.config_loader.config_path),
                "simulation_type": "batch",
                "batch_config": {
                    "num_variations": self.batch_config.num_variations,
                    "random_seed": self.batch_config.random_seed,
                    "accuracy_range": self.batch_config.accuracy_range,
                    "latency_range": self.batch_config.latency_range,
                    "battery_capacity_range": self.batch_config.battery_capacity_range,
                    "charge_rate_range": self.batch_config.charge_rate_range,
                },
            }
            logger.debug(f"Batch metadata: {metadata}")

            metadata_filename = f"batch-run-{self.timestamp}-metadata.json"
            logger.info(f"Exporting metadata to {metadata_filename}")
            metadata_file = self.exporter.export_json(metadata, metadata_filename)
            if metadata_file:
                logger.info(f"Metadata exported to {metadata_file}")

                # Log file size
                if Path(metadata_file).exists():
                    file_size = Path(metadata_file).stat().st_size
                    file_size_kb = file_size / 1024
                    logger.info(f"Metadata file size: {file_size_kb:.2f} KB")

        except Exception as e:
            logger.error(f"Failed to export batch results: {e}")
            logger.exception("Full traceback for batch results export error")
            raise


def main():
    """Main entry point for batch simulation runner."""
    logger.info("=" * 80)
    logger.info("BATCH SIMULATION RUNNER - MAIN ENTRY POINT")
    logger.info("=" * 80)

    start_time = datetime.now()
    logger.info(f"Batch simulation runner start time: {start_time}")

    print("🚀 Starting Batch Simulation Runner...")
    logger.info("Starting Batch Simulation Runner...")

    try:
        print("📋 Loading configuration and setting up batch runner...")
        logger.info("Loading configuration and setting up batch runner...")
        # Create batch simulation runner
        runner = BatchSimulationRunner(
            config_path="config.jsonc",
            max_workers=100,  # Parallel execution with 100 workers
        )
        print("✓ Batch simulation runner initialized")
        logger.info("✓ Batch simulation runner initialized")

        print("🔄 Running batch simulations in parallel...")
        logger.info("Running batch simulations in parallel...")
        # Run batch simulations in parallel
        success = runner.run_batch_simulations(parallel=True)

        if success:
            print("✓ Batch simulations completed successfully")
            logger.info("✓ Batch simulations completed successfully")

            # Print summary statistics
            stats = runner.get_summary_stats()
            logger.info(f"Summary statistics: {stats}")

            print("📊 === Batch Simulation Summary ===")
            print(f"Total simulations: {stats['total_simulations']}")
            print(f"Overall completion rate: {stats['overall_completion_rate']:.2f}%")
            print(
                f"Overall clean energy usage: {stats['overall_clean_energy_percentage']:.2f}%"
            )
            print(f"Total energy consumed: {stats['total_energy_wh']:.2f} Wh")

            print("\n🎯 === Controller Performance ===")
            for controller, perf in stats["controller_performance"].items():
                print(f"{controller}:")
                print(f"  Simulations: {perf['count']}")
                print(f"  Avg completion rate: {perf['avg_completion_rate']:.2f}%")
                print(f"  Avg clean energy: {perf['avg_clean_energy_pct']:.2f}%")
                print(f"  Total energy: {perf['total_energy']:.2f} Wh")

            print("\n✅ All batch simulations completed successfully!")
            print(f"📁 Results exported to: {runner.exporter.output_dir}")

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"All batch simulations completed successfully in {duration:.2f} seconds"
            )
            logger.info(f"Results exported to: {runner.exporter.output_dir}")
            logger.info("=" * 80)
            logger.info("BATCH SIMULATION RUNNER COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

            return 0
        else:
            print("✗ Some batch simulations failed. Check logs for details.")
            logger.error("Some batch simulations failed. Check logs for details.")
            return 1

    except Exception as e:
        print(f"✗ Batch simulation runner failed: {e}")
        logger.error(f"✗ Batch simulation runner failed: {e}")
        import traceback

        print(f"Full error: {traceback.format_exc()}")
        logger.exception("Full traceback for batch simulation runner error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
