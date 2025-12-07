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

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.logging_config import setup_logging
from src.metrics_collector import CSVExporter
from src.simulation_runner_base import SimulationRunnerBase


class BatchSimulationRunner(SimulationRunnerBase):
    """Orchestrates execution of batch simulations with parameter variations."""

    def __init__(self, config_path: str = "config.jsonc", max_workers: int = 100):
        super().__init__(config_path, max_workers)

        # Get batch configuration
        self.batch_config = self.config_loader.get_batch_config()

        # Initialize exporter
        output_dir = self.config_loader.get_output_dir()
        self.exporter = CSVExporter(output_dir)

        # Generate timestamp for filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_batch_simulations(self, parallel: bool = True) -> bool:
        """
        Run batch simulations with parameter variations.

        Args:
            parallel: Whether to run simulations in parallel

        Returns:
            True if all simulations succeeded, False if any failed
        """
        try:
            # Generate parameter variations
            variations = self.generate_parameter_variations(self.batch_config)

            # Get configuration for simulation matrix
            locations = self.config_loader.get_locations()
            seasons = self.config_loader.get_seasons()
            controllers = self.config_loader.get_controllers()

            # Generate base simulation list (full matrix: 4 locations × 4 seasons × 4 controllers × 3 weeks = 192)
            base_simulations = self._generate_simulation_list(
                locations=locations,
                seasons=seasons,
                controllers=controllers,
                weeks=[1, 2, 3],  # All 3 weeks
            )

            self.logger.info(
                f"Starting batch simulation with {len(variations)} parameter variations"
            )
            self.logger.info(
                f"Each variation will run {len(base_simulations)} simulations"
            )
            self.logger.info(
                f"Total simulations to run: {len(variations) * len(base_simulations)}"
            )

            successful_results = []

            # Run simulations for each parameter variation
            for variation in variations:
                self.logger.info(
                    f"Running variation {variation['variation_id']}/{len(variations)}"
                )

                # Create simulation config with parameter overrides
                base_config = self.config_loader.get_simulation_config()
                sim_config = self._create_simulation_config_with_overrides(
                    base_config=base_config,
                    accuracy_override=variation["accuracy_requirement"],
                    latency_override=variation["latency_requirement"],
                    battery_capacity_override=variation["battery_capacity_wh"],
                    charge_rate_override=variation["charge_rate_watts"],
                )

                # Run all simulations for this variation
                variation_results = self._run_variation_simulations(
                    base_simulations=base_simulations,
                    sim_config=sim_config,
                    variation=variation,
                    parallel=parallel,
                )

                successful_results.extend(variation_results)

            # Check if any simulations failed
            if len(self.failed_simulations) > 0:
                self.logger.error(f"{len(self.failed_simulations)} simulations failed")
                self.logger.error(
                    "TERMINATING DUE TO FAILURES - NO OUTPUT WILL BE GENERATED"
                )

                # Print clear error information
                for failure in self.failed_simulations[:5]:  # Show first 5 failures
                    self.logger.error(
                        f"FAILED: {failure.get('simulation_id', 'unknown')} - {failure.get('error', 'unknown error')}"
                    )

                if len(self.failed_simulations) > 5:
                    self.logger.error(
                        f"... and {len(self.failed_simulations) - 5} more failures"
                    )

                return False

            # Store successful results
            self.all_results = successful_results

            # Export results
            self._export_batch_results()

            self.logger.info(
                f"All {len(successful_results)} batch simulations completed successfully"
            )
            return True

        except Exception as e:
            self.logger.error(f"Batch simulation failed with exception: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return False

    def _run_variation_simulations(
        self, base_simulations: list, sim_config, variation: dict, parallel: bool = True
    ) -> list:
        """Run all simulations for a single parameter variation."""
        successful_results = []
        variation_start_time = datetime.now()

        self.logger.info(
            f"[Variation {variation['variation_id']}] Starting {len(base_simulations)} simulations"
        )
        self.logger.debug(
            f"[Variation {variation['variation_id']}] Parameters: {variation}"
        )

        if parallel:
            import concurrent.futures

            # Run simulations in parallel
            self.logger.debug(
                f"[Variation {variation['variation_id']}] Using parallel execution with {self.max_workers} workers"
            )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # Submit all simulations with variation config
                future_to_sim = {
                    executor.submit(self._run_single_simulation, sim, sim_config): sim
                    for sim in base_simulations
                }

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
                        else:
                            error_count += 1

                        if completed_count % 10 == 0:  # Log every 10 completions
                            progress = (completed_count / len(base_simulations)) * 100
                            print(
                                f"[Variation {variation['variation_id']}] Progress: {completed_count}/{len(base_simulations)} ({progress:.1f}%) - Errors: {error_count}"
                            )
                            self.logger.info(
                                f"[Variation {variation['variation_id']}] Progress: {progress:.1f}% ({completed_count}/{len(base_simulations)})"
                            )

                    except Exception as e:
                        self.logger.error(
                            f"[Variation {variation['variation_id']}] Simulation {sim['simulation_id']} raised exception: {e}"
                        )
                        # Failure already recorded in _run_single_simulation
        else:
            # Run simulations sequentially
            self.logger.debug(
                f"[Variation {variation['variation_id']}] Using sequential execution"
            )
            for i, sim in enumerate(base_simulations):
                if (i + 1) % 10 == 0:  # Log every 10 simulations
                    progress = ((i + 1) / len(base_simulations)) * 100
                    self.logger.info(
                        f"[Variation {variation['variation_id']}] Progress: {progress:.1f}% ({i + 1}/{len(base_simulations)})"
                    )

                result = self._run_single_simulation(sim, sim_config)
                if result:
                    # Add variation metadata
                    result["variation_id"] = variation["variation_id"]
                    successful_results.append(result)

        variation_time = (datetime.now() - variation_start_time).total_seconds()
        self.logger.info(
            f"[Variation {variation['variation_id']}] Completed in {variation_time:.2f}s, "
            f"{len(successful_results)} successful, {len(base_simulations) - len(successful_results)} failed"
        )

        return successful_results

    def _export_batch_results(self):
        """Export batch simulation results to CSV files with timestamps."""
        if not self.all_results:
            self.logger.warning("No results to export")
            return

        try:
            # Export aggregated results with batch prefix
            aggregated_filename = f"batch-run-{self.timestamp}-aggregated-results.csv"
            aggregated_file = self.exporter.export_aggregated_results(
                self.all_results, filename=aggregated_filename
            )
            if aggregated_file:
                self.logger.info(f"Aggregated results exported to {aggregated_file}")

            # Export detailed results with all parameters if configured
            if self.batch_config.output_detailed_csv:
                detailed_filename = f"batch-run-{self.timestamp}-detailed-results.csv"
                detailed_file = self.exporter.export_detailed_results(
                    self.all_results, filename=detailed_filename
                )
                if detailed_file:
                    self.logger.info(f"Detailed results exported to {detailed_file}")

            # Export batch metadata
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

            metadata_filename = f"batch-run-{self.timestamp}-metadata.json"
            metadata_file = self.exporter.export_json(metadata, metadata_filename)
            if metadata_file:
                self.logger.info(f"Metadata exported to {metadata_file}")

        except Exception as e:
            self.logger.error(f"Failed to export batch results: {e}")
            raise


def main():
    """Main entry point for batch simulation runner."""
    # Setup logging
    setup_logging()
    logging.getLogger(__name__)

    print("🚀 Starting Batch Simulation Runner...")

    try:
        print("📋 Loading configuration and setting up batch runner...")
        # Create batch simulation runner
        runner = BatchSimulationRunner(
            config_path="config.jsonc",
            max_workers=100,  # Parallel execution with 100 workers
        )
        print("✓ Batch simulation runner initialized")

        print("🔄 Running batch simulations in parallel...")
        # Run batch simulations in parallel
        success = runner.run_batch_simulations(parallel=True)

        if success:
            print("✓ Batch simulations completed successfully")
            # Print summary statistics
            stats = runner.get_summary_stats()
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
            return 0
        else:
            print("✗ Some batch simulations failed. Check logs for details.")
            return 1

    except Exception as e:
        print(f"✗ Batch simulation runner failed: {e}")
        import traceback

        print(f"Full error: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
