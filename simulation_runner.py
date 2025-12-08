#!/usr/bin/env python3
"""
Basic Security Camera Simulation Runner

This file runs a simplified simulation with 64 total simulations:
- 4 locations × 4 seasons × 4 controllers (week 1 only)
- Uses default accuracy/latency values from config.jsonc
- All simulation metrics as defined in config.jsonc

Usage:
    python simulation_runner.py [--parallel] [--workers N] [--config config.jsonc]
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.logging_config import setup_logging, get_logger
from src.metrics_collector import JSONExporter
from src.simulation_runner_base import SimulationRunnerBase

# Initialize logging
logger = setup_logging()


class BasicSimulationRunner(SimulationRunnerBase):
    """Orchestrates execution of basic simulations (16 total)."""

    def __init__(self, config_path: str = "config.jsonc", max_workers: int = 100):
        logger.info("Initializing BasicSimulationRunner")
        logger.debug(f"Config path: {config_path}, Max workers: {max_workers}")

        super().__init__(config_path, max_workers)

        # Initialize exporter
        output_dir = self.config_loader.get_output_dir()
        self.exporter = JSONExporter(output_dir)

        logger.info(f"BasicSimulationRunner initialized")
        logger.info(f"Output directory: {output_dir}")
        logger.debug(f"Exporter: {self.exporter}")

    def run_basic_simulations(self, parallel: bool = True) -> bool:
        """
        Run basic simulations (64 total: 4 locations × 4 seasons × 4 controllers, week 1 only).

        Args:
            parallel: Whether to run simulations in parallel

        Returns:
            True if all simulations succeeded, False if any failed
        """
        logger.info("=" * 80)
        logger.info("STARTING BASIC SIMULATION RUN")
        logger.info("=" * 80)

        start_time = datetime.now()
        logger.info(f"Basic simulation start time: {start_time}")
        logger.info(f"Parallel execution: {parallel}")

        # Get configuration
        locations = self.config_loader.get_locations()
        seasons = self.config_loader.get_seasons()
        controllers = self.config_loader.get_controllers()

        logger.info(f"Configuration loaded:")
        logger.info(f"  Locations: {locations}")
        logger.info(f"  Seasons: {seasons}")
        logger.info(f"  Controllers: {controllers}")

        # Generate simulations for week 1 only
        simulations = self._generate_simulation_list(
            locations=locations,
            seasons=seasons,
            controllers=controllers,
            weeks=[1],  # Week 1 only
        )

        logger.info(f"Generated {len(simulations)} basic simulations")
        logger.debug(
            f"Simulation IDs: {[sim['simulation_id'] for sim in simulations[:5]]}..."
        )

        successful_results = []

        if parallel:
            logger.info(
                f"Running {len(simulations)} simulations in parallel with {self.max_workers} workers"
            )
            import concurrent.futures

            # Run simulations in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # Submit all simulations
                future_to_sim = {
                    executor.submit(self._run_single_simulation, sim): sim
                    for sim in simulations
                }
                logger.info(
                    f"Submitted all {len(simulations)} simulations to thread pool"
                )

                completed_count = 0
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_sim):
                    sim = future_to_sim[future]
                    completed_count += 1

                    if completed_count % 10 == 0:
                        progress = (completed_count / len(simulations)) * 100
                        logger.info(
                            f"Progress: {completed_count}/{len(simulations)} ({progress:.1f}%)"
                        )

                    try:
                        result = future.result()
                        if result:
                            successful_results.append(result)
                            logger.debug(
                                f"✓ Simulation {sim['simulation_id']} completed successfully"
                            )
                        else:
                            logger.warning(
                                f"✗ Simulation {sim['simulation_id']} returned no result"
                            )
                    except Exception as e:
                        logger.error(
                            f"Simulation {sim['simulation_id']} raised exception: {e}"
                        )
                        logger.exception(
                            f"Full traceback for simulation {sim['simulation_id']}"
                        )
                        # Failure already recorded in _run_single_simulation
        else:
            logger.info(f"Running {len(simulations)} simulations sequentially")
            # Run simulations sequentially
            for i, sim in enumerate(simulations):
                if (i + 1) % 5 == 0:
                    progress = ((i + 1) / len(simulations)) * 100
                    logger.info(
                        f"Progress: {i + 1}/{len(simulations)} ({progress:.1f}%)"
                    )

                result = self._run_single_simulation(sim)
                if result:
                    successful_results.append(result)
                    logger.debug(
                        f"✓ Simulation {sim['simulation_id']} completed successfully"
                    )
                else:
                    logger.warning(
                        f"✗ Simulation {sim['simulation_id']} returned no result"
                    )

        # Check if any simulations failed
        if len(self.failed_simulations) > 0:
            logger.error(f"{len(self.failed_simulations)} simulations failed")
            logger.error(
                "Terminating due to failures - no CSV output will be generated"
            )
            for failure in self.failed_simulations[:5]:  # Show first 5 failures
                logger.error(f"Failed simulation: {failure}")
            return False

        # Store successful results
        self.all_results = successful_results
        logger.info(f"Collected {len(successful_results)} successful results")

        # Export results
        logger.info("Exporting simulation results...")
        export_success = self._export_results()
        if not export_success:
            logger.error("Failed to export results")
            return False

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"All {len(successful_results)} basic simulations completed successfully in {duration:.2f} seconds"
        )
        return True

    def _export_results(self):
        """Export simulation results to JSON files."""
        logger.info("Starting results export process")

        if not self.all_results:
            logger.warning("No results to export")
            return False

        try:
            # Export results using JSONExporter
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"basic_simulation_{timestamp}.json"

            logger.info(f"Export filename: {filename}")
            logger.info(f"Number of simulations to export: {len(self.all_results)}")

            exported_file = self.exporter.export_results(
                all_simulations=self.all_results,
                aggregated_data=[],  # Basic simulation doesn't use aggregated data
                filename=filename,
            )

            if exported_file:
                logger.info(f"Results exported to {exported_file}")
                logger.info("✓ Basic simulation results exported successfully")

                # Log file size
                if Path(exported_file).exists():
                    file_size = Path(exported_file).stat().st_size
                    file_size_mb = file_size / (1024 * 1024)
                    logger.info(f"Exported file size: {file_size_mb:.2f} MB")

                return True
            else:
                logger.error("Failed to export results")
                return False

        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            logger.exception("Full traceback for results export error")
            return False


def main():
    """Main entry point for basic simulation runner."""
    logger.info("=" * 80)
    logger.info("BASIC SIMULATION RUNNER - MAIN ENTRY POINT")
    logger.info("=" * 80)

    start_time = datetime.now()
    logger.info(f"Basic simulation runner start time: {start_time}")

    print("🚀 Starting Basic Simulation Runner...")
    logger.info("Starting Basic Simulation Runner...")

    try:
        print("📋 Loading configuration...")
        logger.info("Loading configuration...")
        # Create basic simulation runner
        runner = BasicSimulationRunner(
            config_path="config.jsonc",
            max_workers=1,  # Sequential execution only
        )
        print("✓ Configuration loaded successfully")
        logger.info("✓ Configuration loaded successfully")

        print("🔄 Running simulations sequentially...")
        logger.info("Running simulations sequentially...")
        # Run simulations sequentially
        success = runner.run_basic_simulations(parallel=False)

        if success:
            print("✓ Simulations completed successfully")
            logger.info("✓ Simulations completed successfully")

            # Print summary statistics
            stats = runner.get_summary_stats()
            logger.info(f"Summary statistics: {stats}")

            print("📊 === Basic Simulation Summary ===")
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

            print("\n✅ All basic simulations completed successfully!")
            print(f"📁 Results exported to: {runner.exporter.output_dir}")

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"All basic simulations completed successfully in {duration:.2f} seconds"
            )
            logger.info(f"Results exported to: {runner.exporter.output_dir}")
            logger.info("=" * 80)
            logger.info("BASIC SIMULATION RUNNER COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

            return 0
        else:
            print("✗ Some simulations failed. Check logs for details.")
            logger.error("Some simulations failed. Check logs for details.")
            return 1

    except Exception as e:
        print(f"✗ Basic simulation runner failed: {e}")
        logger.error(f"✗ Basic simulation runner failed: {e}")
        import traceback

        print(f"Full error: {traceback.format_exc()}")
        logger.exception("Full traceback for basic simulation runner error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
