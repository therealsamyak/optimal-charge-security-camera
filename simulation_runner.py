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

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.logging_config import setup_logging
from src.metrics_collector import CSVExporter
from src.simulation_runner_base import SimulationRunnerBase


class BasicSimulationRunner(SimulationRunnerBase):
    """Orchestrates execution of basic simulations (16 total)."""

    def __init__(self, config_path: str = "config.jsonc", max_workers: int = 4):
        super().__init__(config_path, max_workers)

        # Initialize exporter
        output_dir = self.config_loader.get_output_dir()
        self.exporter = CSVExporter(output_dir)

    def run_basic_simulations(self, parallel: bool = True) -> bool:
        """
        Run basic simulations (64 total: 4 locations × 4 seasons × 4 controllers, week 1 only).

        Args:
            parallel: Whether to run simulations in parallel

        Returns:
            True if all simulations succeeded, False if any failed
        """
        # Get configuration
        locations = self.config_loader.get_locations()
        seasons = self.config_loader.get_seasons()
        controllers = self.config_loader.get_controllers()

        # Generate simulations for week 1 only
        simulations = self._generate_simulation_list(
            locations=locations,
            seasons=seasons,
            controllers=controllers,
            weeks=[1],  # Week 1 only
        )

        successful_results = []

        if parallel:
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

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_sim):
                    sim = future_to_sim[future]
                    try:
                        result = future.result()
                        if result:
                            successful_results.append(result)
                    except Exception as e:
                        self.logger.error(
                            f"Simulation {sim['simulation_id']} raised exception: {e}"
                        )
                        # Failure already recorded in _run_single_simulation
        else:
            # Run simulations sequentially
            for sim in simulations:
                result = self._run_single_simulation(sim)
                if result:
                    successful_results.append(result)

        # Check if any simulations failed
        if len(self.failed_simulations) > 0:
            self.logger.error(f"{len(self.failed_simulations)} simulations failed")
            self.logger.error(
                "Terminating due to failures - no CSV output will be generated"
            )
            return False

        # Store successful results
        self.all_results = successful_results

        # Export results
        self._export_results()

        self.logger.info(
            f"All {len(successful_results)} basic simulations completed successfully"
        )
        return True

    def _export_results(self):
        """Export simulation results to CSV files."""
        if not self.all_results:
            self.logger.warning("No results to export")
            return

        try:
            # Export aggregated results
            aggregated_file = self.exporter.export_aggregated_results(self.all_results)
            if aggregated_file:
                self.logger.info(f"Aggregated results exported to {aggregated_file}")

            # Export metadata
            metadata = {
                "total_simulations": len(self.all_results),
                "locations": list(set(r["location"] for r in self.all_results)),
                "seasons": list(set(r["season"] for r in self.all_results)),
                "controllers": list(set(r["controller"] for r in self.all_results)),
                "export_timestamp": "Basic Simulation Complete",
                "config_file": str(self.config_loader.config_path),
                "simulation_type": "basic",
                "weeks_simulated": [1],
            }

            metadata_file = self.exporter.export_json(
                metadata, "basic_simulation_metadata.json"
            )
            if metadata_file:
                self.logger.info(f"Metadata exported to {metadata_file}")

        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            raise


def main():
    """Main entry point for basic simulation runner."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    print("🚀 Starting Basic Simulation Runner...")
    
    try:
        print("📋 Loading configuration...")
        # Create basic simulation runner
        runner = BasicSimulationRunner(
            config_path="config.jsonc",
            max_workers=1,  # Sequential execution only
        )
        print("✓ Configuration loaded successfully")

        print("🔄 Running simulations sequentially...")
        print(f"Total simulations to run: {len(runner.simulation_configs)}")
        # Run simulations sequentially
        success = runner.run_basic_simulations(parallel=False)

        if success:
            print("✓ Simulations completed successfully")
            # Print summary statistics
            stats = runner.get_summary_stats()
            print("📊 === Basic Simulation Summary ===")
            print(f"Total simulations: {stats['total_simulations']}")
            print(
                f"Overall completion rate: {stats['overall_completion_rate']:.2f}%"
            )
            print(
                f"Overall clean energy usage: {stats['overall_clean_energy_percentage']:.2f}%"
            )
            print(f"Total energy consumed: {stats['total_energy_wh']:.2f} Wh")

            print("\n🎯 === Controller Performance ===")
            for controller, perf in stats["controller_performance"].items():
                print(f"{controller}:")
                print(f"  Simulations: {perf['count']}")
                print(
                    f"  Avg completion rate: {perf['avg_completion_rate']:.2f}%"
                )
                print(f"  Avg clean energy: {perf['avg_clean_energy_pct']:.2f}%")
                print(f"  Total energy: {perf['total_energy']:.2f} Wh")

            print("\n✅ All basic simulations completed successfully!")
            print(f"📁 Results exported to: {runner.exporter.output_dir}")
            return 0
        else:
            print("✗ Some simulations failed. Check logs for details.")
            return 1

    except Exception as e:
        print(f"✗ Basic simulation runner failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
