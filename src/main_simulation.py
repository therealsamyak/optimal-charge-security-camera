"""Main entry point for the security camera simulation."""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation.runner import SimulationRunner
from src.utils.config import load_config


def main():
    """Main simulation entry point."""
    parser = argparse.ArgumentParser(description="Run security camera simulation")
    parser.add_argument(
        "--config", default="config.jsonc", help="Configuration file path"
    )
    parser.add_argument(
        "--output", default="simulation_results.csv", help="Output CSV file"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Run simulation
        runner = SimulationRunner(config)
        results = runner.run_simulation()

        # Save results
        runner.save_results(results, args.output)

        # Print metrics summary
        summary = runner.get_metrics_summary()
        logger.info("Simulation Summary:")
        logger.info(f"  Total Inferences: {summary['total_inferences']}")
        logger.info(f"  Small Miss Rate: {summary['small_miss_rate']:.2f}%")
        logger.info(f"  Large Miss Rate: {summary['large_miss_rate']:.2f}%")
        logger.info(
            f"  Clean Energy Percentage: {summary['clean_energy_percentage']:.2f}%"
        )

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
