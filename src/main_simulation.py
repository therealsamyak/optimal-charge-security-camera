"""Main entry point for the security camera simulation."""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.runner import SimulationRunner  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main():
    """Main simulation entry point."""
    logger.info("=" * 60)
    logger.info("Starting Optimal Charge Security Camera Simulation")
    logger.info("=" * 60)

    parser = argparse.ArgumentParser(description="Run security camera simulation")
    parser.add_argument(
        "--config", default="config.jsonc", help="Configuration file path"
    )
    parser.add_argument(
        "--output", default="simulation_results.csv", help="Output CSV file"
    )

    args = parser.parse_args()
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Output file: {args.output}")

    try:
        logger.info("Step 1: Loading configuration...")
        config = load_config(args.config)
        logger.info("✓ Configuration loaded successfully")
        logger.info(f"  Controller type: {config['simulation']['controller_type']}")
        logger.info(f"  Simulation date: {config['simulation']['date']}")
        logger.info(f"  Image quality: {config['simulation']['image_quality']}")

        logger.info("Step 2: Initializing simulation runner...")
        runner = SimulationRunner(config)
        logger.info("✓ Simulation runner initialized")

        logger.info("Step 3: Running 24-hour simulation...")
        results = runner.run_simulation()
        logger.info(f"✓ Simulation completed: {len(results)} time steps")

        logger.info("Step 4: Saving results to CSV...")
        runner.save_results(results, args.output)
        logger.info(f"✓ Results saved to {args.output}")

        logger.info("Step 5: Calculating metrics summary...")
        summary = runner.get_metrics_summary()
        logger.info("=" * 60)
        logger.info("Simulation Summary:")
        logger.info(f"  Total Inferences: {summary['total_inferences']}")
        logger.info(f"  Small Miss Rate: {summary['small_miss_rate']:.2f}%")
        logger.info(f"  Large Miss Rate: {summary['large_miss_rate']:.2f}%")
        logger.info(
            f"  Clean Energy Percentage: {summary['clean_energy_percentage']:.2f}%"
        )
        logger.info("=" * 60)
        logger.info("Simulation completed successfully!")

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Simulation failed with error: {e}")
        logger.error("=" * 60)
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
