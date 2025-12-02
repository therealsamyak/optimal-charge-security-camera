#!/usr/bin/env python3
"""
Quick integration test to verify both simulation runners work without running full simulations.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_basic_runner_initialization():
    """Test basic simulation runner can be initialized."""
    import simulation_runner

    runner = simulation_runner.BasicSimulationRunner(
        config_path="config.jsonc", max_workers=1
    )

    # Test configuration loading
    assert runner.config_loader.validate_config()

    # Test simulation list generation
    locations = runner.config_loader.get_locations()
    seasons = runner.config_loader.get_seasons()
    controllers = runner.config_loader.get_controllers()

    simulations = runner._generate_simulation_list(
        locations=locations,
        seasons=seasons,
        controllers=controllers,
        weeks=[1],  # Week 1 only
    )

    expected_count = len(locations) * len(seasons) * len(controllers) * 1
    assert len(simulations) == expected_count

    print("✓ Basic runner initialized successfully")
    print(f"  - Will run {len(simulations)} simulations")
    print(f"  - Locations: {locations}")
    print(f"  - Seasons: {seasons}")
    print(f"  - Controllers: {controllers}")


def test_batch_runner_initialization():
    """Test batch simulation runner can be initialized."""
    import batch_simulation

    runner = batch_simulation.BatchSimulationRunner(
        config_path="config.jsonc", max_workers=1
    )

    # Test batch configuration loading
    batch_config = runner.batch_config
    assert batch_config.num_variations > 0

    # Test parameter variations generation
    variations = runner.generate_parameter_variations(batch_config)
    assert len(variations) == batch_config.num_variations

    # Test simulation list generation
    locations = runner.config_loader.get_locations()
    seasons = runner.config_loader.get_seasons()
    controllers = runner.config_loader.get_controllers()

    simulations = runner._generate_simulation_list(
        locations=locations,
        seasons=seasons,
        controllers=controllers,
        weeks=[1, 2, 3],  # All 3 weeks
    )

    expected_count = len(locations) * len(seasons) * len(controllers) * 3
    assert len(simulations) == expected_count

    total_simulations = len(variations) * len(simulations)

    print("✓ Batch runner initialized successfully")
    print(f"  - Generated {len(variations)} parameter variations")
    print(f"  - Each variation runs {len(simulations)} simulations")
    print(f"  - Total simulations: {total_simulations}")
    print(f"  - Random seed: {batch_config.random_seed}")

    # Show first variation as example
    if variations:
        var = variations[0]
        print("  - Example variation:")
        print(f"    Accuracy: {var['accuracy_requirement']:.1f}%")
        print(f"    Latency: {var['latency_requirement']:.1f}s")
        print(f"    Battery: {var['battery_capacity_wh']:.1f}Wh")
        print(f"    Charge Rate: {var['charge_rate_watts']:.0f}W")


def main():
    """Run integration tests."""
    print("=== Simulation Runner Integration Tests ===\n")

    basic_success = test_basic_runner_initialization()
    print()
    batch_success = test_batch_runner_initialization()
    print()

    if basic_success and batch_success:
        print("✓ All integration tests passed!")
        print("\nBoth simulation runners are ready to use:")
        print("  - python simulation_runner.py  # Runs 64 basic simulations")
        print("  - python batch_simulation.py  # Runs 1920+ batch simulations")
        return 0
    else:
        print("✗ Some integration tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
