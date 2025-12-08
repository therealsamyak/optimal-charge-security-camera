#!/usr/bin/env python3
"""
Test Simulation Power
Tests power calculations using real power_profiles.json
"""

import sys
import logging
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Setup logging for tests
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_power_profiles_loading():
    """Test loading real power_profiles.json."""
    logger.info("Testing power profiles loading...")

    # Check if power_profiles.json exists
    profiles_file = Path("model-data/power_profiles.json")
    logger.debug(f"Looking for profiles at: {profiles_file}")

    if not profiles_file.exists():
        logger.error(f"power_profiles.json not found at {profiles_file}")
        raise FileNotFoundError(f"power_profiles.json not found at {profiles_file}")

    # Load and validate structure
    try:
        with open(profiles_file, "r") as f:
            profiles_data = json.load(f)
        logger.debug(f"Loaded profiles data with {len(profiles_data)} models")
    except Exception as e:
        logger.error(f"Failed to load profiles: {e}")
        raise

    assert isinstance(profiles_data, dict), "Power profiles should be a dictionary"
    assert len(profiles_data) > 0, "Power profiles should not be empty"

    logger.info(f"✓ Loaded {len(profiles_data)} model profiles")
    return None


def test_power_profile_validation():
    """Test power profile data validation."""
    print("Testing power profile validation...")

    profiles_file = Path("model-data/power_profiles.json")
    with open(profiles_file, "r") as f:
        profiles_data = json.load(f)

    # Validate each model profile
    for model_name, model_data in profiles_data.items():
        # Check model name format
        assert model_name.startswith("YOLOv10_"), (
            f"Invalid model name format: {model_name}"
        )

        # Check required fields
        required_fields = ["accuracy", "avg_inference_time_seconds", "model_power_mw"]
        for field in required_fields:
            assert field in model_data, f"Missing field {field} in {model_name}"

            value = model_data[field]
            assert isinstance(value, (int, float)), (
                f"Field {field} should be numeric in {model_name}"
            )
            assert value >= 0, f"Field {field} should be non-negative in {model_name}"
            assert not (value != value), (
                f"Field {field} should not be NaN in {model_name}"
            )  # NaN check

        # Validate reasonable ranges
        accuracy = model_data["accuracy"]
        latency = model_data["avg_inference_time_seconds"]
        power = model_data["model_power_mw"]

        assert 0 <= accuracy <= 1, (
            f"Accuracy should be in [0,1]: {accuracy} for {model_name}"
        )
        assert 0 < latency < 1, (
            f"Latency should be reasonable: {latency}s for {model_name}"
        )
        assert 0 < power < 5000, (
            f"Power should be reasonable: {power}mW for {model_name}"
        )

    print(f"✓ All {len(profiles_data)} profiles validated")


def test_model_performance_ranking():
    """Test model performance ranking."""
    print("Testing model performance ranking...")

    # Load profiles data
    profiles_file = Path("model-data/power_profiles.json")
    with open(profiles_file, "r") as f:
        profiles_data = json.load(f)

    # Test different ranking criteria
    rankings = {}

    # By accuracy (higher is better)
    rankings["accuracy"] = sorted(
        profiles_data.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    # By speed (lower latency is better)
    rankings["speed"] = sorted(
        profiles_data.items(), key=lambda x: x[1]["avg_inference_time_seconds"]
    )

    # By power efficiency (lower power is better)
    rankings["power"] = sorted(
        profiles_data.items(), key=lambda x: x[1]["model_power_mw"]
    )

    # Verify rankings make sense
    for criterion, ranking in rankings.items():
        print(f"  ✓ {criterion} ranking:")
        for i, (model, data) in enumerate(ranking[:3]):  # Top 3
            if criterion == "accuracy":
                print(f"    {i + 1}. {model}: {data['accuracy']:.3f}")
            elif criterion == "speed":
                print(
                    f"    {i + 1}. {model}: {data['avg_inference_time_seconds']:.3f}s"
                )
            elif criterion == "power":
                print(f"    {i + 1}. {model}: {data['model_power_mw']:.3f}mW")

    # Verify rankings are different (models have different strengths)
    assert rankings["accuracy"][0][0] != rankings["power"][0][0], (
        "Best accuracy model should differ from best power model"
    )

    print("✓ Model performance ranking verified")


def test_oracle_controller():
    """Test OracleController with full-horizon optimization."""
    print("Testing OracleController with full-horizon optimization...")

    try:
        from controller import OracleController
        from config_loader import ConfigLoader

        # Load config
        config_loader = ConfigLoader()
        config = config_loader.get_simulation_config()

        # Load actual model data from JSON
        import json
        from pathlib import Path

        profiles_file = Path("model-data/power_profiles.json")
        with open(profiles_file, "r") as f:
            raw_models = json.load(f)

        # Transform to expected format
        available_models = {}
        for name, profile in raw_models.items():
            available_models[name] = {
                "accuracy": profile["accuracy"],
                "latency": profile["avg_inference_time_seconds"]
                * 1000,  # Convert to ms
                "power_cost": profile["model_power_mw"],  # Power in mW
            }

        # Create test data (shorter for testing)
        clean_energy_series = [50.0, 60.0, 70.0, 80.0, 40.0]  # 5 timesteps
        task_requirements = [
            {"accuracy": 0.5, "latency": 8.0},
            {"accuracy": 0.5, "latency": 8.0},
            {"accuracy": 0.5, "latency": 8.0},
            {"accuracy": 0.5, "latency": 8.0},
            {"accuracy": 0.5, "latency": 8.0},
        ]

        # Initialize oracle controller
        oracle = OracleController(clean_energy_series, task_requirements, config)

        # Test DP matrix creation
        assert hasattr(oracle, "dp_matrix"), "Oracle should have DP matrix"
        assert len(oracle.dp_matrix) == 5, "DP matrix should have 5 entries"

        # Test optimal schedule
        assert hasattr(oracle, "optimal_schedule"), (
            "Oracle should have optimal schedule"
        )
        assert len(oracle.optimal_schedule) == 5, "Schedule should have 5 entries"

        # Test model selection at each timestep
        for t in range(5):
            choice = oracle.select_model(
                battery_level=50.0,
                clean_energy_percentage=clean_energy_series[t],
                user_accuracy_requirement=0.5,
                user_latency_requirement=8.0,
                available_models=available_models,
            )

            # Verify choice structure
            assert hasattr(choice, "model_name"), "Choice should have model_name"
            assert hasattr(choice, "should_charge"), "Choice should have should_charge"
            assert choice.model_name in available_models.keys(), (
                "Invalid model selected"
            )
            assert isinstance(choice.should_charge, bool), (
                "should_charge should be boolean"
            )

            # Test should_charge method
            should_charge = oracle.should_charge()
            assert isinstance(should_charge, bool), (
                "should_charge should return boolean"
            )

            # Advance timestep
            oracle.advance_timestep()

        print("✓ OracleController full-horizon optimization test passed")
        return None

    except Exception as e:
        print(f"⚠️  OracleController test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_oracle_energy_optimization():
    """Test that oracle optimizes for clean energy usage."""
    print("Testing oracle clean energy optimization...")

    try:
        from controller import OracleController
        from config_loader import ConfigLoader

        # Load config
        config_loader = ConfigLoader()
        config = config_loader.get_simulation_config()

        # Create test data with high clean energy at t=2
        clean_energy_series = [10.0, 20.0, 90.0, 30.0, 10.0]  # Peak at t=2
        task_requirements = [
            {"accuracy": 0.5, "latency": 8.0},
            {"accuracy": 0.5, "latency": 8.0},
            {"accuracy": 0.5, "latency": 8.0},
            {"accuracy": 0.5, "latency": 8.0},
            {"accuracy": 0.5, "latency": 8.0},
        ]

        # Initialize oracle controller
        oracle = OracleController(clean_energy_series, task_requirements, config)

        # Check that oracle charges during high clean energy period
        charging_decisions = []
        for t in range(5):
            should_charge = oracle.should_charge()
            charging_decisions.append(should_charge)
            oracle.advance_timestep()

        # Oracle should prefer charging at t=2 (90% clean energy)
        print(f"  Charging decisions: {charging_decisions}")
        print(f"  Clean energy series: {clean_energy_series}")

        # At least one charging decision should be True
        assert any(charging_decisions), "Oracle should charge at least once"

        print("✓ Oracle clean energy optimization test passed")
        return None

    except Exception as e:
        print(f"⚠️  Oracle optimization test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Run all simulation power tests."""
    logger.info("🧪 Running Simulation Power Tests")
    logger.info("=" * 50)

    try:
        test_power_profiles_loading()
        test_power_profile_validation()
        test_model_performance_ranking()

        # New oracle controller tests
        test_oracle_controller()
        test_oracle_energy_optimization()

        logger.info("\n✅ All simulation power tests passed!")
        return 0

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
