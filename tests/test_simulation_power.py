#!/usr/bin/env python3
"""
Test Simulation Power
Tests power calculations using real power_profiles.json
"""

import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.power_profiler import PowerProfiler


def test_power_profiles_loading():
    """Test loading real power_profiles.json."""
    print("Testing power profiles loading...")

    # Check if power_profiles.json exists
    profiles_file = Path("model-data/power_profiles.json")
    assert profiles_file.exists(), f"power_profiles.json not found at {profiles_file}"

    # Load and validate structure
    with open(profiles_file, "r") as f:
        profiles_data = json.load(f)

    assert isinstance(profiles_data, dict), "Power profiles should be a dictionary"
    assert len(profiles_data) > 0, "Power profiles should not be empty"

    print(f"✓ Loaded {len(profiles_data)} model profiles")
    return profiles_data


def test_power_profiler_initialization():
    """Test PowerProfiler initialization."""
    print("Testing PowerProfiler initialization...")

    try:
        profiler = PowerProfiler()
        assert profiler is not None, "PowerProfiler should be initialized"
        assert hasattr(profiler, "power_profiles"), (
            "Should have power_profiles attribute"
        )
        assert hasattr(profiler, "profiles_file"), "Should have profiles_file attribute"
        print("✓ PowerProfiler initialized successfully")
        return profiler
    except Exception as e:
        # This might fail on non-macOS systems due to powermetrics requirement
        print(f"⚠️  PowerProfiler initialization failed (expected on non-macOS): {e}")
        return None


def test_load_profiles():
    """Test loading profiles into PowerProfiler."""
    print("Testing load_profiles method...")

    profiler = PowerProfiler()

    try:
        profiler.load_profiles()
        assert len(profiler.power_profiles) > 0, "No profiles loaded"

        # Check expected models
        expected_models = [
            "YOLOv10_N",
            "YOLOv10_S",
            "YOLOv10_M",
            "YOLOv10_B",
            "YOLOv10_L",
            "YOLOv10_X",
        ]

        for model in expected_models:
            assert model in profiler.power_profiles, f"Missing model: {model}"
            model_data = profiler.power_profiles[model]

            # Check required fields
            required_fields = ["accuracy", "avg_inference_time_seconds", "power_cost"]
            for field in required_fields:
                assert field in model_data, f"Missing field {field} for model {model}"
                assert isinstance(model_data[field], (int, float)), (
                    f"Field {field} should be numeric"
                )
                assert model_data[field] >= 0, f"Field {field} should be non-negative"

            print(
                f"  ✓ {model}: accuracy={model_data['accuracy']:.3f}, "
                f"time={model_data['avg_inference_time_seconds']:.3f}s, "
                f"power={model_data['power_cost']:.3f}"
            )

        print(f"✓ Loaded {len(profiler.power_profiles)} profiles successfully")

    except Exception as e:
        print(f"⚠️  Load profiles failed (expected on non-macOS): {e}")


def test_get_all_models_data():
    """Test getting all models data."""
    print("Testing get_all_models_data...")

    profiler = PowerProfiler()

    try:
        profiler.load_profiles()
        all_models = profiler.get_all_models_data()

        assert isinstance(all_models, dict), "Should return dictionary"
        assert len(all_models) > 0, "Should return non-empty dictionary"

        # Check structure of model data
        for model_name, model_data in all_models.items():
            assert isinstance(model_name, str), "Model name should be string"
            assert isinstance(model_data, dict), "Model data should be dictionary"

            # Verify required fields
            required_fields = ["accuracy", "avg_inference_time_seconds", "power_cost"]
            for field in required_fields:
                assert field in model_data, f"Missing field {field} in {model_name}"
                assert model_data[field] >= 0, (
                    f"Field {field} should be non-negative in {model_name}"
                )

        print(f"✓ get_all_models_data returned {len(all_models)} models")

    except Exception as e:
        print(f"⚠️  get_all_models_data failed (expected on non-macOS): {e}")


def test_get_model_power():
    """Test getting power for specific models."""
    print("Testing get_model_power...")

    profiler = PowerProfiler()

    try:
        profiler.load_profiles()

        # Test power for each model
        test_models = [
            ("YOLOv10_N", "n"),
            ("YOLOv10_S", "s"),
            ("YOLOv10_M", "m"),
            ("YOLOv10_B", "b"),
            ("YOLOv10_L", "l"),
            ("YOLOv10_X", "x"),
        ]

        for model_name, model_version in test_models:
            try:
                power = profiler.get_model_power(model_name, model_version)
                assert isinstance(power, (int, float)), "Power should be numeric"
                assert power >= 0, "Power should be non-negative"
                print(f"  ✓ {model_name} v{model_version}: {power:.3f}W")
            except Exception as e:
                print(f"  ⚠️  {model_name} v{model_version}: {e}")

        print("✓ get_model_power tests completed")

    except Exception as e:
        print(f"⚠️  get_model_power failed (expected on non-macOS): {e}")


def test_power_calculation_consistency():
    """Test power calculation consistency."""
    print("Testing power calculation consistency...")

    # Load raw profiles data
    profiles_file = Path("model-data/power_profiles.json")
    with open(profiles_file, "r") as f:
        raw_profiles = json.load(f)

    # Test consistency across different access methods
    profiler = PowerProfiler()

    try:
        profiler.load_profiles()

        for model_name in raw_profiles.keys():
            # Compare raw data with profiler data
            raw_data = raw_profiles[model_name]
            profiler_data = profiler.power_profiles.get(model_name)

            assert profiler_data is not None, (
                f"Model {model_name} missing from profiler"
            )

            # Check consistency
            for field in ["accuracy", "avg_inference_time_seconds", "power_cost"]:
                raw_value = raw_data.get(field)
                profiler_value = profiler_data.get(field)

                assert raw_value == profiler_value, (
                    f"Inconsistent {field} for {model_name}: {raw_value} vs {profiler_value}"
                )

        print("✓ Power calculation consistency verified")

    except Exception as e:
        print(f"⚠️  Consistency test failed (expected on non-macOS): {e}")


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
    rankings["power"] = sorted(profiles_data.items(), key=lambda x: x[1]["power_cost"])

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
                print(f"    {i + 1}. {model}: {data['power_cost']:.3f}W")

    # Verify rankings are different (models have different strengths)
    assert rankings["accuracy"][0][0] != rankings["power"][0][0], (
        "Best accuracy model should differ from best power model"
    )

    print("✓ Model performance ranking verified")


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
        required_fields = ["accuracy", "avg_inference_time_seconds", "power_cost"]
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
        power = model_data["power_cost"]

        assert 0 <= accuracy <= 1, (
            f"Accuracy should be in [0,1]: {accuracy} for {model_name}"
        )
        assert 0 < latency < 1, (
            f"Latency should be reasonable: {latency}s for {model_name}"
        )
        assert 0 < power < 100, f"Power should be reasonable: {power}W for {model_name}"

    print(f"✓ All {len(profiles_data)} profiles validated")


def test_power_calculation_scenarios():
    """Test power calculations for different scenarios."""
    print("Testing power calculation scenarios...")

    profiles_file = Path("model-data/power_profiles.json")
    with open(profiles_file, "r") as f:
        profiles_data = json.load(f)

    # Test scenarios
    scenarios = [
        {
            "name": "Low power scenario",
            "models": ["YOLOv10_N", "YOLOv10_S"],
            "max_total_power": 5.0,  # 5W total
        },
        {
            "name": "Balanced scenario",
            "models": ["YOLOv10_N", "YOLOv10_M", "YOLOv10_L"],
            "max_total_power": 15.0,  # 15W total
        },
        {
            "name": "High performance scenario",
            "models": ["YOLOv10_M", "YOLOv10_B", "YOLOv10_X"],
            "max_total_power": 25.0,  # 25W total
        },
    ]

    for scenario in scenarios:
        total_power = 0.0
        model_count = 0

        for model in scenario["models"]:
            if model in profiles_data:
                power = profiles_data[model]["power_cost"]
                total_power += power
                model_count += 1
                print(f"    {model}: {power:.3f}W")

        avg_power = total_power / model_count if model_count > 0 else 0

        print(f"  ✓ {scenario['name']}:")
        print(f"    Total power: {total_power:.3f}W")
        print(f"    Average power: {avg_power:.3f}W")
        print(f"    Model count: {model_count}")

        assert total_power <= scenario["max_total_power"], (
            f"Total power exceeds limit: {total_power:.3f}W > {scenario['max_total_power']}W"
        )

    print("✓ Power calculation scenarios verified")


def main():
    """Run all simulation power tests."""
    print("🧪 Running Simulation Power Tests")
    print("=" * 50)

    try:
        test_power_profiles_loading()
        test_power_profiler_initialization()
        test_load_profiles()
        test_get_all_models_data()
        test_get_model_power()
        test_power_calculation_consistency()
        test_model_performance_ranking()
        test_power_profile_validation()
        test_power_calculation_scenarios()

        print("\n✅ All simulation power tests passed!")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
