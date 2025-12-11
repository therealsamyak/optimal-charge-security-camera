#!/usr/bin/env python3
"""
Test script to verify preprocess_data.py functionality.
Tests with specific data points to ensure correct processing.
"""

import sys

# Add current directory to path for imports
sys.path.append(".")
from preprocess_data import process_timestep_to_training_sample


def test_preprocess_data():
    """Test preprocess_data.py with sample data."""
    print("Testing preprocess_data.py functionality...")

    # Create sample metadata structure like in real files
    sample_metadata = {
        "metadata": {
            "location": "CA",
            "horizon": 288,
            "total_leaves_explored": 120,
            "runtime_seconds": 179.71,
            "naive_runtime_seconds": 179.68,
            "timestamp": "2025-12-10T15:48:35.594155",
            "safety_limits_hit": False,
            "parallel_mode": True,
            "energy_data_location": "CA",
            "energy_data_date": "2024-11-30",
            "user_accuracy_requirement": 90.5,
            "user_latency_requirement": 0.01,
            "duration_days": 1,
            "task_interval_seconds": 300,
            "battery_capacity_wh": 1.211,
            "charge_rate_hours": 2.45,
        },
        "results": {
            "top_success": [
                {
                    "action_sequence": [
                        {
                            "timestep": 0,
                            "model": "YOLOv10_L",
                            "charged": False,
                            "outcome": "success",
                            "battery_before": 1.211,
                            "battery_after": 1.206313408252127,
                            "clean_energy_before": 0.0,
                            "clean_energy_after": 0.0,
                            "dirty_energy_before": 0.0,
                            "dirty_energy_after": 0.0,
                            "charge_energy": 0.0,
                            "model_energy": 0.004686591747873029,
                            "clean_energy_pct": 21.29,
                        },
                        {
                            "timestep": 1,
                            "model": "YOLOv10_B",
                            "charged": True,
                            "outcome": "success",
                            "battery_before": 1.206313408252127,
                            "battery_after": 1.209407301383274,
                            "clean_energy_before": 0.0,
                            "clean_energy_after": 0.001995550766244372,
                            "dirty_energy_before": 0.0,
                            "dirty_energy_after": 0.0073776327295018585,
                            "charge_energy": 0.009373183495746229,
                            "model_energy": 0.0038880972742090622,
                            "clean_energy_pct": 21.29,
                        },
                    ]
                }
            ],
            "top_success_small_miss": [
                {
                    "action_sequence": [
                        {
                            "timestep": 0,
                            "model": "YOLOv10_N",
                            "charged": True,
                            "outcome": "small_miss",
                            "battery_before": 0.5,
                            "battery_after": 0.51,
                            "clean_energy_before": 0.1,
                            "clean_energy_after": 0.15,
                            "dirty_energy_before": 0.05,
                            "dirty_energy_after": 0.0,
                            "charge_energy": 0.01,
                            "model_energy": 0.001,
                            "clean_energy_pct": 75.0,
                        }
                    ]
                }
            ],
            "top_most_clean_energy": [
                {
                    "action_sequence": [
                        {
                            "timestep": 0,
                            "model": "YOLOv10_S",
                            "charged": True,
                            "outcome": "success",
                            "battery_before": 0.8,
                            "battery_after": 0.85,
                            "clean_energy_before": 0.9,
                            "clean_energy_after": 0.95,
                            "dirty_energy_before": 0.1,
                            "dirty_energy_after": 0.0,
                            "charge_energy": 0.05,
                            "model_energy": 0.002,
                            "clean_energy_pct": 95.0,
                        }
                    ]
                }
            ],
            "top_least_total_energy": [
                {
                    "action_sequence": [
                        {
                            "timestep": 0,
                            "model": "YOLOv10_N",
                            "charged": False,
                            "outcome": "success",
                            "battery_before": 0.9,
                            "battery_after": 0.895,
                            "clean_energy_before": 0.2,
                            "clean_energy_after": 0.18,
                            "dirty_energy_before": 0.1,
                            "dirty_energy_after": 0.12,
                            "charge_energy": 0.0,
                            "model_energy": 0.005,
                            "clean_energy_pct": 60.0,
                        }
                    ]
                }
            ],
        },
    }

    # Test processing functions
    try:
        # Import functions from preprocess_data.py
        from preprocess_data import extract_bucket_sequence

        print("Successfully imported preprocessing functions")

        # Test timestep processing
        metadata = sample_metadata["metadata"]
        test_timestep = sample_metadata["results"]["top_success"][0]["action_sequence"][
            0
        ]

        training_sample = process_timestep_to_training_sample(
            test_timestep, metadata, "test_file.json"
        )

        print("Timestep processing test passed")
        if training_sample:
            print(f"  Sample keys: {list(training_sample.keys())}")
        else:
            print("  Error: training_sample is None")
            return False

        # Test bucket extraction
        success_seq = extract_bucket_sequence(sample_metadata, "top_success")
        small_miss_seq = extract_bucket_sequence(
            sample_metadata, "top_success_small_miss"
        )
        clean_energy_seq = extract_bucket_sequence(
            sample_metadata, "top_most_clean_energy"
        )
        least_energy_seq = extract_bucket_sequence(
            sample_metadata, "top_least_total_energy"
        )

        print("Bucket extraction test passed")
        print(f"  Success sequence length: {len(success_seq)}")
        print(f"  Small miss sequence length: {len(small_miss_seq)}")
        print(f"  Clean energy sequence length: {len(clean_energy_seq)}")
        print(f"  Least energy sequence length: {len(least_energy_seq)}")

        # Test that all sequences have expected structure
        expected_models = {"YOLOv10_L", "YOLOv10_B", "YOLOv10_N", "YOLOv10_S"}
        all_sequences = (
            success_seq + small_miss_seq + clean_energy_seq + least_energy_seq
        )

        for seq in all_sequences:
            for timestep_data in seq:
                if isinstance(timestep_data, dict):
                    model = timestep_data.get("model", "")
                    if model not in expected_models:
                        print(f"Unexpected model: {model}")
                        return False

        print("Model validation test passed")
        print("All tests passed! preprocess_data.py should work correctly.")

    except ImportError as e:
        print(f"Import error: {e}")
        assert False, f"Import error: {e}"
    except Exception as e:
        print(f"Test error: {e}")
        assert False, f"Test error: {e}"


if __name__ == "__main__":
    test_preprocess_data()
