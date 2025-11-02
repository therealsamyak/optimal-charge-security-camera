#!/usr/bin/env python3
"""
Test script to verify model selection logic under different resource scenarios.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.controller.intelligent_controller import ModelController


def test_resource_scenarios():
    """Test model selection under various resource conditions."""
    controller = ModelController()

    scenarios = [
        # (battery_level, energy_cleanliness, is_charging, description)
        (90.0, 95.0, False, "High battery + Clean energy"),
        (80.0, 80.0, False, "Medium-high battery + Clean energy"),
        (60.0, 60.0, False, "Medium battery + Moderate energy"),
        (40.0, 40.0, False, "Low battery + Dirty energy"),
        (20.0, 20.0, False, "Critical battery + Very dirty energy"),
        (50.0, 95.0, True, "Medium battery + Clean energy + Charging"),
        (30.0, 90.0, True, "Low battery + Very clean energy + Charging"),
    ]

    print("Model Selection Test Results")
    print("=" * 80)
    print(f"{'Scenario':<40} {'Model':<12} {'Score':<8} {'Should Charge':<12}")
    print("-" * 80)

    for battery, energy, charging, description in scenarios:
        decision = controller.select_optimal_model(battery, energy, charging)

        print(
            f"{description:<40} {decision.selected_model:<12} {decision.score:<8.1f} {decision.should_charge!s:<12}"
        )

        # Show detailed reasoning for first few scenarios
        if battery >= 80.0 or battery <= 30.0:
            print(f"  Reasoning: {decision.reasoning}")
            print()

    print("\nModel Profiles Reference:")
    print("=" * 50)
    for model_name, profile in controller.model_profiles.items():
        print(
            f"{model_name:<8} | Acc:{profile.accuracy:>5.1f}% | Lat:{profile.latency_ms:>5.0f}ms | Battery:{profile.battery_consumption:>4.1f} | Size:{profile.size_rank}"
        )


if __name__ == "__main__":
    test_resource_scenarios()
