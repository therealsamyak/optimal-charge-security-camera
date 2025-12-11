#!/usr/bin/env python3
"""
Test new graph methods without displaying
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph_methods import (
    create_oracle_comparison_bars,
    create_clean_energy_comparison,
    create_battery_comparison,
    create_success_clean_scatter,
)


def test_graph_methods():
    """Test graph methods with sample data."""

    # Sample data
    sample_data = {
        "top_success": [
            {
                "decision_counts": {"success": 200, "small_miss": 50, "large_miss": 38},
                "total_decisions": 288,
                "action_sequence": [
                    {"timestep": i, "clean_energy_after": i * 0.1} for i in range(10)
                ],
            }
        ],
        "naive": {
            "decision_counts": {"success": 150, "small_miss": 80, "large_miss": 58},
            "total_decisions": 288,
            "action_sequence": [
                {"timestep": i, "clean_energy_after": i * 0.08} for i in range(10)
            ],
        },
        "custom_controller": {
            "decision_counts": {"success": 180, "small_miss": 60, "large_miss": 48},
            "total_decisions": 288,
            "action_sequence": [
                {"timestep": i, "clean_energy_after": i * 0.09} for i in range(10)
            ],
        },
    }

    sample_metadata = {
        "battery_capacity_wh": 0.080,
        "charge_rate_hours": 2.0,
        "user_accuracy_requirement": 90.0,
        "user_latency_requirement": 0.006,
    }

    colors = {
        "top_success": "#4169E1",
        "naive": "#808080",
        "custom_controller": "#FF6B6B",
    }

    print("Testing graph methods...")

    import matplotlib.pyplot as plt

    try:
        fig1 = create_oracle_comparison_bars(sample_data, colors, sample_metadata)
        print("✅ Oracle comparison bars: OK")
        plt.close(fig1)
    except Exception as e:
        print(f"❌ Oracle comparison bars: {e}")

    try:
        fig2 = create_clean_energy_comparison(sample_data, colors, sample_metadata)
        print("✅ Clean energy comparison: OK")
        plt.close(fig2)
    except Exception as e:
        print(f"❌ Clean energy comparison: {e}")

    try:
        fig3 = create_battery_comparison(sample_data, colors, sample_metadata)
        print("✅ Battery comparison: OK")
        plt.close(fig3)
    except Exception as e:
        print(f"❌ Battery comparison: {e}")

    try:
        fig4 = create_success_clean_scatter(sample_data, colors, sample_metadata)
        print("✅ Success vs clean scatter: OK")
        plt.close(fig4)
    except Exception as e:
        print(f"❌ Success vs clean scatter: {e}")

    print("Graph method testing complete!")


if __name__ == "__main__":
    test_graph_methods()
