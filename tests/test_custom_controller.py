#!/usr/bin/env python3
"""
Test Custom Controller
Standalone test for the trained custom neural controller.
Simulates controller behavior similar to tree_search.py but focused only on the custom controller.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from load_controller import load_controller, predict_with_controller
from tree_search import TreeSearch, TreeNode


def test_custom_controller(
    config_path: str, location: str = "CA", season: str = "winter"
):
    """Test custom controller in isolation."""
    print(f"🧠 Testing Custom Controller: {location} {season}")
    print(f"📁 Config: {config_path}")

    # Create tree search instance for infrastructure
    tree_search = TreeSearch(config_path, location, season, parallel=False)

    # Load trained controller
    try:
        model, controller_data = load_controller()
        print("✅ Custom controller loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load custom controller: {e}")
        return None

    # Create root node
    current_node = TreeNode(
        battery_level_wh=tree_search.config["battery"]["capacity_wh"],
        timestep=0,
        action_sequence=[],
        agg_clean_energy=0.0,
        agg_dirty_energy=0.0,
        decision_history=[],
    )

    print(f"🔋 Starting battery: {current_node.battery_level_wh:.6f}Wh")
    print(f"⏱️ Horizon: {tree_search.horizon} timesteps")
    print(
        f"⚡ Task interval: {tree_search.config['simulation']['task_interval_seconds']}s"
    )

    # Track statistics
    decisions = []
    model_counts = {}
    charge_count = 0
    success_count = 0
    small_miss_count = 0
    large_miss_count = 0

    start_time = time.time()

    # Run custom controller simulation for full horizon
    for timestep in range(tree_search.horizon):
        # Get clean energy percentage for current timestep
        timestamp = timestep * tree_search.config["simulation"]["task_interval_seconds"]
        clean_pct = tree_search.energy_data.get(int(timestamp % 86400), 50.0)

        # Extract features for controller input (6 features - removed task_interval_seconds)
        input_features = [
            current_node.battery_level_wh
            / tree_search.config["battery"]["capacity_wh"],  # battery_level normalized
            clean_pct,  # clean_energy_percentage
            tree_search.config["battery"]["capacity_wh"],  # battery_capacity_wh
            tree_search.config["battery"]["charge_rate_hours"],  # charge_rate_hours
            tree_search.config["simulation"][
                "user_accuracy_requirement"
            ],  # user_accuracy_requirement
            tree_search.config["simulation"][
                "user_latency_requirement"
            ],  # user_latency_requirement
        ]

        # Get controller prediction
        try:
            selected_model, should_charge, model_probs, charge_prob = (
                predict_with_controller(model, controller_data, input_features)
            )
            # Debug first few timesteps
            if timestep < 5:
                print(
                    f"🔍 DEBUG T{timestep}: battery={input_features[0]:.3f}, clean={input_features[1]:.1f}%, "
                    f"charge_prob={float(charge_prob):.3f}, should_charge={should_charge}"
                )
        except Exception as e:
            print(f"❌ Controller prediction failed at timestep {timestep}: {e}")
            break

        # Check if selected model has enough battery
        if selected_model != "IDLE" and not tree_search._has_enough_battery(
            selected_model, current_node.battery_level_wh
        ):
            # Fallback: find largest runnable model and force charging
            models_by_energy = sorted(
                tree_search.models.keys(),
                key=lambda x: tree_search.models[x]["energy_per_inference_mwh"],
                reverse=True,
            )
            fallback_model = None
            for model_name in models_by_energy:
                if tree_search._has_enough_battery(
                    model_name, current_node.battery_level_wh
                ):
                    fallback_model = model_name
                    break

            if fallback_model:
                action = (fallback_model, True)  # Force charging
                print(
                    f"⚠️ Controller fallback: {selected_model} → {fallback_model} + charge"
                )
            else:
                action = ("IDLE", True)  # No models can run
                print("⚠️ Controller fallback: IDLE + charge")
        else:
            action = (selected_model, should_charge)

        # Apply action
        next_node = tree_search._apply_action(current_node, action, clean_pct)

        if not next_node:
            print(f"❌ Controller failed at timestep {timestep}")
            break

        # Track decision
        decision_info = {
            "timestep": timestep,
            "model": action[0],
            "should_charge": action[1],
            "battery_before": current_node.battery_level_wh,
            "battery_after": next_node.battery_level_wh,
            "clean_energy_pct": clean_pct,
            "model_probs": model_probs.tolist()
            if hasattr(model_probs, "tolist")
            else model_probs,
            "charge_prob": float(charge_prob)
            if hasattr(charge_prob, "item")
            else charge_prob,
        }
        decisions.append(decision_info)

        # Count model usage
        model_name = action[0]
        model_counts[model_name] = model_counts.get(model_name, 0) + 1
        if action[1]:
            charge_count += 1

        # Count outcomes
        if timestep < len(next_node.decision_history):
            outcome = next_node.decision_history[timestep]["outcome"]
            if outcome == "success":
                success_count += 1
            elif outcome == "small_miss":
                small_miss_count += 1
            elif outcome == "large_miss":
                large_miss_count += 1

        # Progress logging
        if (timestep + 1) % 50 == 0 or timestep == tree_search.horizon - 1:
            print(
                f"📊 Timestep {timestep + 1}/{tree_search.horizon}: "
                f"model={action[0]}, charge={action[1]}, "
                f"battery={next_node.battery_level_wh:.4f}Wh"
            )

        current_node = next_node

    runtime = time.time() - start_time

    # Final statistics
    print("\n🎯 Custom Controller Test Results:")
    print(f"⏱️ Runtime: {runtime:.2f} seconds")
    print(f"🔋 Final battery: {current_node.battery_level_wh:.6f}Wh")
    print(f"📊 Total decisions: {len(decisions)}")
    print(
        f"⚡ Charging decisions: {charge_count} ({charge_count / len(decisions) * 100:.1f}%)"
    )
    print(f"✅ Success: {success_count} ({success_count / len(decisions) * 100:.1f}%)")
    print(
        f"⚠️ Small miss: {small_miss_count} ({small_miss_count / len(decisions) * 100:.1f}%)"
    )
    print(
        f"❌ Large miss: {large_miss_count} ({large_miss_count / len(decisions) * 100:.1f}%)"
    )
    print("🤖 Model usage:")
    for model, count in sorted(model_counts.items()):
        print(f"   {model}: {count} ({count / len(decisions) * 100:.1f}%)")

    # Energy statistics
    final_clean_energy = current_node.agg_clean_energy
    final_dirty_energy = current_node.agg_dirty_energy
    total_energy = final_clean_energy + final_dirty_energy
    clean_energy_pct = (
        (final_clean_energy / total_energy * 100) if total_energy > 0 else 0
    )

    print("⚡ Energy usage:")
    print(f"   Clean energy: {final_clean_energy:.6f}Wh")
    print(f"   Dirty energy: {final_dirty_energy:.6f}Wh")
    print(f"   Total energy: {total_energy:.6f}Wh")
    print(f"   Clean energy %: {clean_energy_pct:.2f}%")

    # Save detailed results
    results = {
        "test_metadata": {
            "config_path": config_path,
            "location": location,
            "season": season,
            "horizon": tree_search.horizon,
            "runtime_seconds": runtime,
            "timestamp": datetime.now().isoformat(),
        },
        "final_state": {
            "battery_level_wh": current_node.battery_level_wh,
            "agg_clean_energy": final_clean_energy,
            "agg_dirty_energy": final_dirty_energy,
            "total_energy": total_energy,
            "clean_energy_percentage": clean_energy_pct,
        },
        "statistics": {
            "total_decisions": len(decisions),
            "charging_decisions": charge_count,
            "success_count": success_count,
            "small_miss_count": small_miss_count,
            "large_miss_count": large_miss_count,
            "model_usage": model_counts,
        },
        "detailed_decisions": decisions,
    }

    output_path = Path(
        f"test-results/test_custom_controller_{location}_{season}_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"💾 Detailed results saved to {output_path}")

    return results


def main():
    """Main test entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test custom controller")
    parser.add_argument(
        "--config", default="config/config1.jsonc", help="Configuration file path"
    )
    args = parser.parse_args()
    config_path = args.config

    print("🧪 Custom Controller Test Suite")
    print("=" * 50)

    # Test different configurations
    test_cases = [
        ("CA", "winter"),
        ("CA", "summer"),
        ("FL", "winter"),
        ("FL", "summer"),
    ]

    all_results = {}

    for location, season in test_cases:
        print(f"\n🌍 Testing {location} {season}...")
        print("-" * 30)

        try:
            result = test_custom_controller(config_path, location, season)
            if result:
                all_results[f"{location}_{season}"] = result
                print(f"✅ {location} {season} test completed successfully")
            else:
                print(f"❌ {location} {season} test failed")
        except Exception as e:
            print(f"❌ {location} {season} test error: {e}")

    # Summary
    print("\n📊 Test Suite Summary:")
    print("=" * 30)
    for test_name, result in all_results.items():
        stats = result["statistics"]
        final_state = result["final_state"]
        print(f"{test_name}:")
        print(
            f"  Success rate: {stats['success_count'] / stats['total_decisions'] * 100:.1f}%"
        )
        print(f"  Clean energy: {final_state['clean_energy_percentage']:.2f}%")
        print(f"  Runtime: {result['test_metadata']['runtime_seconds']:.2f}s")

    print(f"\n✅ Completed {len(all_results)}/{len(test_cases)} tests successfully")

    return 0 if all_results else 1


if __name__ == "__main__":
    sys.exit(main())
