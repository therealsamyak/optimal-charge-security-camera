#!/usr/bin/env python3
"""
Results Parser - Parses and displays JSON data from simulation results in human-readable format
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import statistics


def read_json_full(filepath: str) -> Dict[str, Any]:
    """Read entire JSON file"""
    try:
        with open(filepath, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}


def analyze_battery_levels(battery_levels: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze battery levels data"""
    if not battery_levels:
        return {}

    try:
        battery_values = [entry["level"] for entry in battery_levels]
        return {
            "initial": battery_values[0] if battery_values else 0,
            "final": battery_values[-1] if battery_values else 0,
            "min": min(battery_values) if battery_values else 0,
            "max": max(battery_values) if battery_values else 0,
            "avg": statistics.mean(battery_values) if battery_values else 0,
            "data_points": len(battery_values),
        }
    except (KeyError, TypeError):
        return {}


def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)


def print_subsection(title: str):
    """Print formatted subsection header"""
    print(f"\n--- {title} ---")


def display_metadata(metadata_path: str):
    """Display simulation metadata"""
    print_section("SIMULATION METADATA")

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        print(f"Total Simulations: {metadata.get('total_simulations', 'N/A')}")
        print(f"Total Variations: {metadata.get('total_variations', 'N/A')}")
        print(f"Locations: {', '.join(metadata.get('locations', []))}")
        print(f"Seasons: {', '.join(metadata.get('seasons', []))}")
        print(f"Controllers: {', '.join(metadata.get('controllers', []))}")
        print(
            f"Weeks Simulated: {', '.join(map(str, metadata.get('weeks_simulated', [])))}"
        )
        print(f"Export Timestamp: {metadata.get('export_timestamp', 'N/A')}")
        print(f"Config File: {metadata.get('config_file', 'N/A')}")

        batch_config = metadata.get("batch_config", {})
        if batch_config:
            print_subsection("Batch Configuration Ranges")
            for key, range_data in batch_config.items():
                if (
                    isinstance(range_data, dict)
                    and "min" in range_data
                    and "max" in range_data
                ):
                    print(
                        f"  {key.replace('_', ' ').title()}: {range_data['min']} - {range_data['max']}"
                    )
                else:
                    print(f"  {key.replace('_', ' ').title()}: {range_data}")

    except Exception as e:
        print(f"Error reading metadata: {e}")


def display_aggregated_results(data: List[Dict[str, Any]]):
    """Display aggregated results summary."""
    print_section("AGGREGATED RESULTS SUMMARY")

    if not data:
        print("No data available")
        return

    # Basic statistics
    controllers = list(set(row.get("controller", "Unknown") for row in data))
    locations = list(set(row.get("location", "Unknown") for row in data))

    print(f"Total Records Analyzed: {len(data)}")
    print(f"Controllers: {', '.join(controllers)}")
    print(f"Locations: {', '.join(locations)}")

    # Success rates by controller
    print_subsection("Success Rates by Controller")
    controller_stats = {}
    for controller in controllers:
        controller_data = [row for row in data if row.get("controller") == controller]
        successful = sum(1 for row in controller_data if row.get("success", True))
        total = len(controller_data)
        success_rate = (successful / total * 100) if total > 0 else 0
        controller_stats[controller] = {
            "success_rate": success_rate,
            "total": total,
            "successful": successful,
        }
        print(f"  {controller}: {success_rate:.1f}% ({successful}/{total} successful)")

    # Performance metrics
    print_subsection("Performance Metrics")

    # Task completion rates
    completion_rates = []
    for row in data:
        if row.get("avg_task_completion_rate"):
            try:
                completion_rates.append(float(row["avg_task_completion_rate"]))
            except ValueError:
                pass

    if completion_rates:
        print(
            f"  Task Completion Rate - Avg: {statistics.mean(completion_rates):.1f}%, "
            f"Min: {min(completion_rates):.1f}%, Max: {max(completion_rates):.1f}%"
        )

    # Enhanced energy statistics
    print_subsection("Energy Analysis")
    clean_energy_percentages = []
    total_energies = []
    clean_energies = []
    dirty_energies = []
    energy_per_task = []

    for row in data:
        if row.get("avg_clean_energy_percentage"):
            try:
                clean_energy_percentages.append(
                    float(row["avg_clean_energy_percentage"])
                )
            except ValueError:
                pass

        if row.get("total_energy_wh"):
            try:
                total_energies.append(float(row["total_energy_wh"]))
            except ValueError:
                pass

        if row.get("clean_energy_wh"):
            try:
                clean_energies.append(float(row["clean_energy_wh"]))
            except ValueError:
                pass

        if row.get("total_dirty_energy_mwh"):
            try:
                dirty_energies.append(
                    float(row["total_dirty_energy_mwh"]) * 1000
                )  # Convert to Wh
            except ValueError:
                pass

        if row.get("energy_per_task_wh"):
            try:
                energy_per_task.append(float(row["energy_per_task_wh"]))
            except ValueError:
                pass

    if clean_energy_percentages:
        print(
            f"  Clean Energy Usage - Avg: {statistics.mean(clean_energy_percentages):.1f}%, "
            f"Min: {min(clean_energy_percentages):.1f}%, Max: {max(clean_energy_percentages):.1f}%"
        )

    if total_energies:
        print(
            f"  Total Energy Consumption - Avg: {statistics.mean(total_energies):.3f}Wh, "
            f"Min: {min(total_energies):.3f}Wh, Max: {max(total_energies):.3f}Wh"
        )

    if clean_energies:
        print(
            f"  Clean Energy Consumption - Avg: {statistics.mean(clean_energies):.3f}Wh, "
            f"Total: {sum(clean_energies):.1f}Wh"
        )

    if dirty_energies:
        print(
            f"  Dirty Energy Consumption - Avg: {statistics.mean(dirty_energies):.3f}Wh, "
            f"Total: {sum(dirty_energies):.1f}Wh"
        )

    if energy_per_task:
        print(
            f"  Energy Efficiency - Avg: {statistics.mean(energy_per_task):.3f}Wh per task"
        )


def display_detailed_results(data: List[Dict[str, Any]], limit: int = 10):
    """Display detailed results for first few records"""
    print_section(f"DETAILED RESULTS (First {limit} Records)")

    if not data:
        print("No data available")
        return

    for i, row in enumerate(data[:limit]):
        print_subsection(f"Record {i + 1}")

        # Basic info
        print(f"  Controller: {row.get('controller', 'N/A')}")
        print(f"  Location: {row.get('location', 'N/A')}")
        print(f"  Season: {row.get('season', 'N/A')}")
        print(f"  Week: {row.get('week', 'N/A')}")
        print(f"  Success: {row.get('success', 'N/A')}")

        # Performance metrics
        print(f"  Task Completion Rate: {row.get('task_completion_rate', 'N/A')}%")
        print(f"  Completed Tasks: {row.get('completed_tasks', 'N/A')}")
        print(f"  Total Tasks: {row.get('total_tasks', 'N/A')}")
        print(f"  Missed Deadlines: {row.get('missed_deadlines', 'N/A')}")

        # Energy metrics
        print(
            f"  Clean Energy Percentage: {row.get('clean_energy_percentage', 'N/A')}%"
        )
        print(f"  Total Energy (Wh): {row.get('total_energy_wh', 'N/A')}")
        print(f"  Battery Capacity (Wh): {row.get('battery_capacity_wh', 'N/A')}")

        # Model performance
        print(f"  Large Model Miss Rate: {row.get('large_model_miss_rate', 'N/A')}%")
        print(f"  Small Model Miss Rate: {row.get('small_model_miss_rate', 'N/A')}%")

        # Requirements
        print(f"  Accuracy Requirement: {row.get('accuracy_requirement', 'N/A')}%")
        print(f"  Latency Requirement: {row.get('latency_requirement', 'N/A')}s")


def display_model_usage_breakdown(data: List[Dict[str, Any]]):
    """Display detailed model usage analysis by controller."""
    print_section("MODEL USAGE BREAKDOWN")

    if not data:
        print("No data available")
        return

    controllers = list(set(row.get("controller", "Unknown") for row in data))

    for controller in controllers:
        controller_data = [row for row in data if row.get("controller") == controller]
        print_subsection(f"{controller.upper()} Controller Model Usage")

        # Aggregate model counts and energy
        model_counts = {}
        model_energy = {}

        for row in controller_data:
            for model in [
                "YOLOv10_N",
                "YOLOv10_S",
                "YOLOv10_M",
                "YOLOv10_B",
                "YOLOv10_L",
                "YOLOv10_X",
            ]:
                count_key = f"{model}_count"
                energy_key = f"{model}_energy_wh"

                if count_key in row:
                    model_counts[model] = model_counts.get(model, 0) + int(
                        row[count_key]
                    )

                if energy_key in row:
                    model_energy[model] = model_energy.get(model, 0) + float(
                        row[energy_key]
                    )

        total_tasks = sum(model_counts.values())

        if total_tasks > 0:
            print(f"  Total Tasks: {total_tasks}")
            print("  Model Distribution:")
            for model, count in sorted(
                model_counts.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total_tasks) * 100
                energy_used = model_energy.get(model, 0)
                print(
                    f"    {model}: {count} tasks ({percentage:.1f}%), {energy_used:.3f}Wh"
                )


def display_energy_efficiency_analysis(data: List[Dict[str, Any]]):
    """Display detailed energy efficiency analysis."""
    print_section("ENERGY EFFICIENCY ANALYSIS")

    if not data:
        print("No data available")
        return

    # Energy efficiency by controller
    controllers = list(set(row.get("controller", "Unknown") for row in data))

    print_subsection("Energy Efficiency by Controller")
    for controller in controllers:
        controller_data = [row for row in data if row.get("controller") == controller]

        energy_per_tasks = [
            float(row.get("energy_per_task_wh", 0))
            for row in controller_data
            if row.get("energy_per_task_wh")
        ]
        clean_energy_per_tasks = [
            float(row.get("clean_energy_per_task_wh", 0))
            for row in controller_data
            if row.get("clean_energy_per_task_wh")
        ]

        if energy_per_tasks:
            print(f"  {controller}:")
            print(f"    Avg Energy per Task: {statistics.mean(energy_per_tasks):.3f}Wh")
            print(
                f"    Total Energy Consumed: {sum([float(row.get('total_energy_wh', 0)) for row in controller_data]):.1f}Wh"
            )

        if clean_energy_per_tasks:
            print(
                f"    Avg Clean Energy per Task: {statistics.mean(clean_energy_per_tasks):.3f}Wh"
            )


def display_temporal_patterns(json_data: Dict[str, Any]):
    """Display temporal patterns from JSON time series data."""
    print_section("TEMPORAL PATTERNS ANALYSIS")

    time_series = json_data.get("time_series", {})

    # Battery levels analysis
    battery_data = time_series.get("battery_levels", {})
    if battery_data:
        print_subsection("Battery Level Patterns")
        for sim_id, sim_data in list(battery_data.items())[
            :3
        ]:  # Show first 3 simulations
            levels = sim_data.get("levels", [])
            if levels:
                initial = levels[0]["level"]
                final = levels[-1]["level"]
                depletion = initial - final
                print(
                    f"  {sim_data.get('controller', 'Unknown')} ({sim_data.get('location', 'Unknown')}): "
                    f"{initial:.1f}% → {final:.1f}% ({depletion:.1f}% depletion)"
                )

    # Model selection timeline
    model_data = time_series.get("model_selections", {})
    if model_data:
        print_subsection("Model Selection Timeline")
        for sim_id, sim_data in list(model_data.items())[
            :3
        ]:  # Show first 3 simulations
            selections = sim_data.get("selections", {})
            if selections:
                total_selections = sum(selections.values())
                most_used = max(selections.items(), key=lambda x: x[1])
                print(
                    f"  {sim_data.get('controller', 'Unknown')}: "
                    f"Most used model: {most_used[0]} ({most_used[1]} tasks, "
                    f"{(most_used[1] / total_selections) * 100:.1f}%)"
                )


def display_key_insights(data: List[Dict[str, Any]]):
    """Display key insights and conclusions."""
    print_section("KEY INSIGHTS & CONCLUSIONS")

    if not data:
        print("No data available for analysis")
        return

    insights = []

    # Controller performance comparison
    controllers = list(set(row.get("controller", "Unknown") for row in data))
    controller_performance = {}

    for controller in controllers:
        controller_data = [row for row in data if row.get("controller") == controller]
        success_rate = (
            sum(1 for row in controller_data if row.get("success", True))
            / len(controller_data)
            * 100
        )
        avg_completion = statistics.mean(
            [
                float(row.get("task_completion_rate", 0))
                for row in controller_data
                if row.get("task_completion_rate")
            ]
        )
        avg_efficiency = statistics.mean(
            [
                float(row.get("energy_per_task_wh", 0))
                for row in controller_data
                if row.get("energy_per_task_wh")
            ]
        )
        controller_performance[controller] = {
            "success_rate": success_rate,
            "avg_completion": avg_completion,
            "avg_efficiency": avg_efficiency,
        }

    best_controller = max(
        controller_performance.items(), key=lambda x: x[1]["success_rate"]
    )
    insights.append(
        f"Best performing controller: {best_controller[0]} with {best_controller[1]['success_rate']:.1f}% success rate"
    )

    # Most energy efficient controller
    efficient_controller = min(
        [
            (c, p["avg_efficiency"])
            for c, p in controller_performance.items()
            if p["avg_efficiency"] > 0
        ],
        key=lambda x: x[1],
    )
    if efficient_controller:
        insights.append(
            f"Most energy efficient: {efficient_controller[0]} with {efficient_controller[1]:.3f}Wh per task"
        )

    # Location analysis
    locations = list(set(row.get("location", "Unknown") for row in data))
    location_performance = {}

    for location in locations:
        location_data = [row for row in data if row.get("location") == location]
        avg_clean_energy = statistics.mean(
            [
                float(row.get("clean_energy_percentage", 0))
                for row in location_data
                if row.get("clean_energy_percentage")
            ]
        )
        total_energy = sum(
            [float(row.get("total_energy_wh", 0)) for row in location_data]
        )
        location_performance[location] = {
            "clean_energy": avg_clean_energy,
            "total_energy": total_energy,
        }

    best_location = max(
        location_performance.items(), key=lambda x: x[1]["clean_energy"]
    )
    insights.append(
        f"Best location for clean energy: {best_location[0]} with {best_location[1]['clean_energy']:.1f}% clean energy usage"
    )

    # Display insights
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")

    # Recommendations
    print_subsection("Recommendations")
    print("• Consider using best performing controller for critical applications")
    print(
        "• Focus on locations with higher clean energy availability for sustainability"
    )
    print("• Optimize battery management strategies based on efficiency patterns")
    print(
        "• Consider most energy efficient controller for energy-constrained deployments"
    )
    print("• Further analysis needed on seasonal variations and their impact")


def main():
    """Main function to parse and display results"""
    results_dir = Path("results")

    if not results_dir.exists():
        print(f"Results directory '{results_dir}' not found!")
        return

    # Look for the most recent JSON results file
    json_files = list(results_dir.glob("batch-run-*-results.json"))
    if not json_files:
        print("No JSON results files found!")
        return

    # Use the most recent file
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    metadata_file = latest_file.with_name(
        latest_file.name.replace("-results.json", "-metadata.json")
    )

    print("SECURITY CAMERA SIMULATION RESULTS ANALYZER")
    print("=" * 60)
    print(f"Analyzing results from: {latest_file}")

    # Read JSON data
    json_data = read_json_full(str(latest_file))

    # Display metadata
    if metadata_file.exists():
        display_metadata(str(metadata_file))
    else:
        print(f"Metadata file not found: {metadata_file}")

    # Extract and display aggregated results
    aggregated_data = json_data.get("aggregated_metrics", [])
    if aggregated_data:
        display_aggregated_results(aggregated_data)
    else:
        print("No aggregated metrics found in JSON data")

    # Extract and display detailed results
    detailed_data = json_data.get("detailed_metrics", [])
    if detailed_data:
        display_detailed_results(detailed_data, limit=20)
    else:
        print("No detailed metrics found in JSON data")

    # Display comprehensive analysis
    if aggregated_data:
        display_model_usage_breakdown(aggregated_data)
        display_energy_efficiency_analysis(aggregated_data)
        display_temporal_patterns(json_data)
        display_key_insights(aggregated_data)

    print_section("ANALYSIS COMPLETE")
    print(
        "Note: This analysis processes complete JSON datasets with hierarchical structure"
    )


if __name__ == "__main__":
    main()
