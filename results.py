#!/usr/bin/env python3
"""
Results Parser - Parses and displays CSV data from simulation results in human-readable format
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any
import statistics


def read_csv_full(filepath: str) -> List[Dict[str, Any]]:
    """Read entire CSV file"""
    data = []
    try:
        # Increase field size limit to handle large battery_levels data
        csv.field_size_limit(1000000)  # 1MB limit
        with open(filepath, "r", newline="") as file:
            reader = csv.DictReader(file)
            data = list(reader)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
    return data


def parse_json_field(field_str: str) -> Any:
    """Parse JSON field if possible, otherwise return as string"""
    if not field_str or field_str.strip() == "":
        return None
    try:
        return json.loads(field_str)
    except (json.JSONDecodeError, TypeError):
        return field_str


def analyze_battery_levels(battery_levels_str: str) -> Dict[str, float]:
    """Analyze battery levels data"""
    if not battery_levels_str:
        return {}

    try:
        levels = json.loads(battery_levels_str)
        if not levels:
            return {}

        battery_values = [entry["level"] for entry in levels]
        return {
            "initial": battery_values[0] if battery_values else 0,
            "final": battery_values[-1] if battery_values else 0,
            "min": min(battery_values) if battery_values else 0,
            "max": max(battery_values) if battery_values else 0,
            "avg": statistics.mean(battery_values) if battery_values else 0,
            "data_points": len(battery_values),
        }
    except (json.JSONDecodeError, KeyError, TypeError):
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
        successful = sum(1 for row in controller_data if row.get("success") == "True")
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
        if row.get("task_completion_rate"):
            try:
                completion_rates.append(float(row["task_completion_rate"]))
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
        if row.get("clean_energy_percentage"):
            try:
                clean_energy_percentages.append(float(row["clean_energy_percentage"]))
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

        if row.get("dirty_energy_mwh"):
            try:
                dirty_energies.append(
                    float(row["dirty_energy_mwh"]) * 1000
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

    # Power analysis
    print_subsection("Power Analysis")
    peak_powers = []
    avg_powers = []

    for row in data:
        if row.get("peak_power_mw"):
            try:
                peak_powers.append(float(row["peak_power_mw"]))
            except ValueError:
                pass

        if row.get("average_power_mw"):
            try:
                avg_powers.append(float(row["average_power_mw"]))
            except ValueError:
                pass

    if peak_powers:
        print(
            f"  Peak Power Usage - Avg: {statistics.mean(peak_powers):.1f}mW, "
            f"Max: {max(peak_powers):.1f}mW"
        )

    if avg_powers:
        print(f"  Average Power Usage - Avg: {statistics.mean(avg_powers):.1f}mW")

    # Battery analysis
    print_subsection("Battery Analysis")
    battery_stats = []
    battery_efficiency_scores = []
    charging_events = []

    for row in data:
        if row.get("battery_levels"):
            stats = analyze_battery_levels(row["battery_levels"])
            if stats:
                battery_stats.append(stats)

        if row.get("battery_efficiency_score"):
            try:
                score = float(row["battery_efficiency_score"])
                if score != float("inf"):
                    battery_efficiency_scores.append(score)
            except ValueError:
                pass

        if row.get("charging_events_count"):
            try:
                charging_events.append(int(row["charging_events_count"]))
            except ValueError:
                pass

    if battery_stats:
        avg_initial = statistics.mean([s["initial"] for s in battery_stats])
        avg_final = statistics.mean([s["final"] for s in battery_stats])
        avg_depletion = avg_initial - avg_final
        print(f"  Average Initial Battery: {avg_initial:.1f}%")
        print(f"  Average Final Battery: {avg_final:.1f}%")
        print(f"  Average Battery Depletion: {avg_depletion:.1f}%")

    if battery_efficiency_scores:
        print(
            f"  Battery Efficiency - Avg: {statistics.mean(battery_efficiency_scores):.1f} tasks per % depletion"
        )

    if charging_events:
        print(
            f"  Charging Events - Avg: {statistics.mean(charging_events):.1f}, Total: {sum(charging_events)}"
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

        # Battery analysis
        if row.get("battery_levels"):
            battery_stats = analyze_battery_levels(row["battery_levels"])
            if battery_stats:
                print("  Battery Analysis:")
                print(f"    Initial: {battery_stats['initial']:.1f}%")
                print(f"    Final: {battery_stats['final']:.1f}%")
                print(
                    f"    Depletion: {battery_stats['initial'] - battery_stats['final']:.1f}%"
                )
                print(f"    Average: {battery_stats['avg']:.1f}%")


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
        sum(model_energy.values())

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


def display_temporal_patterns(data: List[Dict[str, Any]]):
    """Display temporal patterns from JSON time series data."""
    print_section("TEMPORAL PATTERNS ANALYSIS")

    # Try to load time series data
    results_dir = Path("results")

    battery_file = results_dir / "battery_time_series.json"
    model_file = results_dir / "model_selection_timeline.json"
    results_dir / "energy_usage_summary.json"

    if battery_file.exists():
        print_subsection("Battery Level Patterns")
        try:
            with open(battery_file, "r") as f:
                battery_data = json.load(f)

            # Analyze battery depletion patterns
            for sim_id, sim_data in list(battery_data.items())[
                :3
            ]:  # Show first 3 simulations
                levels = sim_data.get("battery_levels", [])
                if levels:
                    initial = levels[0]["level"]
                    final = levels[-1]["level"]
                    depletion = initial - final
                    print(
                        f"  {sim_data.get('controller', 'Unknown')} ({sim_data.get('location', 'Unknown')}): "
                        f"{initial:.1f}% → {final:.1f}% ({depletion:.1f}% depletion)"
                    )
        except Exception as e:
            print(f"  Error analyzing battery data: {e}")

    if model_file.exists():
        print_subsection("Model Selection Timeline")
        try:
            with open(model_file, "r") as f:
                model_data = json.load(f)

            for sim_id, sim_data in list(model_data.items())[
                :3
            ]:  # Show first 3 simulations
                selections = sim_data.get("model_selections", {})
                if selections:
                    total_selections = sum(selections.values())
                    most_used = max(selections.items(), key=lambda x: x[1])
                    print(
                        f"  {sim_data.get('controller', 'Unknown')}: "
                        f"Most used model: {most_used[0]} ({most_used[1]} tasks, "
                        f"{(most_used[1] / total_selections) * 100:.1f}%)"
                    )
        except Exception as e:
            print(f"  Error analyzing model timeline: {e}")


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
            sum(1 for row in controller_data if row.get("success") == "True")
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

    # Battery efficiency
    battery_efficiencies = []
    for row in data:
        if row.get("battery_levels") and row.get("total_tasks"):
            battery_stats = analyze_battery_levels(row["battery_levels"])
            if battery_stats and battery_stats["initial"] > 0:
                depletion = battery_stats["initial"] - battery_stats["final"]
                tasks_per_percent = (
                    float(row["total_tasks"]) / depletion
                    if depletion > 0
                    else float("inf")
                )
                battery_efficiencies.append(tasks_per_percent)

    if battery_efficiencies:
        avg_efficiency = statistics.mean(
            [e for e in battery_efficiencies if e != float("inf")]
        )
        insights.append(
            f"Average battery efficiency: {avg_efficiency:.2f} tasks per percent battery depletion"
        )

    # Model usage insights
    model_usage = {}
    for row in data:
        for model in [
            "YOLOv10_N",
            "YOLOv10_S",
            "YOLOv10_M",
            "YOLOv10_B",
            "YOLOv10_L",
            "YOLOv10_X",
        ]:
            count_key = f"{model}_count"
            if count_key in row:
                model_usage[model] = model_usage.get(model, 0) + int(row[count_key])

    if model_usage:
        most_used_model = max(model_usage.items(), key=lambda x: x[1])
        total_model_usage = sum(model_usage.values())
        insights.append(
            f"Most used model: {most_used_model[0]} ({most_used_model[1]} tasks, {(most_used_model[1] / total_model_usage) * 100:.1f}% of all usage)"
        )

    # Display insights
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")

    # Recommendations
    print_subsection("Recommendations")
    print("• Consider using the best performing controller for critical applications")
    print(
        "• Focus on locations with higher clean energy availability for sustainability"
    )
    print("• Optimize battery management strategies based on efficiency patterns")
    print(
        "• Consider the most energy efficient controller for energy-constrained deployments"
    )
    print("• Further analysis needed on seasonal variations and their impact")


def main():
    """Main function to parse and display results"""
    results_dir = Path("results")

    if not results_dir.exists():
        print(f"Results directory '{results_dir}' not found!")
        return

    # File paths
    aggregated_file = results_dir / "batch-run-20251201_015744-aggregated-results.csv"
    detailed_file = results_dir / "batch-run-20251201_015744-detailed-results.csv"
    metadata_file = results_dir / "batch-run-20251201_015744-metadata.json"

    print("SECURITY CAMERA SIMULATION RESULTS ANALYZER")
    print("=" * 60)
    print(f"Analyzing results from: {results_dir}")

    # Display metadata
    if metadata_file.exists():
        display_metadata(str(metadata_file))
    else:
        print(f"Metadata file not found: {metadata_file}")

    # Read and display aggregated results
    if aggregated_file.exists():
        aggregated_data = read_csv_full(str(aggregated_file))
        display_aggregated_results(aggregated_data)
    else:
        print(f"Aggregated results file not found: {aggregated_file}")

    # Read and display detailed results
    if detailed_file.exists():
        detailed_data = read_csv_full(str(detailed_file))
        display_detailed_results(detailed_data, limit=20)
    else:
        print(f"Detailed results file not found: {detailed_file}")

    # Display comprehensive analysis
    aggregated_data = locals().get("aggregated_data")
    if aggregated_data:
        display_model_usage_breakdown(aggregated_data)
        display_energy_efficiency_analysis(aggregated_data)
        display_temporal_patterns(aggregated_data)
        display_key_insights(aggregated_data)

    print_section("ANALYSIS COMPLETE")
    print(
        "Note: This analysis processes the complete datasets including new comprehensive metrics"
    )


if __name__ == "__main__":
    main()
