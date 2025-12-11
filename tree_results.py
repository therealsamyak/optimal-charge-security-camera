#!/usr/bin/env python3
"""
Tree Search Results Visualization Script - Overhauled Version
Creates comprehensive visualizations comparing oracle, naive, and custom controllers.
"""

import json
import argparse
import sys
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Mapping
import numpy as np

import matplotlib.pyplot as plt
from src.graph_methods import (
    create_enhanced_heatmap,
    create_oracle_comparison_bars,
    create_clean_energy_comparison,
    create_battery_comparison,
    create_success_clean_scatter,
    create_regional_subplots,
)


class TreeResultsAnalyzer:
    """Analyzes and visualizes tree search results."""

    def __init__(self, results_file: str = "", aggregate_seasons: bool = False):
        """Initialize analyzer."""
        self.aggregate_seasons = aggregate_seasons
        self.base_categories = [
            "top_most_clean_energy",
            "top_success",
            "top_success_small_miss",
            "top_least_total_energy",
            "naive",
            "custom_controller",
        ]

        self.colors = {
            "top_most_clean_energy": "#2E8B57",  # Sea Green
            "top_success": "#4169E1",  # Royal Blue
            "top_success_small_miss": "#FFD700",  # Gold
            "top_least_total_energy": "#9370DB",  # Medium Purple
            "naive": "#808080",  # Gray
            "custom_controller": "#FF6B6B",  # Coral Red
        }

    def detect_config_type(self, metadata: Dict) -> str:
        """Detect config type based on metadata parameters."""
        capacity = metadata.get("battery_capacity_wh", 0)
        charge_rate = metadata.get("charge_rate_hours", 0)
        accuracy = metadata.get("user_accuracy_requirement", 0)
        latency = metadata.get("user_latency_requirement", 0)

        # Config1: 0.06055Wh, 1hr, 69% accuracy, 0.0016s latency
        if (
            abs(capacity - 0.06055) < 0.001
            and abs(charge_rate - 1.0) < 0.01
            and abs(accuracy - 69.0) < 0.1
            and abs(latency - 0.0016) < 0.0001
        ):
            return "config1"

        # Config2: 0.040Wh, 0.5hr, 81% accuracy, 0.0027s latency
        elif (
            abs(capacity - 0.040) < 0.001
            and abs(charge_rate - 0.5) < 0.01
            and abs(accuracy - 81.0) < 0.1
            and abs(latency - 0.0027) < 0.0001
        ):
            return "config2"

        # Config3: 0.080Wh, 2hr, 90% accuracy, 0.0060s latency
        elif (
            abs(capacity - 0.080) < 0.001
            and abs(charge_rate - 2.0) < 0.01
            and abs(accuracy - 90.0) < 0.1
            and abs(latency - 0.0060) < 0.0001
        ):
            return "config3"

        return "unknown"

    def find_config_files(self):
        """Find all result files grouped by config type and region."""
        all_files = glob.glob("results2/*-metadata.json")
        config_data = {}

        for file_path in all_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    metadata = data.get("metadata", {})

                    config_type = self.detect_config_type(metadata)
                    location = metadata.get("location", "Unknown")
                    season = metadata.get("season", "unknown")

                    if config_type not in config_data:
                        config_data[config_type] = {}

                    if location not in config_data[config_type]:
                        config_data[config_type][location] = {}

                    config_data[config_type][location][season] = Path(file_path)

            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

        return config_data

    def load_config_data(self, config_type: str) -> Dict[str, Dict]:
        """Load all data for a specific config type across regions and seasons."""
        config_files = self.find_config_files()

        if config_type not in config_files:
            print(f"⚠️  No files found for {config_type}")
            return {}

        config_data = config_files[config_type]
        loaded_data = {}

        for location, season_files in config_data.items():
            loaded_data[location] = {}
            for season, file_path in season_files.items():
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        loaded_data[location][season] = {
                            "data": data,
                            "metadata": data.get("metadata", {}),
                            "results": data.get("results", {}),
                        }
                except Exception as e:
                    print(f"❌ Error loading {location} {season}: {e}")

        return loaded_data

    def create_aggregate_graphs(self, config_data: Dict[str, Dict]):
        """Create aggregate graphs combining all regions."""
        # Combine data from all regions for each config
        aggregate_data = {}

        for config_type, regions in config_data.items():
            aggregate_data[config_type] = {"results": {}, "metadata": {}}

            # Combine all region data
            all_results = {}
            for region, seasons in regions.items():
                for season, season_data in seasons.items():
                    results = season_data.get("results", {})
                    metadata = season_data.get("metadata", {})

                    # Merge results
                    for cat, result in results.items():
                        if cat not in all_results:
                            all_results[cat] = []

                        if cat in ["naive", "custom_controller"]:
                            all_results[cat].append(result)
                        else:
                            all_results[cat].append(
                                result[0] if isinstance(result, list) else result
                            )

                    # Use metadata from first season
                    if not aggregate_data[config_type]["metadata"]:
                        aggregate_data[config_type]["metadata"] = metadata

            # Aggregate results for each category
            for cat, result_list in all_results.items():
                if cat in ["naive", "custom_controller"]:
                    # For direct results, just take the first one (they should be similar)
                    aggregate_data[config_type]["results"][cat] = (
                        result_list[0] if result_list else {}
                    )
                else:
                    # Aggregate across multiple results
                    if result_list:
                        # Average metrics
                        aggregated = {
                            "final_battery": np.mean(
                                [r.get("final_battery", 0) for r in result_list]
                            ),
                            "total_energy": np.mean(
                                [r.get("total_energy", 0) for r in result_list]
                            ),
                            "clean_energy_percentage": np.mean(
                                [
                                    r.get("clean_energy_percentage", 0)
                                    for r in result_list
                                ]
                            ),
                            "decision_counts": {
                                "success": sum(
                                    [
                                        r.get("decision_counts", {}).get("success", 0)
                                        for r in result_list
                                    ]
                                ),
                                "small_miss": sum(
                                    [
                                        r.get("decision_counts", {}).get(
                                            "small_miss", 0
                                        )
                                        for r in result_list
                                    ]
                                ),
                                "large_miss": sum(
                                    [
                                        r.get("decision_counts", {}).get(
                                            "large_miss", 0
                                        )
                                        for r in result_list
                                    ]
                                ),
                            },
                            "total_decisions": sum(
                                [r.get("total_decisions", 0) for r in result_list]
                            ),
                        }

                        # Aggregate action sequences
                        if result_list[0].get("action_sequence"):
                            max_length = max(
                                len(r.get("action_sequence", [])) for r in result_list
                            )
                            aggregated_action_sequence = []

                            for timestep in range(max_length):
                                timestep_actions = [
                                    r.get("action_sequence", [])[timestep]
                                    for r in result_list
                                    if timestep < len(r.get("action_sequence", []))
                                ]

                                if timestep_actions:
                                    # Average numeric fields
                                    aggregated_action = {
                                        "timestep": timestep,
                                        "model": timestep_actions[0].get(
                                            "model", "Unknown"
                                        ),
                                        "charged": timestep_actions[0].get(
                                            "charged", False
                                        ),
                                        "outcome": timestep_actions[0].get(
                                            "outcome", "unknown"
                                        ),
                                        "battery_before": np.mean(
                                            [
                                                a.get("battery_before", 0)
                                                for a in timestep_actions
                                            ]
                                        ),
                                        "battery_after": np.mean(
                                            [
                                                a.get("battery_after", 0)
                                                for a in timestep_actions
                                            ]
                                        ),
                                        "clean_energy_before": np.mean(
                                            [
                                                a.get("clean_energy_before", 0)
                                                for a in timestep_actions
                                            ]
                                        ),
                                        "clean_energy_after": np.mean(
                                            [
                                                a.get("clean_energy_after", 0)
                                                for a in timestep_actions
                                            ]
                                        ),
                                        "dirty_energy_before": np.mean(
                                            [
                                                a.get("dirty_energy_before", 0)
                                                for a in timestep_actions
                                            ]
                                        ),
                                        "dirty_energy_after": np.mean(
                                            [
                                                a.get("dirty_energy_after", 0)
                                                for a in timestep_actions
                                            ]
                                        ),
                                        "charge_energy": np.mean(
                                            [
                                                a.get("charge_energy", 0)
                                                for a in timestep_actions
                                            ]
                                        ),
                                        "model_energy": np.mean(
                                            [
                                                a.get("model_energy", 0)
                                                for a in timestep_actions
                                            ]
                                        ),
                                        "clean_energy_pct": np.mean(
                                            [
                                                a.get("clean_energy_pct", 0)
                                                for a in timestep_actions
                                            ]
                                        ),
                                    }
                                    aggregated_action_sequence.append(aggregated_action)

                            aggregated["action_sequence"] = aggregated_action_sequence

                        aggregate_data[config_type]["results"][cat] = aggregated

        return aggregate_data

    def run_analysis(
        self,
        show_plots: bool = True,
        save_plots: bool = False,
        config_focus: Optional[str] = None,
    ):
        """Run complete analysis with new visualizations."""

        # Load config data
        config_files = self.find_config_files()

        if config_focus:
            if config_focus not in config_files:
                print(f"⚠️  No files found for {config_focus}")
                return False
            config_files = {config_focus: config_files[config_focus]}

        figures = []

        try:
            # Create graphs for each config type
            for config_type, config_paths in config_files.items():
                print(f"\n📊 Processing {config_type}...")

                # Load actual data for this config
                config_data = self.load_config_data(config_type)
                if not config_data:
                    print(f"⚠️  No data loaded for {config_type}")
                    continue

                # Create regional subplots (2x2 layout)
                regional_fig = create_regional_subplots(
                    config_data, "oracle_comparison", self.colors
                )
                figures.append(regional_fig)

                regional_fig = create_regional_subplots(
                    config_data, "clean_energy", self.colors
                )
                figures.append(regional_fig)

                regional_fig = create_regional_subplots(
                    config_data, "battery_comparison", self.colors
                )
                figures.append(regional_fig)

                regional_fig = create_regional_subplots(
                    config_data, "success_clean_scatter", self.colors
                )
                figures.append(regional_fig)

                # Create aggregate graphs for this config
                aggregate_data = self.create_aggregate_graphs(config_data)
                if aggregate_data and config_type in aggregate_data:
                    metadata = aggregate_data[config_type]["metadata"]
                    results = aggregate_data[config_type]["results"]

                    # Enhanced model usage heatmap
                    heatmap_fig = create_enhanced_heatmap(
                        results, self.colors, metadata
                    )
                    figures.append(heatmap_fig)

                    # Oracle comparison bars
                    oracle_fig = create_oracle_comparison_bars(
                        results, self.colors, metadata
                    )
                    figures.append(oracle_fig)

                    # Clean energy comparison
                    clean_energy_fig = create_clean_energy_comparison(
                        results, self.colors, metadata
                    )
                    figures.append(clean_energy_fig)

                    # Battery comparison
                    battery_fig = create_battery_comparison(
                        results, self.colors, metadata
                    )
                    figures.append(battery_fig)

                    # Success vs clean scatter
                    scatter_fig = create_success_clean_scatter(
                        results, self.colors, metadata
                    )
                    figures.append(scatter_fig)

            # Save plots if requested
            if save_plots:
                print("💾 Saving plots to files...")
                tree_images_dir = Path("tree_images")
                tree_images_dir.mkdir(exist_ok=True)

                for i, fig in enumerate(figures):
                    filename = tree_images_dir / f"graph_{i + 1}.png"
                    fig.savefig(filename, dpi=300, bbox_inches="tight")
                    print(f"   Saved: {filename}")

            # Show plots if requested
            if show_plots:
                print("🖼️  Displaying plots (close windows to continue)...")
                plt.show()

            print("✅ Visualization complete!")
            return True

        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize tree search results with comprehensive graphs"
    )
    parser.add_argument("--save", action="store_true", help="Save plots to PNG files")
    parser.add_argument(
        "--no-show", action="store_true", help="Don't display interactive plots"
    )
    parser.add_argument(
        "--config-focus",
        help="Focus on specific config type (config1, config2, config3)",
    )

    args = parser.parse_args()

    # Check if results2/ has files
    all_files = glob.glob("results2/*-metadata.json")
    if not all_files:
        print("❌ No result files found in results2/")
        sys.exit(1)

    # Create analyzer and run analysis
    analyzer = TreeResultsAnalyzer()

    if analyzer.run_analysis(
        show_plots=not args.no_show,
        save_plots=args.save,
        config_focus=args.config_focus,
    ):
        print("✅ Analysis complete!")
    else:
        print("❌ Analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
