#!/usr/bin/env python3
"""
Tree Search Results Visualization Script
Parses and visualizes tree search results from tree_search.py output.
Creates separate matplotlib windows for different analysis perspectives.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm


class TreeResultsAnalyzer:
    """Analyzes and visualizes tree search results."""

    def __init__(self, results_file: str):
        """Initialize analyzer with results file path."""
        self.results_file = Path(results_file)
        self.data: Optional[Dict] = None
        self.metadata: Optional[Dict] = None
        self.results: Optional[Dict] = None
        self.base_categories = [
            "top_most_clean_energy",
            "top_success",
            "top_success_small_miss",
            "top_least_total_energy",
            "naive",
        ]

        self.colors = {
            "top_most_clean_energy": "#2E8B57",  # Sea Green
            "top_success": "#4169E1",  # Royal Blue
            "top_success_small_miss": "#FFD700",  # Gold
            "top_least_total_energy": "#9370DB",  # Medium Purple
            "naive": "#808080",  # Gray
        }
        # Distinct colors for different models
        self.model_colors = {
            "yolov8n": "#FF6B6B",  # Red
            "yolov8s": "#4ECDC4",  # Teal
            "yolov8m": "#45B7D1",  # Blue
            "yolov8l": "#96CEB4",  # Green
            "yolov8x": "#FFEAA7",  # Yellow
            "efficientdet": "#DDA0DD",  # Plum
            "mobilenet": "#F4A460",  # Sandy Brown
        }

    @property
    def categories(self):
        """Get available categories based on loaded data."""
        if self.results is None:
            return self.base_categories

        available = []
        for cat in self.base_categories:
            if cat in self.results and self.results[cat]:
                available.append(cat)

        return available

    def load_data(self) -> bool:
        """Load and validate tree search results data."""
        try:
            if not self.results_file.exists():
                print(f"❌ Error: Results file not found: {self.results_file}")
                print(f"💡 Tip: Use --timestamp 20251209_230426 to specify timestamp")
                return False

            print(f"📊 Loading tree search results from {self.results_file}...")
            with open(self.results_file, "r") as f:
                self.data = json.load(f)

            self.metadata = self.data.get("metadata", {}) if self.data else {}
            self.results = self.data.get("results", {}) if self.data else {}

            # Validate data structure
            if self.results is None or not all(
                cat in self.results for cat in self.categories
            ):
                missing = [
                    cat
                    for cat in self.categories
                    if self.results is None or cat not in self.results
                ]
                print(f"❌ Error: Missing result categories: {missing}")
                print(
                    f"📋 Available categories: {list(self.results.keys()) if self.results else 'None'}"
                )
                return False

            print(
                f"✅ Loaded results for {self.metadata.get('location', 'Unknown') if self.metadata else 'Unknown'} {self.metadata.get('season', 'Unknown') if self.metadata else 'Unknown'}"
            )
            print(
                f"   Horizon: {self.metadata.get('horizon', 0) if self.metadata else 0} timesteps"
            )
            print(
                f"   Total leaves explored: {self.metadata.get('total_leaves_explored', 0) if self.metadata else 0}"
            )
            return True

        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in results file: {e}")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def extract_time_series(self, result_data: Dict) -> Dict[str, List]:
        """Extract time series data from a single result."""
        action_sequence = result_data.get("action_sequence", [])

        timesteps = []
        battery_levels = []
        clean_energy = []
        dirty_energy = []
        models = []
        charging = []
        outcomes = []

        for action in action_sequence:
            timesteps.append(action.get("timestep", 0))
            battery_levels.append(action.get("battery_after", 0))
            clean_energy.append(action.get("clean_energy_after", 0))
            dirty_energy.append(action.get("dirty_energy_after", 0))
            models.append(action.get("model", "Unknown"))
            charging.append(action.get("charged", False))
            outcomes.append(action.get("outcome", "unknown"))

        return {
            "timesteps": timesteps,
            "battery_levels": battery_levels,
            "clean_energy": clean_energy,
            "dirty_energy": dirty_energy,
            "models": models,
            "charging": charging,
            "outcomes": outcomes,
        }

    def get_category_summary(self, category: str) -> Dict:
        """Get summary statistics for a category."""
        if (
            self.results is None
            or category not in self.results
            or not self.results[category]
        ):
            return {}

        # Handle both list (top_*) and dict (naive) result types
        if category == "naive":
            result = self.results[category]
        else:
            result = self.results[category][0]

        return {
            "final_battery": result.get("final_battery", 0),
            "total_energy": result.get("total_energy", 0),
            "clean_energy_percentage": result.get("clean_energy_percentage", 0),
            "decision_counts": result.get("decision_counts", {}),
            "total_decisions": result.get("total_decisions", 0),
        }

    def plot_decision_outcomes_stacked(self):
        """Chart 1: Stacked bar graph of decision outcomes for all categories."""
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Decision Outcomes by Category", fontsize=14, fontweight="bold")

        categories = self.categories
        success_counts = []
        small_miss_counts = []
        large_miss_counts = []

        for cat in categories:
            if self.results is None or not self.results.get(cat):
                success_counts.append(0)
                small_miss_counts.append(0)
                large_miss_counts.append(0)
                continue

            # Handle both list (top_*) and dict (naive) result types
            if cat == "naive":
                result = self.results[cat]
            else:
                result = self.results[cat][0]  # Best result from list

            decision_counts = result.get("decision_counts", {})
            success_counts.append(decision_counts.get("success", 0))
            small_miss_counts.append(decision_counts.get("small_miss", 0))
            large_miss_counts.append(decision_counts.get("large_miss", 0))

        # Create stacked bars
        width = 0.6
        x = np.arange(len(categories))

        bars1 = ax.bar(x, success_counts, width, label="Success", color="#2E8B57")
        bars2 = ax.bar(
            x,
            small_miss_counts,
            width,
            bottom=success_counts,
            label="Small Miss",
            color="#FF8C00",
        )
        bars3 = ax.bar(
            x,
            large_miss_counts,
            width,
            bottom=np.array(success_counts) + np.array(small_miss_counts),
            label="Large Miss",
            color="#DC143C",
        )

        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.set_title("Decision Outcomes")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [cat.replace("top_", "").replace("_", " ").title() for cat in categories]
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_clean_vs_success_scatter(self):
        """Chart 2: Clean energy % vs success % scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Clean Energy vs Success Tradeoff", fontsize=14, fontweight="bold")

        clean_pcts = []
        success_pcts = []

        for cat in self.categories:
            if self.results is None or not self.results.get(cat):
                continue

            # Handle both list (top_*) and dict (naive) result types
            if cat == "naive":
                result = self.results[cat]
            else:
                result = self.results[cat][0]

            clean_pcts.append(result.get("clean_energy_percentage", 0))

            decision_counts = result.get("decision_counts", {})
            total_decisions = result.get("total_decisions", 1)
            successes = decision_counts.get("success", 0)
            success_pct = (
                (successes / total_decisions * 100) if total_decisions > 0 else 0
            )
            success_pcts.append(success_pct)

        # Filter categories that have results
        valid_cats = [
            c for c in self.categories if self.results and self.results.get(c)
        ]
        colors = [self.colors.get(cat, "#808080") for cat in valid_cats]

        scatter = ax.scatter(
            clean_pcts, success_pcts, c=colors, s=100, alpha=0.8, edgecolors="black"
        )

        # Add category labels
        for i, cat in enumerate(valid_cats):
            ax.annotate(
                cat.replace("top_", "").replace("_", " ").title(),
                (clean_pcts[i], success_pcts[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        ax.set_xlabel("Clean Energy %")
        ax.set_ylabel("Success %")
        ax.set_title("Clean Energy vs Success")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_clean_energy_bars(self):
        """Chart 3: Simple bar chart of clean energy percentage."""
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(
            "Clean Energy Percentage by Category", fontsize=14, fontweight="bold"
        )

        clean_pcts = []
        cat_labels = []

        for cat in self.categories:
            if self.results is None or not self.results.get(cat):
                continue

            # Handle both list (top_*) and dict (naive) result types
            if cat == "naive":
                result = self.results[cat]
            else:
                result = self.results[cat][0]

            clean_pcts.append(result.get("clean_energy_percentage", 0))
            cat_labels.append(cat.replace("top_", "").replace("_", " ").title())

        colors = [
            self.colors.get(cat, "#808080")
            for cat in self.categories
            if self.results and self.results.get(cat)
        ]

        bars = ax.bar(cat_labels, clean_pcts, color=colors)

        # Add percentage labels on bars
        for bar, val in zip(bars, clean_pcts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
            )

        ax.set_ylabel("Clean Energy %")
        ax.set_title("Clean Energy %")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_battery_decay_lines(self):
        """Chart 4: Line chart showing battery decay over time."""
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("Battery Levels Over Time", fontsize=14, fontweight="bold")

        for cat in self.categories:
            if self.results is None or not self.results.get(cat):
                continue

            # Handle both list (top_*) and dict (naive) result types
            if cat == "naive":
                result = self.results[cat]
            else:
                result = self.results[cat][0]

            ts_data = self.extract_time_series(result)

            ax.plot(
                ts_data["timesteps"],
                ts_data["battery_levels"],
                label=cat.replace("top_", "").replace("_", " ").title(),
                color=self.colors[cat],
                linewidth=2,
            )

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Battery Level (Wh)")
        ax.set_title("Battery Decay")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_model_usage_heatmap(self):
        """Chart 5: Heatmap showing model usage frequency per category."""
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle("Model Usage Frequency", fontsize=14, fontweight="bold")

        # Collect all unique models across categories
        all_models = set()
        for cat in self.categories:
            if self.results is None or not self.results.get(cat):
                continue

            # Handle both list (top_*) and dict (naive) result types
            if cat == "naive":
                result = self.results[cat]
            else:
                result = self.results[cat][0]

            ts_data = self.extract_time_series(result)
            all_models.update(ts_data["models"])

        all_models = sorted(list(all_models))

        # Create usage matrix
        usage_matrix = []
        cat_labels = []

        for cat in self.categories:
            if self.results is None or not self.results.get(cat):
                continue

            cat_labels.append(cat.replace("top_", "").replace("_", " ").title())

            # Handle both list (top_*) and dict (naive) result types
            if cat == "naive":
                result = self.results[cat]
            else:
                result = self.results[cat][0]

            ts_data = self.extract_time_series(result)

            # Count model usage
            model_counts = {model: 0 for model in all_models}
            for model in ts_data["models"]:
                if model in model_counts:
                    model_counts[model] += 1

            usage_matrix.append([model_counts[model] for model in all_models])

        # Create heatmap
        im = ax.imshow(usage_matrix, cmap="YlOrRd", aspect="auto")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(all_models)))
        ax.set_xticklabels(all_models, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(cat_labels)))
        ax.set_yticklabels(cat_labels)

        # Add text annotations
        for i in range(len(cat_labels)):
            for j in range(len(all_models)):
                text = ax.text(
                    j,
                    i,
                    usage_matrix[i][j],
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

        ax.set_xlabel("Model")
        ax.set_ylabel("Category")
        ax.set_title("Model Usage Heatmap")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Usage Count", rotation=270, labelpad=15)

        plt.tight_layout()
        return fig

    def run_analysis(
        self,
        show_plots: bool = True,
        save_plots: bool = False,
        timestamp: Optional[str] = None,
    ):
        """Run complete analysis with new visualizations."""
        if not self.load_data():
            return False

        print("\n🎨 Generating new visualizations...")

        # Create all visualization windows
        figures = []

        try:
            figures.append(self.plot_decision_outcomes_stacked())
            figures.append(self.plot_clean_vs_success_scatter())
            figures.append(self.plot_clean_energy_bars())
            figures.append(self.plot_battery_decay_lines())
            figures.append(self.plot_model_usage_heatmap())

            # Save plots if requested
            if save_plots:
                print("💾 Saving plots to files...")
                # Create tree_images directory
                tree_images_dir = Path("tree_images")
                tree_images_dir.mkdir(exist_ok=True)

                # Use provided timestamp or extract from filename
                if timestamp:
                    file_timestamp = timestamp
                else:
                    # Extract timestamp from results filename
                    import re

                    match = re.search(
                        r"tree-search-(\d{8}_\d{6})-metadata", str(self.results_file)
                    )
                    file_timestamp = match.group(1) if match else "unknown"

                for i, fig in enumerate(figures):
                    filename = (
                        tree_images_dir
                        / f"tree_results_{file_timestamp}_window_{i + 1}.png"
                    )
                    fig.savefig(filename, dpi=300, bbox_inches="tight")
                    print(f"   Saved: {filename}")

            # Show plots if requested
            if show_plots:
                print("🖼️  Displaying plots (close windows to continue)...")
                plt.show()

            print("✅ Visualization complete!")

        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return False

        return True

    def print_console_summary(self):
        """Print concise summary with new metrics."""
        print("\n" + "=" * 60)
        print("📊 TREE SEARCH RESULTS SUMMARY")
        print("=" * 60)

        # Metadata
        print(
            f"📍 Location: {self.metadata.get('location', 'Unknown') if self.metadata else 'Unknown'}"
        )
        print(
            f"🌤️  Season: {self.metadata.get('season', 'Unknown') if self.metadata else 'Unknown'}"
        )
        print(
            f"⏱️  Horizon: {self.metadata.get('horizon', 0) if self.metadata else 0} timesteps"
        )
        print(
            f"🔬 Total leaves: {self.metadata.get('total_leaves_explored', 0) if self.metadata else 0}"
        )
        print(
            f"⚡ Runtime: {self.metadata.get('runtime_seconds', 0) if self.metadata else 0:.1f}s"
        )

        # Category comparison table
        print(f"\n📈 CATEGORY PERFORMANCE")
        print("-" * 60)
        print(f"{'Category':<20} {'Clean %':<8} {'Success %':<10} {'Battery':<8}")
        print("-" * 60)

        for cat in self.categories:
            if self.results is None or not self.results[cat]:
                continue

            summary = self.get_category_summary(cat)
            decision_counts = summary.get("decision_counts", {})
            total = summary.get("total_decisions", 1)
            successes = decision_counts.get("success", 0)
            success_rate = (successes / total) * 100 if total > 0 else 0

            print(
                f"{cat.replace('top_', '').replace('_', ' ').title():<20} "
                f"{summary.get('clean_energy_percentage', 0):<8.1f} "
                f"{success_rate:<10.1f} "
                f"{summary.get('final_battery', 0):<8.2f}"
            )

        # Key insights
        print(f"\n💡 KEY INSIGHTS")
        print("-" * 30)

        # Best performers
        best_clean = max(
            [c for c in self.categories if self.results and self.results.get(c)],
            key=lambda x: self.get_category_summary(x).get(
                "clean_energy_percentage", 0
            ),
            default=None,
        )
        best_success = max(
            [c for c in self.categories if self.results and self.results.get(c)],
            key=lambda x: self.get_category_summary(x)
            .get("decision_counts", {})
            .get("success", 0),
            default=None,
        )

        if best_clean:
            print(
                f"🏆 Best Clean E%: {best_clean.replace('top_', '').replace('_', ' ').title()}"
            )
        if best_success:
            print(
                f"🎯 Most Successes: {best_success.replace('top_', '').replace('_', ' ').title()}"
            )

        print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize tree search results from tree_search.py"
    )
    parser.add_argument(
        "--timestamp",
        help="Exact timestamp value from tree search run (e.g., 20251209_230426)",
    )
    parser.add_argument(
        "--file",
        help="Direct path to tree search results JSON file (overrides --timestamp)",
    )
    parser.add_argument("--save", action="store_true", help="Save plots to PNG files")
    parser.add_argument(
        "--no-show", action="store_true", help="Don't display interactive plots"
    )

    args = parser.parse_args()

    # Determine file path
    if args.file:
        results_file = args.file
    elif args.timestamp:
        results_file = f"results/tree-search-{args.timestamp}-metadata.json"
    else:
        # Try to find the most recent tree search file
        import glob

        tree_search_files = glob.glob("results/tree-search-*-metadata.json")
        if tree_search_files:
            tree_search_files.sort(reverse=True)  # Most recent first
            results_file = tree_search_files[0]
            print(f"🔍 Using most recent file: {results_file}")
        else:
            results_file = "results/tree_results.json"  # Fallback to old default

    # Create analyzer and run analysis
    analyzer = TreeResultsAnalyzer(results_file)

    # Pass timestamp to run_analysis
    timestamp_arg = args.timestamp if args.timestamp else None
    if analyzer.run_analysis(
        show_plots=not args.no_show, save_plots=args.save, timestamp=timestamp_arg
    ):
        analyzer.print_console_summary()
    else:
        print("❌ Analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
