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
        self.categories = [
            "top_clean_energy",
            "top_success",
            "top_small_miss",
            "top_large_miss",
            "top_least_energy",
        ]
        self.colors = {
            "top_clean_energy": "#2E8B57",  # Sea Green
            "top_success": "#4169E1",  # Royal Blue
            "top_small_miss": "#FF8C00",  # Dark Orange
            "top_large_miss": "#DC143C",  # Crimson
            "top_least_energy": "#9370DB",  # Medium Purple
        }

    def load_data(self) -> bool:
        """Load and validate tree search results data."""
        try:
            if not self.results_file.exists():
                print(f"❌ Error: Results file not found: {self.results_file}")
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

        # Use first (best) result in category
        result = self.results[category][0]

        return {
            "final_battery": result.get("final_battery", 0),
            "total_energy": result.get("total_energy", 0),
            "clean_energy_percentage": result.get("clean_energy_percentage", 0),
            "decision_counts": result.get("decision_counts", {}),
            "total_decisions": result.get("total_decisions", 0),
        }

    def plot_performance_comparison(self):
        """Window 1: Performance Comparison Dashboard."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Tree Search Results: Category Comparison", fontsize=16, fontweight="bold"
        )

        categories_data = {}
        for cat in self.categories:
            categories_data[cat] = self.get_category_summary(cat)

        # Clean Energy Percentage
        clean_percentages = [
            categories_data[cat].get("clean_energy_percentage", 0)
            for cat in self.categories
        ]
        bars1 = ax1.bar(
            self.categories,
            clean_percentages,
            color=[self.colors[cat] for cat in self.categories],
        )
        ax1.set_title("Clean Energy Percentage")
        ax1.set_ylabel("Clean Energy %")
        ax1.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars1, clean_percentages):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
            )

        # Total Energy Consumption
        total_energies = [
            categories_data[cat].get("total_energy", 0) for cat in self.categories
        ]
        bars2 = ax2.bar(
            self.categories,
            total_energies,
            color=[self.colors[cat] for cat in self.categories],
        )
        ax2.set_title("Total Energy Consumption")
        ax2.set_ylabel("Energy (Wh)")
        ax2.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars2, total_energies):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(total_energies) * 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
            )

        # Final Battery Level
        final_batteries = [
            categories_data[cat].get("final_battery", 0) for cat in self.categories
        ]
        bars3 = ax3.bar(
            self.categories,
            final_batteries,
            color=[self.colors[cat] for cat in self.categories],
        )
        ax3.set_title("Final Battery Level")
        ax3.set_ylabel("Battery (Wh)")
        ax3.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars3, final_batteries):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.2f}",
                ha="center",
                va="bottom",
            )

        # Success Rate
        success_rates = []
        for cat in self.categories:
            decision_counts = categories_data[cat].get("decision_counts", {})
            total = categories_data[cat].get("total_decisions", 1)
            successes = decision_counts.get("success", 0)
            success_rates.append((successes / total) * 100 if total > 0 else 0)

        bars4 = ax4.bar(
            self.categories,
            success_rates,
            color=[self.colors[cat] for cat in self.categories],
        )
        ax4.set_title("Success Rate")
        ax4.set_ylabel("Success Rate %")
        ax4.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars4, success_rates):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        return fig

    def plot_battery_energy_time_series(self):
        """Window 2: Battery & Energy Time Series."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(
            "Battery Dynamics & Energy Consumption", fontsize=16, fontweight="bold"
        )

        for cat in self.categories:
            if self.results is None or not self.results[cat]:
                continue

            result = self.results[cat][0]  # Best result
            ts_data = self.extract_time_series(result)

            # Battery levels
            ax1.plot(
                ts_data["timesteps"],
                ts_data["battery_levels"],
                label=cat.replace("top_", "").replace("_", " ").title(),
                color=self.colors[cat],
                linewidth=2,
            )

        ax1.set_title("Battery Levels Over Time")
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Battery Level (Wh)")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Energy consumption
        for cat in self.categories:
            if self.results is None or not self.results[cat]:
                continue

            result = self.results[cat][0]
            ts_data = self.extract_time_series(result)

            # Total energy (clean + dirty)
            total_energy = [
                c + d for c, d in zip(ts_data["clean_energy"], ts_data["dirty_energy"])
            ]
            ax2.plot(
                ts_data["timesteps"],
                total_energy,
                label=cat.replace("top_", "").replace("_", " ").title(),
                color=self.colors[cat],
                linewidth=2,
            )

        ax2.set_title("Cumulative Energy Consumption")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Total Energy (Wh)")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_model_selection_analysis(self):
        """Window 3: Model Selection Analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Model Selection & Charging Strategies", fontsize=16, fontweight="bold"
        )

        # Model usage frequency for each category
        model_counts = {}
        for cat in self.categories:
            if self.results is None or not self.results[cat]:
                continue

            result = self.results[cat][0]
            ts_data = self.extract_time_series(result)

            model_counts[cat] = {}
            for model in ts_data["models"]:
                model_counts[cat][model] = model_counts[cat].get(model, 0) + 1

        # Pie charts for each category
        for i, cat in enumerate(self.categories[:4]):  # First 4 categories
            ax = [ax1, ax2, ax3, ax4][i]
            if cat in model_counts and model_counts[cat]:
                models = list(model_counts[cat].keys())
                counts = list(model_counts[cat].values())
                colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(models)))

                ax.pie(
                    counts,
                    labels=models,
                    autopct="%1.1f%%",
                    colors=colors,
                    startangle=90,
                )
                ax.set_title(cat.replace("top_", "").replace("_", " ").title())
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(cat.replace("top_", "").replace("_", " ").title())

        plt.tight_layout()
        return fig

    def plot_decision_outcomes(self):
        """Window 4: Decision Outcomes Deep Dive."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Decision Outcome Analysis", fontsize=16, fontweight="bold")

        # Outcome distribution for each category
        outcome_data = {}
        for cat in self.categories:
            if self.results is None or not self.results[cat]:
                continue

            result = self.results[cat][0]
            ts_data = self.extract_time_series(result)

            outcome_counts = {"success": 0, "small_miss": 0, "large_miss": 0}
            for outcome in ts_data["outcomes"]:
                if outcome in outcome_counts:
                    outcome_counts[outcome] += 1
            outcome_data[cat] = outcome_counts

        # Stacked bar chart of outcomes
        categories = list(outcome_data.keys())
        successes = [outcome_data[cat].get("success", 0) for cat in categories]
        small_misses = [outcome_data[cat].get("small_miss", 0) for cat in categories]
        large_misses = [outcome_data[cat].get("large_miss", 0) for cat in categories]

        width = 0.6
        ax1.bar(categories, successes, width, label="Success", color="#2E8B57")
        ax1.bar(
            categories,
            small_misses,
            width,
            bottom=successes,
            label="Small Miss",
            color="#FF8C00",
        )
        ax1.bar(
            categories,
            large_misses,
            width,
            bottom=[s + sm for s, sm in zip(successes, small_misses)],
            label="Large Miss",
            color="#DC143C",
        )

        ax1.set_title("Decision Outcome Distribution")
        ax1.set_ylabel("Count")
        ax1.legend()
        ax1.tick_params(axis="x", rotation=45)

        # Success rate vs final battery scatter
        success_rates = []
        final_batteries = []
        colors_list = []

        for cat in self.categories:
            if cat not in outcome_data:
                continue
            total = sum(outcome_data[cat].values())
            success_rate = (
                (outcome_data[cat].get("success", 0) / total) * 100 if total > 0 else 0
            )
            final_battery = self.get_category_summary(cat).get("final_battery", 0)

            success_rates.append(success_rate)
            final_batteries.append(final_battery)
            colors_list.append(self.colors[cat])

        ax2.scatter(final_batteries, success_rates, c=colors_list, s=100, alpha=0.7)
        ax2.set_xlabel("Final Battery (Wh)")
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_title("Success Rate vs Final Battery")
        ax2.grid(True, alpha=0.3)

        # Add category labels to scatter plot
        for i, cat in enumerate(self.categories):
            if i < len(success_rates):
                ax2.annotate(
                    cat.replace("top_", "").replace("_", " ").title(),
                    (final_batteries[i], success_rates[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        # Outcome timeline for top clean energy category
        if (
            self.results is not None
            and "top_clean_energy" in self.results
            and self.results["top_clean_energy"]
        ):
            result = self.results["top_clean_energy"][0]
            ts_data = self.extract_time_series(result)

            # Convert outcomes to numeric for plotting
            outcome_numeric = []
            for outcome in ts_data["outcomes"]:
                if outcome == "success":
                    outcome_numeric.append(1)
                elif outcome == "small_miss":
                    outcome_numeric.append(0.5)
                else:  # large_miss
                    outcome_numeric.append(0)

            ax3.plot(
                ts_data["timesteps"], outcome_numeric, "o-", markersize=2, linewidth=1
            )
            ax3.set_title("Outcome Timeline (Top Clean Energy)")
            ax3.set_xlabel("Timestep")
            ax3.set_ylabel("Outcome (1=Success, 0.5=Small Miss, 0=Large Miss)")
            ax3.set_ylim(-0.1, 1.1)
            ax3.grid(True, alpha=0.3)

        # Charging events vs outcomes
        if (
            self.results is not None
            and "top_success" in self.results
            and self.results["top_success"]
        ):
            result = self.results["top_success"][0]
            ts_data = self.extract_time_series(result)

            charging_events = [1 if charge else 0 for charge in ts_data["charging"]]
            outcome_numeric = []
            for outcome in ts_data["outcomes"]:
                if outcome == "success":
                    outcome_numeric.append(1)
                elif outcome == "small_miss":
                    outcome_numeric.append(0.5)
                else:
                    outcome_numeric.append(0)

            ax4.scatter(
                ts_data["timesteps"],
                outcome_numeric,
                c=["red" if c else "blue" for c in charging_events],
                alpha=0.6,
                s=10,
            )
            ax4.set_title("Outcomes by Charging (Top Success)")
            ax4.set_xlabel("Timestep")
            ax4.set_ylabel("Outcome")
            ax4.set_ylim(-0.1, 1.1)

            # Add legend
            red_patch = mpatches.Patch(color="red", label="Charging")
            blue_patch = mpatches.Patch(color="blue", label="Not Charging")
            ax4.legend(handles=[red_patch, blue_patch])

        plt.tight_layout()
        return fig

    def plot_energy_efficiency(self):
        """Window 5: Energy Efficiency Focus."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Energy Efficiency Analysis", fontsize=16, fontweight="bold")

        # Clean vs dirty energy accumulation
        for cat in self.categories:
            if self.results is None or not self.results[cat]:
                continue

            result = self.results[cat][0]
            ts_data = self.extract_time_series(result)

            ax1.plot(
                ts_data["timesteps"],
                ts_data["clean_energy"],
                label=f"{cat.replace('top_', '').replace('_', ' ').title()} - Clean",
                color=self.colors[cat],
                linewidth=2,
                linestyle="-",
            )
            ax1.plot(
                ts_data["timesteps"],
                ts_data["dirty_energy"],
                label=f"{cat.replace('top_', '').replace('_', ' ').title()} - Dirty",
                color=self.colors[cat],
                linewidth=2,
                linestyle="--",
            )

        ax1.set_title("Clean vs Dirty Energy Accumulation")
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Energy (Wh)")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Energy per successful inference
        energy_per_success = []
        for cat in self.categories:
            if self.results is None or not self.results[cat]:
                continue

            result = self.results[cat][0]
            summary = self.get_category_summary(cat)
            total_energy = summary.get("total_energy", 0)
            successes = summary.get("decision_counts", {}).get("success", 1)

            energy_per_success.append(total_energy / successes if successes > 0 else 0)

        bars2 = ax2.bar(
            self.categories,
            energy_per_success,
            color=[self.colors[cat] for cat in self.categories],
        )
        ax2.set_title("Energy per Successful Inference")
        ax2.set_ylabel("Energy per Success (Wh)")
        ax2.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars2, energy_per_success):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(energy_per_success) * 0.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
            )

        # Battery depletion rates
        depletion_rates = []
        for cat in self.categories:
            if self.results is None or not self.results[cat]:
                continue

            result = self.results[cat][0]
            ts_data = self.extract_time_series(result)

            if len(ts_data["battery_levels"]) > 1:
                initial_battery = ts_data["battery_levels"][0]
                final_battery = ts_data["battery_levels"][-1]
                depletion_rate = (initial_battery - final_battery) / len(
                    ts_data["battery_levels"]
                )
                depletion_rates.append(depletion_rate)
            else:
                depletion_rates.append(0)

        bars3 = ax3.bar(
            self.categories,
            depletion_rates,
            color=[self.colors[cat] for cat in self.categories],
        )
        ax3.set_title("Battery Depletion Rate")
        ax3.set_ylabel("Battery Loss per Timestep (Wh)")
        ax3.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars3, depletion_rates):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(depletion_rates) * 0.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
            )

        # Clean energy percentage vs total energy
        clean_percentages = []
        total_energies = []
        for cat in self.categories:
            summary = self.get_category_summary(cat)
            clean_percentages.append(summary.get("clean_energy_percentage", 0))
            total_energies.append(summary.get("total_energy", 0))

        scatter = ax4.scatter(
            total_energies,
            clean_percentages,
            c=[self.colors[cat] for cat in self.categories],
            s=100,
            alpha=0.7,
        )
        ax4.set_xlabel("Total Energy (Wh)")
        ax4.set_ylabel("Clean Energy Percentage (%)")
        ax4.set_title("Clean Energy % vs Total Energy")
        ax4.grid(True, alpha=0.3)

        # Add category labels
        for i, cat in enumerate(self.categories):
            ax4.annotate(
                cat.replace("top_", "").replace("_", " ").title(),
                (total_energies[i], clean_percentages[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        plt.tight_layout()
        return fig

    def print_console_summary(self):
        """Print comprehensive console summary with insights."""
        print("\n" + "=" * 80)
        print("📊 TREE SEARCH RESULTS ANALYSIS SUMMARY")
        print("=" * 80)

        # Metadata
        print(
            f"\n📍 Location: {self.metadata.get('location', 'Unknown') if self.metadata else 'Unknown'}"
        )
        print(
            f"🌤️  Season: {self.metadata.get('season', 'Unknown') if self.metadata else 'Unknown'}"
        )
        print(
            f"⏱️  Horizon: {self.metadata.get('horizon', 0) if self.metadata else 0} timesteps"
        )
        print(
            f"🔬 Total leaves explored: {self.metadata.get('total_leaves_explored', 0) if self.metadata else 0}"
        )
        print(
            f"⚡ Runtime: {self.metadata.get('runtime_seconds', 0) if self.metadata else 0:.2f} seconds"
        )
        print(
            f"🎯 Beam search enabled: {self.metadata.get('beam_search_enabled', False) if self.metadata else False}"
        )

        # Category comparison table
        print(f"\n📈 CATEGORY PERFORMANCE COMPARISON")
        print("-" * 80)
        print(
            f"{'Category':<20} {'Clean %':<10} {'Total E':<10} {'Final B':<10} {'Success %':<10}"
        )
        print("-" * 80)

        for cat in self.categories:
            summary = self.get_category_summary(cat)
            decision_counts = summary.get("decision_counts", {})
            total = summary.get("total_decisions", 1)
            successes = decision_counts.get("success", 0)
            success_rate = (successes / total) * 100 if total > 0 else 0

            print(
                f"{cat.replace('top_', '').replace('_', ' ').title():<20} "
                f"{summary.get('clean_energy_percentage', 0):<10.1f} "
                f"{summary.get('total_energy', 0):<10.3f} "
                f"{summary.get('final_battery', 0):<10.2f} "
                f"{success_rate:<10.1f}"
            )

        # Key insights
        print(f"\n💡 KEY INSIGHTS")
        print("-" * 40)

        # Best performers
        best_clean = max(
            self.categories,
            key=lambda x: self.get_category_summary(x).get(
                "clean_energy_percentage", 0
            ),
        )
        best_success = max(
            self.categories,
            key=lambda x: self.get_category_summary(x)
            .get("decision_counts", {})
            .get("success", 0),
        )
        most_efficient = min(
            self.categories,
            key=lambda x: self.get_category_summary(x).get(
                "total_energy", float("inf")
            ),
        )

        print(
            f"🏆 Best Clean Energy: {best_clean.replace('top_', '').replace('_', ' ').title()}"
        )
        print(
            f"🎯 Most Successful: {best_success.replace('top_', '').replace('_', ' ').title()}"
        )
        print(
            f"⚡ Most Energy Efficient: {most_efficient.replace('top_', '').replace('_', ' ').title()}"
        )

        # Decision patterns
        print(f"\n📊 DECISION PATTERNS")
        print("-" * 40)

        total_successes = 0
        total_small_misses = 0
        total_large_misses = 0

        for cat in self.categories:
            summary = self.get_category_summary(cat)
            decision_counts = summary.get("decision_counts", {})
            total_successes += decision_counts.get("success", 0)
            total_small_misses += decision_counts.get("small_miss", 0)
            total_large_misses += decision_counts.get("large_miss", 0)

        total_decisions = total_successes + total_small_misses + total_large_misses
        if total_decisions > 0:
            print(
                f"✅ Overall Success Rate: {(total_successes / total_decisions) * 100:.1f}%"
            )
            print(
                f"⚠️  Small Miss Rate: {(total_small_misses / total_decisions) * 100:.1f}%"
            )
            print(
                f"❌ Large Miss Rate: {(total_large_misses / total_decisions) * 100:.1f}%"
            )

        print("\n" + "=" * 80)

    def run_analysis(self, show_plots: bool = True, save_plots: bool = False):
        """Run complete analysis with all visualizations."""
        if not self.load_data():
            return False

        print("\n🎨 Generating visualizations...")

        # Create all visualization windows
        figures = []

        try:
            figures.append(self.plot_performance_comparison())
            figures.append(self.plot_battery_energy_time_series())
            figures.append(self.plot_model_selection_analysis())
            figures.append(self.plot_decision_outcomes())
            figures.append(self.plot_energy_efficiency())

            # Save plots if requested
            if save_plots:
                print("💾 Saving plots to files...")
                # Create tree_images directory
                tree_images_dir = Path("tree_images")
                tree_images_dir.mkdir(exist_ok=True)

                for i, fig in enumerate(figures):
                    filename = tree_images_dir / f"tree_results_window_{i + 1}.png"
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize tree search results from tree_search.py"
    )
    parser.add_argument(
        "--file",
        default="results/tree_results.json",
        help="Path to tree search results JSON file",
    )
    parser.add_argument("--save", action="store_true", help="Save plots to PNG files")
    parser.add_argument(
        "--no-show", action="store_true", help="Don't display interactive plots"
    )

    args = parser.parse_args()

    # Create analyzer and run analysis
    analyzer = TreeResultsAnalyzer(args.file)

    if analyzer.run_analysis(show_plots=not args.no_show, save_plots=args.save):
        analyzer.print_console_summary()
    else:
        print("❌ Analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
