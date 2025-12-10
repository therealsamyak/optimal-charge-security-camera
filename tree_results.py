#!/usr/bin/env python3
"""
Tree Search Results Visualization Script
Parses and visualizes tree search results from tree_search.py output.
Creates separate matplotlib windows for different analysis perspectives.
"""

import json
import argparse
import sys
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt


class TreeResultsAnalyzer:
    """Analyzes and visualizes tree search results."""

    def __init__(self, results_file: str, aggregate_seasons: bool = False):
        """Initialize analyzer with results file path."""
        self.results_file = Path(results_file)
        self.aggregate_seasons = aggregate_seasons
        self.data: Optional[Dict] = None
        self.metadata: Optional[Dict] = None
        self.results: Optional[Dict] = None
        self.seasonal_data: Dict[str, Dict] = {}  # Individual season data
        self.aggregated_data: Optional[Dict] = None  # Aggregated across seasons
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

    def extract_timestamp_and_season(
        self, filename: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract timestamp and season from filename."""
        pattern = r"([A-Z]{2})-(\d{8}_\d{6})-(\w+)-metadata\.json"
        match = re.search(pattern, filename)
        if match:
            return match.group(2), match.group(3)  # timestamp, season
        return None, None

    def _extract_timestamp_from_file(self) -> str:
        """Extract timestamp from the results file name."""
        match = re.search(
            r"[A-Z]{2}-(\d{8}_\d{6})-(?:\w+)-metadata", str(self.results_file)
        )
        return match.group(1) if match else "unknown"

    def find_seasonal_files(self) -> Dict[str, Path]:
        """Find all seasonal files with same timestamp and location."""
        if not self.results_file.exists():
            return {}

        # Extract timestamp and season from current file
        timestamp, current_season = self.extract_timestamp_and_season(
            self.results_file.name
        )
        if not timestamp:
            return {}

        # Find all tree search files
        all_files = glob.glob("results/*-metadata.json")

        seasonal_files = {}
        for file_path in all_files:
            file_timestamp, season = self.extract_timestamp_and_season(
                Path(file_path).name
            )
            if file_timestamp == timestamp and season:
                seasonal_files[season] = Path(file_path)

        return seasonal_files

    def load_seasonal_data(self) -> bool:
        """Load all seasonal data files."""
        if not self.aggregate_seasons:
            return True

        seasonal_files = self.find_seasonal_files()
        if not seasonal_files:
            print("⚠️  No seasonal files found for aggregation")
            return False

        print(f"🔄 Found seasonal files: {list(seasonal_files.keys())}")

        # Load each seasonal file
        for season, file_path in seasonal_files.items():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    self.seasonal_data[season] = {
                        "data": data,
                        "metadata": data.get("metadata", {}),
                        "results": data.get("results", {}),
                    }
                print(f"✅ Loaded {season} data from {file_path.name}")
            except Exception as e:
                print(f"❌ Error loading {season} data: {e}")
                return False

        # Create aggregated data
        return self.aggregate_seasonal_results()

    def aggregate_seasonal_results(self) -> bool:
        """Aggregate results across all seasons."""
        if not self.seasonal_data:
            return False

        seasons = list(self.seasonal_data.keys())
        print(f"📊 Aggregating data across {len(seasons)} seasons: {seasons}")

        # Initialize aggregated structure
        self.aggregated_data = {
            "metadata": self.seasonal_data[seasons[0]]["metadata"].copy(),
            "results": {},
        }

        # Remove season from metadata for aggregated view
        if "season" in self.aggregated_data["metadata"]:
            del self.aggregated_data["metadata"]["season"]

        # Aggregate each category
        for category in self.base_categories:
            aggregated_category = self._aggregate_category(category, seasons)
            if aggregated_category:
                if self.aggregated_data["results"] is None:
                    self.aggregated_data["results"] = {}
                self.aggregated_data["results"][category] = aggregated_category

        print("✅ Seasonal aggregation complete")
        return True

    def _aggregate_category(self, category: str, seasons: List[str]) -> Optional[Dict]:
        """Aggregate a single category across seasons."""
        category_results = []

        for season in seasons:
            season_results = self.seasonal_data[season]["results"]
            if category in season_results and season_results[category]:
                if category == "naive":
                    category_results.append(season_results[category])
                else:
                    # Take best result from list for top_* categories
                    category_results.append(season_results[category][0])

        if not category_results:
            return None

        # Aggregate metrics
        aggregated = {
            "final_battery": np.mean(
                [r.get("final_battery", 0) for r in category_results]
            ),
            "total_energy": np.mean(
                [r.get("total_energy", 0) for r in category_results]
            ),
            "clean_energy_percentage": np.mean(
                [r.get("clean_energy_percentage", 0) for r in category_results]
            ),
            "decision_counts": {},
            "total_decisions": sum(
                [r.get("total_decisions", 0) for r in category_results]
            ),
        }

        # Aggregate decision counts (sum across seasons)
        for outcome_type in ["success", "small_miss", "large_miss"]:
            aggregated["decision_counts"][outcome_type] = sum(
                [
                    r.get("decision_counts", {}).get(outcome_type, 0)
                    for r in category_results
                ]
            )

        # Aggregate action sequences (average each timestep)
        if category_results[0].get("action_sequence"):
            max_length = max(
                len(r.get("action_sequence", [])) for r in category_results
            )
            aggregated_action_sequence = []

            for timestep in range(max_length):
                timestep_actions = []
                for result in category_results:
                    action_seq = result.get("action_sequence", [])
                    if timestep < len(action_seq):
                        timestep_actions.append(action_seq[timestep])

                if timestep_actions:
                    # Average numeric fields, keep categorical from first
                    avg_action = {
                        "timestep": timestep,
                        "model": timestep_actions[0].get("model", "Unknown"),
                        "charged": timestep_actions[0].get("charged", False),
                        "outcome": timestep_actions[0].get("outcome", "unknown"),
                        "battery_before": np.mean(
                            [a.get("battery_before", 0) for a in timestep_actions]
                        ),
                        "battery_after": np.mean(
                            [a.get("battery_after", 0) for a in timestep_actions]
                        ),
                        "clean_energy_before": np.mean(
                            [a.get("clean_energy_before", 0) for a in timestep_actions]
                        ),
                        "clean_energy_after": np.mean(
                            [a.get("clean_energy_after", 0) for a in timestep_actions]
                        ),
                        "dirty_energy_before": np.mean(
                            [a.get("dirty_energy_before", 0) for a in timestep_actions]
                        ),
                        "dirty_energy_after": np.mean(
                            [a.get("dirty_energy_after", 0) for a in timestep_actions]
                        ),
                        "charge_energy": np.mean(
                            [a.get("charge_energy", 0) for a in timestep_actions]
                        ),
                        "model_energy": np.mean(
                            [a.get("model_energy", 0) for a in timestep_actions]
                        ),
                        "clean_energy_pct": np.mean(
                            [a.get("clean_energy_pct", 0) for a in timestep_actions]
                        ),
                    }
                    aggregated_action_sequence.append(avg_action)

            aggregated["action_sequence"] = aggregated_action_sequence

        return aggregated

    def get_seasonal_summary(self, category: str, season: str) -> Dict:
        """Get summary statistics for a specific season and category."""
        if (
            season not in self.seasonal_data
            or category not in self.seasonal_data[season]["results"]
            or not self.seasonal_data[season]["results"][category]
        ):
            return {}

        # Handle both list (top_*) and dict (naive) result types
        if category == "naive":
            result = self.seasonal_data[season]["results"][category]
        else:
            result = self.seasonal_data[season]["results"][category][0]

        return {
            "final_battery": result.get("final_battery", 0),
            "total_energy": result.get("total_energy", 0),
            "clean_energy_percentage": result.get("clean_energy_percentage", 0),
            "decision_counts": result.get("decision_counts", {}),
            "total_decisions": result.get("total_decisions", 0),
        }

    def get_all_seasons(self) -> List[str]:
        """Get list of all available seasons plus aggregated."""
        if not self.aggregate_seasons or not self.seasonal_data:
            return []

        seasons = list(self.seasonal_data.keys())
        seasons.append("aggregated")
        return seasons

    def get_text_color(self, value: float, max_value: float) -> str:
        """Determine text color based on background intensity."""
        if max_value == 0:
            return "black"
        intensity = value / max_value
        return "white" if intensity > 0.6 else "black"

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

            # Load seasonal data if aggregation is requested
            if self.aggregate_seasons:
                if not self.load_seasonal_data():
                    return False
                # Use aggregated data for analysis
                if self.aggregated_data:
                    self.data = self.aggregated_data
                    self.metadata = self.aggregated_data.get("metadata", {})
                    self.results = self.aggregated_data.get("results", {})

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

            if self.aggregate_seasons:
                print(
                    f"✅ Loaded AGGREGATED results for {self.metadata.get('location', 'Unknown') if self.metadata else 'Unknown'} across {len(self.seasonal_data)} seasons"
                )
            else:
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
            result_data = self.results[category]
            # For aggregated data, result_data is already a dict
            # For seasonal data, result_data is a list
            if isinstance(result_data, list) and len(result_data) > 0:
                result = result_data[0]
            elif isinstance(result_data, dict):
                result = result_data
            else:
                return {}

        return {
            "final_battery": result.get("final_battery", 0),
            "total_energy": result.get("total_energy", 0),
            "clean_energy_percentage": result.get("clean_energy_percentage", 0),
            "decision_counts": result.get("decision_counts", {}),
            "total_decisions": result.get("total_decisions", 0),
        }

    def plot_decision_outcomes_stacked(self):
        """Chart 1: Stacked bar graph of decision outcomes for all categories."""
        if self.aggregate_seasons and self.seasonal_data:
            return self._plot_seasonal_decision_outcomes()
        else:
            return self._plot_single_decision_outcomes()

    def _plot_single_decision_outcomes(self):
        """Original decision outcomes chart for single season."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create title with location, timestamp, and season
        location = (
            self.metadata.get("location", "Unknown") if self.metadata else "Unknown"
        )
        season = self.metadata.get("season", "Unknown") if self.metadata else "Unknown"
        timestamp = self._extract_timestamp_from_file()
        title = f"Decision Outcomes - {location} {timestamp} {season.title()}"
        fig.suptitle(title, fontsize=14, fontweight="bold")

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

    def _plot_seasonal_decision_outcomes(self):
        """Seasonal comparison decision outcomes chart - separate figures for each season."""
        seasons = self.get_all_seasons()
        figures = []

        for season in seasons:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get data for this season
            if season == "aggregated":
                season_results = self.results
                season_display = "Aggregated"
            else:
                season_results = self.seasonal_data[season]["results"]
                season_display = season.title()

            # Create title with location, timestamp, and season
            location = (
                self.metadata.get("location", "Unknown") if self.metadata else "Unknown"
            )
            timestamp = self._extract_timestamp_from_file()
            title = f"Decision Outcomes - {location} {timestamp} {season_display}"
            fig.suptitle(title, fontsize=14, fontweight="bold")

            categories = self.categories
            success_counts = []
            small_miss_counts = []
            large_miss_counts = []

            for cat in categories:
                if season_results is None or not season_results.get(cat):
                    success_counts.append(0)
                    small_miss_counts.append(0)
                    large_miss_counts.append(0)
                    continue

                # Handle both list (top_*) and dict (naive) result types
                if cat == "naive":
                    result = season_results[cat]
                else:
                    result_data = season_results[cat]
                    # For aggregated data, result_data is already a dict
                    # For seasonal data, result_data is a list
                    if isinstance(result_data, list) and len(result_data) > 0:
                        result = result_data[0]
                    elif isinstance(result_data, dict):
                        result = result_data
                    else:
                        continue

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
                [
                    cat.replace("top_", "").replace("_", " ").title()
                    for cat in categories
                ],
                rotation=45,
                ha="right",
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            figures.append(fig)

        return figures

    def plot_clean_vs_success_scatter(self):
        """Chart 2: Clean energy % vs success % scatter plot."""
        if self.aggregate_seasons and self.seasonal_data:
            return self._plot_seasonal_clean_vs_success()
        else:
            return self._plot_single_clean_vs_success()

    def _plot_single_clean_vs_success(self):
        """Original clean vs success chart for single season."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create title with location, timestamp, and season
        location = (
            self.metadata.get("location", "Unknown") if self.metadata else "Unknown"
        )
        season = self.metadata.get("season", "Unknown") if self.metadata else "Unknown"
        timestamp = self._extract_timestamp_from_file()
        title = f"Clean Energy vs Success Tradeoff - {location} {timestamp} {season.title()}"
        fig.suptitle(title, fontsize=14, fontweight="bold")

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
            c
            for c in self.categories
            if self.results and self.results.get(c) is not None
        ]
        colors = [self.colors.get(cat, "#808080") for cat in valid_cats]

        if len(clean_pcts) > 0 and len(success_pcts) > 0:
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

    def _plot_seasonal_clean_vs_success(self):
        """Seasonal comparison clean vs success scatter plot - separate figures for each season."""
        seasons = self.get_all_seasons()
        figures = []

        for season in seasons:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get data for this season
            if season == "aggregated":
                season_results = self.results
                season_display = "Aggregated"
            else:
                season_results = self.seasonal_data[season]["results"]
                season_display = season.title()

            # Create title with location, timestamp, and season
            location = (
                self.metadata.get("location", "Unknown") if self.metadata else "Unknown"
            )
            timestamp = self._extract_timestamp_from_file()
            title = f"Clean Energy vs Success Tradeoff - {location} {timestamp} {season_display}"
            fig.suptitle(title, fontsize=14, fontweight="bold")

            clean_pcts = []
            success_pcts = []

            for cat in self.categories:
                if season_results is None or not season_results.get(cat):
                    continue

                # Handle both list (top_*) and dict (naive) result types
                if cat == "naive":
                    result = season_results[cat]
                else:
                    result_data = season_results[cat]
                    # For aggregated data, result_data is already a dict
                    # For seasonal data, result_data is a list
                    if isinstance(result_data, list) and len(result_data) > 0:
                        result = result_data[0]
                    elif isinstance(result_data, dict):
                        result = result_data
                    else:
                        continue

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
                c
                for c in self.categories
                if season_results and season_results.get(c) is not None
            ]
            colors = [self.colors.get(cat, "#808080") for cat in valid_cats]

            if len(clean_pcts) > 0 and len(success_pcts) > 0:
                scatter = ax.scatter(
                    clean_pcts,
                    success_pcts,
                    c=colors,
                    s=100,
                    alpha=0.8,
                    edgecolors="black",
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
            figures.append(fig)

        return figures

    def plot_clean_energy_bars(self):
        """Chart 3: Simple bar chart of clean energy percentage."""
        if self.aggregate_seasons and self.seasonal_data:
            return self._plot_seasonal_clean_energy_bars()
        else:
            return self._plot_single_clean_energy_bars()

    def _plot_single_clean_energy_bars(self):
        """Original clean energy bars chart for single season."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create title with location, timestamp, and season
        location = (
            self.metadata.get("location", "Unknown") if self.metadata else "Unknown"
        )
        season = self.metadata.get("season", "Unknown") if self.metadata else "Unknown"
        timestamp = self._extract_timestamp_from_file()
        title = f"Clean Energy Percentage - {location} {timestamp} {season.title()}"
        fig.suptitle(title, fontsize=14, fontweight="bold")

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

    def _plot_seasonal_clean_energy_bars(self):
        """Seasonal comparison clean energy bars chart - separate figures for each season."""
        seasons = self.get_all_seasons()
        figures = []

        for season in seasons:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get data for this season
            if season == "aggregated":
                season_results = self.results
                season_display = "Aggregated"
            else:
                season_results = self.seasonal_data[season]["results"]
                season_display = season.title()

            # Create title with location, timestamp, and season
            location = (
                self.metadata.get("location", "Unknown") if self.metadata else "Unknown"
            )
            timestamp = self._extract_timestamp_from_file()
            title = f"Clean Energy Percentage - {location} {timestamp} {season_display}"
            fig.suptitle(title, fontsize=14, fontweight="bold")

            clean_pcts = []
            cat_labels = []

            for cat in self.categories:
                if season_results is None or not season_results.get(cat):
                    continue

                # Handle both list (top_*) and dict (naive) result types
                if cat == "naive":
                    result = season_results[cat]
                else:
                    result_data = season_results[cat]
                    # For aggregated data, result_data is already a dict
                    # For seasonal data, result_data is a list
                    if isinstance(result_data, list) and len(result_data) > 0:
                        result = result_data[0]
                    elif isinstance(result_data, dict):
                        result = result_data
                    else:
                        continue

                clean_pcts.append(result.get("clean_energy_percentage", 0))
                cat_labels.append(cat.replace("top_", "").replace("_", " ").title())

            colors = [
                self.colors.get(cat, "#808080")
                for cat in self.categories
                if season_results and season_results.get(cat)
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
            figures.append(fig)

        return figures

    def plot_battery_decay_lines(self):
        """Chart 4: Line chart showing battery decay over time."""
        if self.aggregate_seasons and self.seasonal_data:
            return self._plot_seasonal_battery_decay()
        else:
            return self._plot_single_battery_decay()

    def _plot_single_battery_decay(self):
        """Original battery decay chart for single season."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create title with location, timestamp, and season
        location = (
            self.metadata.get("location", "Unknown") if self.metadata else "Unknown"
        )
        season = self.metadata.get("season", "Unknown") if self.metadata else "Unknown"
        timestamp = self._extract_timestamp_from_file()
        title = f"Battery Levels Over Time - {location} {timestamp} {season.title()}"
        fig.suptitle(title, fontsize=14, fontweight="bold")

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

    def _plot_seasonal_battery_decay(self):
        """Seasonal comparison battery decay chart - separate figures for each season."""
        seasons = self.get_all_seasons()
        figures = []

        for season in seasons:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Get data for this season
            if season == "aggregated":
                season_results = self.results
                season_display = "Aggregated"
            else:
                season_results = self.seasonal_data[season]["results"]
                season_display = season.title()

            # Create title with location, timestamp, and season
            location = (
                self.metadata.get("location", "Unknown") if self.metadata else "Unknown"
            )
            timestamp = self._extract_timestamp_from_file()
            title = (
                f"Battery Levels Over Time - {location} {timestamp} {season_display}"
            )
            fig.suptitle(title, fontsize=14, fontweight="bold")

            for cat in self.categories:
                if season_results is None or not season_results.get(cat):
                    continue

                # Handle both list (top_*) and dict (naive) result types
                if cat == "naive":
                    result = season_results[cat]
                else:
                    result_data = season_results[cat]
                    # For aggregated data, result_data is already a dict
                    # For seasonal data, result_data is a list
                    if isinstance(result_data, list) and len(result_data) > 0:
                        result = result_data[0]
                    elif isinstance(result_data, dict):
                        result = result_data
                    else:
                        continue

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
            figures.append(fig)

        return figures

    def plot_model_usage_heatmap(self):
        """Chart 5: Heatmap showing model usage frequency per category."""
        if self.aggregate_seasons and self.seasonal_data:
            return self._plot_seasonal_model_usage_heatmap()
        else:
            return self._plot_single_model_usage_heatmap()

    def _plot_single_model_usage_heatmap(self):
        """Original model usage heatmap for single season."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create title with location, timestamp, and season
        location = (
            self.metadata.get("location", "Unknown") if self.metadata else "Unknown"
        )
        season = self.metadata.get("season", "Unknown") if self.metadata else "Unknown"
        timestamp = self._extract_timestamp_from_file()
        title = f"Model Usage Frequency - {location} {timestamp} {season.title()}"
        fig.suptitle(title, fontsize=14, fontweight="bold")

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

        # Add text annotations with improved contrast
        max_value = max(max(row) for row in usage_matrix) if usage_matrix else 1
        for i in range(len(cat_labels)):
            for j in range(len(all_models)):
                text_color = self.get_text_color(usage_matrix[i][j], max_value)
                text = ax.text(
                    j,
                    i,
                    usage_matrix[i][j],
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_xlabel("Model")
        ax.set_ylabel("Category")
        ax.set_title("Model Usage Heatmap")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Usage Count", rotation=270, labelpad=15)

        plt.tight_layout()
        return fig

    def _plot_seasonal_model_usage_heatmap(self):
        """Seasonal comparison model usage heatmap - separate figures for each season."""
        seasons = self.get_all_seasons()
        figures = []

        for season in seasons:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Get data for this season
            if season == "aggregated":
                season_results = self.results
                season_display = "Aggregated"
            else:
                season_results = self.seasonal_data[season]["results"]
                season_display = season.title()

            # Create title with location, timestamp, and season
            location = (
                self.metadata.get("location", "Unknown") if self.metadata else "Unknown"
            )
            timestamp = self._extract_timestamp_from_file()
            title = f"Model Usage Frequency - {location} {timestamp} {season_display}"
            fig.suptitle(title, fontsize=14, fontweight="bold")

            # Collect all unique models across categories
            all_models = set()
            for cat in self.categories:
                if season_results is None or not season_results.get(cat):
                    continue

                # Handle both list (top_*) and dict (naive) result types
                if cat == "naive":
                    result = season_results[cat]
                else:
                    result_data = season_results[cat]
                    # For aggregated data, result_data is already a dict
                    # For seasonal data, result_data is a list
                    if isinstance(result_data, list) and len(result_data) > 0:
                        result = result_data[0]
                    elif isinstance(result_data, dict):
                        result = result_data
                    else:
                        continue

                ts_data = self.extract_time_series(result)
                all_models.update(ts_data["models"])

            all_models = sorted(list(all_models))

            # Create usage matrix
            usage_matrix = []
            cat_labels = []

            for cat in self.categories:
                if season_results is None or not season_results.get(cat):
                    continue

                cat_labels.append(cat.replace("top_", "").replace("_", " ").title())

                # Handle both list (top_*) and dict (naive) result types
                if cat == "naive":
                    result = season_results[cat]
                else:
                    result_data = season_results[cat]
                    # For aggregated data, result_data is already a dict
                    # For seasonal data, result_data is a list
                    if isinstance(result_data, list) and len(result_data) > 0:
                        result = result_data[0]
                    elif isinstance(result_data, dict):
                        result = result_data
                    else:
                        continue

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

            # Add text annotations with improved contrast
            max_value = max(max(row) for row in usage_matrix) if usage_matrix else 1
            for i in range(len(cat_labels)):
                for j in range(len(all_models)):
                    text_color = self.get_text_color(usage_matrix[i][j], max_value)
                    text = ax.text(
                        j,
                        i,
                        usage_matrix[i][j],
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=9,
                        fontweight="bold",
                    )

            ax.set_xlabel("Model")
            ax.set_ylabel("Category")
            ax.set_title("Model Usage Heatmap")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Usage Count", rotation=270, labelpad=15)

            plt.tight_layout()
            figures.append(fig)

        return figures

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
            # Get figures from each plotting method
            decision_figures = self.plot_decision_outcomes_stacked()
            clean_vs_success_figures = self.plot_clean_vs_success_scatter()
            clean_energy_figures = self.plot_clean_energy_bars()
            battery_figures = self.plot_battery_decay_lines()
            heatmap_figures = self.plot_model_usage_heatmap()

            # Flatten all figures into a single list
            if isinstance(decision_figures, list):
                figures.extend(decision_figures)
            else:
                figures.append(decision_figures)

            if isinstance(clean_vs_success_figures, list):
                figures.extend(clean_vs_success_figures)
            else:
                figures.append(clean_vs_success_figures)

            if isinstance(clean_energy_figures, list):
                figures.extend(clean_energy_figures)
            else:
                figures.append(clean_energy_figures)

            if isinstance(battery_figures, list):
                figures.extend(battery_figures)
            else:
                figures.append(battery_figures)

            if isinstance(heatmap_figures, list):
                figures.extend(heatmap_figures)
            else:
                figures.append(heatmap_figures)

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
                    match = re.search(
                        r"[A-Z]{2}-(\d{8}_\d{6})-(?:\w+)-metadata",
                        str(self.results_file),
                    )
                    file_timestamp = match.group(1) if match else "unknown"

                # Get location for naming
                location = (
                    self.metadata.get("location", "Unknown")
                    if self.metadata
                    else "Unknown"
                )

                chart_types = [
                    "decision_outcomes",
                    "clean_vs_success",
                    "clean_energy",
                    "battery_decay",
                    "model_usage",
                ]
                seasons = (
                    self.get_all_seasons()
                    if self.aggregate_seasons and self.seasonal_data
                    else [
                        self.metadata.get("season", "unknown")
                        if self.metadata
                        else "unknown"
                    ]
                )

                chart_idx = 0
                for chart_type in chart_types:
                    for season_idx, season in enumerate(seasons):
                        if chart_idx < len(figures):
                            season_display = (
                                "aggregated" if season == "aggregated" else season
                            )
                            filename = (
                                tree_images_dir
                                / f"{location}_{file_timestamp}_{season_display}_{chart_type}.png"
                            )
                            figures[chart_idx].savefig(
                                filename, dpi=300, bbox_inches="tight"
                            )
                            print(f"   Saved: {filename}")
                            chart_idx += 1

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
        if self.aggregate_seasons and self.seasonal_data:
            self._print_seasonal_console_summary()
        else:
            self._print_single_console_summary()

    def _print_single_console_summary(self):
        """Original console summary for single season."""
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

    def _print_seasonal_console_summary(self):
        """Enhanced console summary for seasonal data."""
        print("\n" + "=" * 80)
        print("📊 SEASONAL TREE SEARCH RESULTS SUMMARY")
        print("=" * 80)

        # Metadata
        location = (
            self.metadata.get("location", "Unknown") if self.metadata else "Unknown"
        )
        print(f"📍 Location: {location}")
        print(f"🌤️  Seasons: {', '.join(self.seasonal_data.keys())}")
        print(
            f"⏱️  Horizon: {self.metadata.get('horizon', 0) if self.metadata else 0} timesteps"
        )
        print(
            f"🔬 Total leaves: {self.metadata.get('total_leaves_explored', 0) if self.metadata else 0}"
        )

        # Seasonal comparison table
        print(f"\n📈 SEASONAL PERFORMANCE COMPARISON")
        print("-" * 80)
        print(
            f"{'Season':<12} {'Category':<20} {'Clean %':<8} {'Success %':<10} {'Battery':<8}"
        )
        print("-" * 80)

        seasons = list(self.seasonal_data.keys())
        seasons.append("aggregated")

        for season in seasons:
            for cat in self.categories:
                if season == "aggregated":
                    if self.results is None or not self.results.get(cat):
                        continue
                    summary = self.get_category_summary(cat)
                else:
                    summary = self.get_seasonal_summary(cat, season)
                    if not summary:
                        continue

                decision_counts = summary.get("decision_counts", {})
                total = summary.get("total_decisions", 1)
                successes = decision_counts.get("success", 0)
                success_rate = (successes / total) * 100 if total > 0 else 0

                season_display = (
                    season.title() if season != "aggregated" else "AGGREGATED"
                )
                cat_display = cat.replace("top_", "").replace("_", " ").title()

                print(
                    f"{season_display:<12} {cat_display:<20} "
                    f"{summary.get('clean_energy_percentage', 0):<8.1f} "
                    f"{success_rate:<10.1f} "
                    f"{summary.get('final_battery', 0):<8.2f}"
                )

        # Key insights
        print(f"\n💡 SEASONAL INSIGHTS")
        print("-" * 40)

        # Best performers by season
        for season in seasons:
            season_display = season.title() if season != "aggregated" else "AGGREGATED"

            # Find best clean energy for this season
            if season == "aggregated":
                candidates = [
                    c for c in self.categories if self.results and self.results.get(c)
                ]
                best_clean = max(
                    candidates,
                    key=lambda x: self.get_category_summary(x).get(
                        "clean_energy_percentage", 0
                    ),
                    default=None,
                )
            else:
                candidates = [
                    c
                    for c in self.categories
                    if self.seasonal_data[season]["results"].get(c)
                ]
                best_clean = max(
                    candidates,
                    key=lambda x: self.get_seasonal_summary(x, season).get(
                        "clean_energy_percentage", 0
                    ),
                    default=None,
                )

            if best_clean:
                cat_display = best_clean.replace("top_", "").replace("_", " ").title()
                print(f"🏆 {season_display:<12} Best Clean E%: {cat_display}")

        print("\n" + "=" * 80)


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
    parser.add_argument(
        "--aggregate-seasons",
        action="store_true",
        help="Aggregate data across all seasons for same timestamp/location",
    )

    args = parser.parse_args()

    # Determine file path
    if args.file:
        results_file = args.file
    elif args.timestamp:
        results_file = f"results/*-{args.timestamp}-*-metadata.json"
    else:
        # Try to find the most recent tree search file
        tree_search_files = glob.glob("results/*-metadata.json")
        if tree_search_files:
            tree_search_files.sort(reverse=True)  # Most recent first
            results_file = tree_search_files[0]
            print(f"🔍 Using most recent file: {results_file}")
        else:
            results_file = "results/tree_results.json"  # Fallback to old default

    # Create analyzer and run analysis
    analyzer = TreeResultsAnalyzer(
        results_file, aggregate_seasons=args.aggregate_seasons
    )

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
