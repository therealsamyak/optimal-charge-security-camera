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
        ]

        self.colors = {
            "top_most_clean_energy": "#2E8B57",  # Sea Green
            "top_success": "#4169E1",  # Royal Blue
            "top_success_small_miss": "#FFD700",  # Gold
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
            "Model Selection & Energy-Accuracy Tradeoffs",
            fontsize=16,
            fontweight="bold",
        )

        # Collect all model usage data across categories
        all_model_data = {}
        for cat in self.categories:
            if self.results is None or not self.results[cat]:
                continue

            result = self.results[cat][0]
            ts_data = self.extract_time_series(result)

            # Track model usage with outcomes
            for i, model in enumerate(ts_data["models"]):
                if model not in all_model_data:
                    all_model_data[model] = {
                        "usage_count": 0,
                        "success_count": 0,
                        "total_energy": 0,
                        "categories": set(),
                    }

                all_model_data[model]["usage_count"] += 1
                all_model_data[model]["categories"].add(cat)

                if ts_data["outcomes"][i] == "success":
                    all_model_data[model]["success_count"] += 1

        # Model usage frequency across all categories
        models = list(all_model_data.keys())
        usage_counts = [all_model_data[model]["usage_count"] for model in models]
        success_rates = [
            (
                all_model_data[model]["success_count"]
                / all_model_data[model]["usage_count"]
            )
            * 100
            if all_model_data[model]["usage_count"] > 0
            else 0
            for model in models
        ]

        # Get colors for each model
        model_colors_list = [
            self.model_colors.get(model, "#808080") for model in models
        ]

        # Bar chart of model usage
        bars1 = ax1.bar(models, usage_counts, color=model_colors_list)
        ax1.set_title("Model Usage Frequency")
        ax1.set_ylabel("Usage Count")
        ax1.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars1, usage_counts):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(val),
                ha="center",
                va="bottom",
            )

        # Success rate by model
        bars2 = ax2.bar(models, success_rates, color=model_colors_list)
        ax2.set_title("Model Success Rate")
        ax2.set_ylabel("Success Rate (%)")
        ax2.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars2, success_rates):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
            )

        # Clean energy vs accuracy scatter plot
        clean_percentages = []
        model_names = []
        colors_scatter = []

        for cat in self.categories:
            if self.results is None or not self.results[cat]:
                continue

            result = self.results[cat][0]
            ts_data = self.extract_time_series(result)
            summary = self.get_category_summary(cat)

            # Calculate per-model clean energy percentage
            model_clean_energy = {}
            model_usage = {}

            for i, model in enumerate(ts_data["models"]):
                if model not in model_clean_energy:
                    model_clean_energy[model] = 0
                    model_usage[model] = 0

                model_usage[model] += 1
                # Add clean energy contribution for this timestep
                if i < len(ts_data["clean_energy"]):
                    model_clean_energy[model] += ts_data["clean_energy"][i] - (
                        ts_data["clean_energy"][i - 1] if i > 0 else 0
                    )

            # Average clean energy per model
            for model in model_usage:
                if model_usage[model] > 0:
                    avg_clean = model_clean_energy[model] / model_usage[model]
                    clean_percentages.append(avg_clean * 100)  # Convert to percentage
                    model_names.append(
                        f"{model}\n({cat.replace('top_', '').replace('_', ' ').title()})"
                    )
                    colors_scatter.append(self.model_colors.get(model, "#808080"))

        ax3.scatter(
            range(len(clean_percentages)),
            clean_percentages,
            c=colors_scatter,
            s=100,
            alpha=0.7,
        )
        ax3.set_title("Clean Energy Usage by Model & Category")
        ax3.set_ylabel("Clean Energy per Usage (%)")
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Model selection timeline for top clean energy category
        if (
            self.results
            and "top_most_clean_energy" in self.results
            and self.results["top_most_clean_energy"]
        ):
            result = self.results["top_most_clean_energy"][0]
            ts_data = self.extract_time_series(result)

            # Create timeline showing model switches
            unique_models = list(set(ts_data["models"]))
            model_to_y = {model: i for i, model in enumerate(unique_models)}

            y_positions = [model_to_y[model] for model in ts_data["models"]]
            colors_timeline = [
                self.model_colors.get(model, "#808080") for model in ts_data["models"]
            ]

            ax4.scatter(
                ts_data["timesteps"], y_positions, c=colors_timeline, s=20, alpha=0.8
            )
            ax4.set_title("Model Selection Timeline (Top Clean Energy)")
            ax4.set_xlabel("Timestep")
            ax4.set_ylabel("Model")
            ax4.set_yticks(range(len(unique_models)))
            ax4.set_yticklabels(unique_models)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_clean_energy_vs_accuracy(self):
        """Window 4: Clean Energy vs Accuracy Tradeoff Analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Clean Energy vs Accuracy Tradeoffs", fontsize=16, fontweight="bold"
        )

        # Collect tradeoff data for each category
        tradeoff_data = {}
        for cat in self.categories:
            if self.results is None or not self.results[cat]:
                continue

            result = self.results[cat][0]
            ts_data = self.extract_time_series(result)
            summary = self.get_category_summary(cat)

            # Calculate metrics
            success_count = sum(
                1 for outcome in ts_data["outcomes"] if outcome == "success"
            )
            total_decisions = len(ts_data["outcomes"])
            accuracy = (
                (success_count / total_decisions * 100) if total_decisions > 0 else 0
            )

            tradeoff_data[cat] = {
                "accuracy": accuracy,
                "clean_energy_pct": summary.get("clean_energy_percentage", 0),
                "total_energy": summary.get("total_energy", 0),
                "final_battery": summary.get("final_battery", 0),
                "model_diversity": len(set(ts_data["models"])),
                "avg_clean_per_success": summary.get("total_energy", 0) / success_count
                if success_count > 0
                else 0,
            }

        # Scatter plot: Clean Energy % vs Accuracy
        clean_pcts = [tradeoff_data[cat]["clean_energy_pct"] for cat in self.categories]
        accuracies = [tradeoff_data[cat]["accuracy"] for cat in self.categories]
        colors_scatter = [self.colors[cat] for cat in self.categories]

        ax1.scatter(
            clean_pcts,
            accuracies,
            c=colors_scatter,
            s=150,
            alpha=0.8,
            edgecolors="black",
            linewidth=2,
        )
        ax1.set_xlabel("Clean Energy Percentage (%)")
        ax1.set_ylabel("Accuracy (Success Rate %)")
        ax1.set_title("Clean Energy vs Accuracy Tradeoff")
        ax1.grid(True, alpha=0.3)

        # Add category labels
        for i, cat in enumerate(self.categories):
            ax1.annotate(
                cat.replace("top_", "").replace("_", " ").title(),
                (clean_pcts[i], accuracies[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

        # Energy efficiency vs accuracy
        energy_per_success = [
            tradeoff_data[cat]["avg_clean_per_success"] for cat in self.categories
        ]
        bars2 = ax2.bar(
            range(len(self.categories)),
            energy_per_success,
            color=[self.colors[cat] for cat in self.categories],
        )
        ax2.set_title("Energy per Successful Inference")
        ax2.set_ylabel("Energy per Success (Wh)")
        ax2.set_xticks(range(len(self.categories)))
        ax2.set_xticklabels(
            [
                cat.replace("top_", "").replace("_", " ").title()
                for cat in self.categories
            ],
            rotation=45,
            ha="right",
        )
        for bar, val in zip(bars2, energy_per_success):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(energy_per_success) * 0.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Model diversity vs performance
        diversities = [tradeoff_data[cat]["model_diversity"] for cat in self.categories]
        ax3.scatter(
            diversities,
            accuracies,
            c=colors_scatter,
            s=150,
            alpha=0.8,
            edgecolors="black",
            linewidth=2,
        )
        ax3.set_xlabel("Number of Different Models Used")
        ax3.set_ylabel("Accuracy (Success Rate %)")
        ax3.set_title("Model Diversity vs Accuracy")
        ax3.grid(True, alpha=0.3)

        # Add labels
        for i, cat in enumerate(self.categories):
            ax3.annotate(
                cat.replace("top_", "").replace("_", " ").title(),
                (diversities[i], accuracies[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

        # Performance radar-style comparison
        categories_short = [
            cat.replace("top_", "").replace("_", " ").title() for cat in self.categories
        ]

        # Normalize metrics for comparison (0-1 scale)
        norm_clean = [pct / 100 for pct in clean_pcts]
        norm_accuracy = [acc / 100 for acc in accuracies]
        norm_battery = [
            tradeoff_data[cat]["final_battery"]
            / max([tradeoff_data[c]["final_battery"] for c in self.categories])
            for cat in self.categories
        ]

        x = np.arange(len(categories_short))
        width = 0.25

        ax4.bar(x - width, norm_clean, width, label="Clean Energy %", alpha=0.8)
        ax4.bar(x, norm_accuracy, width, label="Accuracy", alpha=0.8)
        ax4.bar(x + width, norm_battery, width, label="Final Battery", alpha=0.8)

        ax4.set_xlabel("Strategy Categories")
        ax4.set_ylabel("Normalized Performance (0-1)")
        ax4.set_title("Multi-Metric Performance Comparison")
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories_short, rotation=45, ha="right")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_bucket_analysis(self):
        """Window 5: Per-Bucket Cross-Analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Per-Bucket Cross-Analysis", fontsize=16, fontweight="bold")

        # Collect all data points across all categories for bucket analysis
        all_data_points = []
        for cat in self.categories:
            if self.results is None or not self.results[cat]:
                continue

            result = self.results[cat][0]
            ts_data = self.extract_time_series(result)

            for i, timestep in enumerate(ts_data["timesteps"]):
                all_data_points.append(
                    {
                        "category": cat,
                        "timestep": timestep,
                        "model": ts_data["models"][i],
                        "battery": ts_data["battery_levels"][i],
                        "clean_energy": ts_data["clean_energy"][i],
                        "dirty_energy": ts_data["dirty_energy"][i],
                        "charging": ts_data["charging"][i],
                        "outcome": ts_data["outcomes"][i],
                        "total_energy": ts_data["clean_energy"][i]
                        + ts_data["dirty_energy"][i],
                    }
                )

        # Define buckets
        success_bucket = [dp for dp in all_data_points if dp["outcome"] == "success"]
        clean_energy_bucket = sorted(
            all_data_points, key=lambda x: x["clean_energy"], reverse=True
        )[: len(all_data_points) // 3]
        high_battery_bucket = sorted(
            all_data_points, key=lambda x: x["battery"], reverse=True
        )[: len(all_data_points) // 3]

        # 1. Clean energy distribution in success bucket
        if success_bucket:
            clean_energies_success = [dp["clean_energy"] for dp in success_bucket]
            models_in_success = [dp["model"] for dp in success_bucket]

            # Group by model
            model_clean_energy = {}
            for model, clean_e in zip(models_in_success, clean_energies_success):
                if model not in model_clean_energy:
                    model_clean_energy[model] = []
                model_clean_energy[model].append(clean_e)

            # Box plot of clean energy by model in success bucket
            models = list(model_clean_energy.keys())
            clean_data = [model_clean_energy[model] for model in models]
            colors_box = [self.model_colors.get(model, "#808080") for model in models]

            bp1 = ax1.boxplot(clean_data, patch_artist=True, labels=models)
            for patch, color in zip(bp1["boxes"], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax1.set_title("Clean Energy Distribution in Success Bucket")
            ax1.set_ylabel("Clean Energy (Wh)")
            ax1.tick_params(axis="x", rotation=45)
            ax1.grid(True, alpha=0.3)

        # 2. Success rate by model in clean energy bucket
        if clean_energy_bucket:
            model_success_counts = {}
            model_total_counts = {}

            for dp in clean_energy_bucket:
                model = dp["model"]
                if model not in model_success_counts:
                    model_success_counts[model] = 0
                    model_total_counts[model] = 0
                model_total_counts[model] += 1
                if dp["outcome"] == "success":
                    model_success_counts[model] += 1

            models = list(model_total_counts.keys())
            success_rates = [
                (model_success_counts[model] / model_total_counts[model]) * 100
                for model in models
            ]
            colors_bar = [self.model_colors.get(model, "#808080") for model in models]

            bars2 = ax2.bar(models, success_rates, color=colors_bar)
            ax2.set_title("Success Rate in Top Clean Energy Bucket")
            ax2.set_ylabel("Success Rate (%)")
            ax2.tick_params(axis="x", rotation=45)
            for bar, val in zip(bars2, success_rates):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                )

        # 3. Battery levels by outcome across all buckets
        outcome_battery_data = {"success": [], "small_miss": [], "large_miss": []}
        for dp in all_data_points:
            outcome_battery_data[dp["outcome"]].append(dp["battery"])

        outcomes = list(outcome_battery_data.keys())
        battery_data = [outcome_battery_data[outcome] for outcome in outcomes]
        colors_outcome = ["#2E8B57", "#FF8C00", "#DC143C"]  # Green, Orange, Red

        bp3 = ax3.boxplot(battery_data, patch_artist=True, labels=outcomes)
        for patch, color in zip(bp3["boxes"], colors_outcome):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_title("Battery Levels by Outcome (All Data)")
        ax3.set_ylabel("Battery Level (Wh)")
        ax3.grid(True, alpha=0.3)

        # 4. Cross-bucket performance matrix
        bucket_metrics = {
            "Success Bucket": {
                "avg_clean": np.mean([dp["clean_energy"] for dp in success_bucket])
                if success_bucket
                else 0,
                "avg_battery": np.mean([dp["battery"] for dp in success_bucket])
                if success_bucket
                else 0,
                "success_rate": 100.0,  # By definition
            },
            "Clean Energy Bucket": {
                "avg_clean": np.mean([dp["clean_energy"] for dp in clean_energy_bucket])
                if clean_energy_bucket
                else 0,
                "avg_battery": np.mean([dp["battery"] for dp in clean_energy_bucket])
                if clean_energy_bucket
                else 0,
                "success_rate": (
                    len(
                        [dp for dp in clean_energy_bucket if dp["outcome"] == "success"]
                    )
                    / len(clean_energy_bucket)
                    * 100
                )
                if clean_energy_bucket
                else 0,
            },
            "High Battery Bucket": {
                "avg_clean": np.mean([dp["clean_energy"] for dp in high_battery_bucket])
                if high_battery_bucket
                else 0,
                "avg_battery": np.mean([dp["battery"] for dp in high_battery_bucket])
                if high_battery_bucket
                else 0,
                "success_rate": (
                    len(
                        [dp for dp in high_battery_bucket if dp["outcome"] == "success"]
                    )
                    / len(high_battery_bucket)
                    * 100
                )
                if high_battery_bucket
                else 0,
            },
        }

        bucket_names = list(bucket_metrics.keys())
        metrics = ["avg_clean", "avg_battery", "success_rate"]
        metric_labels = ["Avg Clean Energy", "Avg Battery", "Success Rate"]
        colors_metrics = ["#2E8B57", "#4169E1", "#FFD700"]

        x = np.arange(len(bucket_names))
        width = 0.25

        for i, (metric, label, color) in enumerate(
            zip(metrics, metric_labels, colors_metrics)
        ):
            values = [bucket_metrics[bucket][metric] for bucket in bucket_names]
            # Normalize for comparison (except success rate which is already 0-100)
            if metric != "success_rate":
                max_val = max(values) if max(values) > 0 else 1
                values = [
                    v / max_val * 100 for v in values
                ]  # Scale to 0-100 for comparison

            ax4.bar(x + i * width, values, width, label=label, color=color, alpha=0.8)

        ax4.set_xlabel("Buckets")
        ax4.set_ylabel("Normalized Performance (0-100)")
        ax4.set_title("Cross-Bucket Performance Comparison")
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(bucket_names, rotation=45, ha="right")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

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

    def run_analysis(
        self,
        show_plots: bool = True,
        save_plots: bool = False,
        timestamp: Optional[str] = None,
    ):
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
            figures.append(self.plot_clean_energy_vs_accuracy())
            figures.append(self.plot_bucket_analysis())
            figures.append(self.plot_energy_efficiency())

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
