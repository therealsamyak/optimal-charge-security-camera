#!/usr/bin/env python3
"""
New graph methods for tree_results.py overhaul
"""

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
from typing import Dict


def create_enhanced_heatmap(
    data: Dict, colors: Dict, metadata: Dict
) -> matplotlib.figure.Figure:
    """Create enhanced model usage heatmap with better colors and visibility."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Categories to include
    base_categories = [
        "top_most_clean_energy",
        "top_success",
        "top_success_small_miss",
        "top_least_total_energy",
        "naive",
        "custom_controller",
    ]

    # Collect all unique models across categories
    all_models = set()
    for cat in base_categories:
        if cat in data and data[cat]:
            if cat in ["naive", "custom_controller"]:
                result = data[cat]
            else:
                result = data[cat][0] if isinstance(data[cat], list) else data[cat]

            action_sequence = result.get("action_sequence", [])
            for action in action_sequence:
                all_models.add(action.get("model", "Unknown"))

    all_models = sorted(list(all_models))

    # Create usage matrix
    usage_matrix = []
    cat_labels = []

    for cat in base_categories:
        if cat in data and data[cat]:
            cat_labels.append(cat.replace("top_", "").replace("_", " ").title())

            if cat in ["naive", "custom_controller"]:
                result = data[cat]
            else:
                result = data[cat][0] if isinstance(data[cat], list) else data[cat]

            action_sequence = result.get("action_sequence", [])

            # Count model usage
            model_counts = {model: 0 for model in all_models}
            for action in action_sequence:
                model = action.get("model", "Unknown")
                if model in model_counts:
                    model_counts[model] += 1

            usage_matrix.append([model_counts[model] for model in all_models])

    if not usage_matrix:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Create heatmap with better color scheme (use plasma instead of viridis for better contrast)
    im = ax.imshow(usage_matrix, cmap="plasma", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(all_models)))
    ax.set_xticklabels(all_models, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cat_labels)))
    ax.set_yticklabels(cat_labels)

    # Add text annotations with improved contrast
    max_value = max(max(row) for row in usage_matrix) if usage_matrix else 1
    for i in range(len(cat_labels)):
        for j in range(len(all_models)):
            if usage_matrix[i][j] > 0:
                # Determine text color based on background intensity
                intensity = usage_matrix[i][j] / max_value
                text_color = "white" if intensity > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    usage_matrix[i][j],
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                    fontweight="bold",
                )

    # Add config parameters to title
    config_params = (
        f"Battery: {metadata.get('battery_capacity_wh', 'N/A')}Wh, "
        f"Charge: {metadata.get('charge_rate_hours', 'N/A')}hr, "
        f"Accuracy: {metadata.get('user_accuracy_requirement', 'N/A')}%, "
        f"Latency: {metadata.get('user_latency_requirement', 'N/A')}s"
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Category")
    ax.set_title(f"Model Usage Frequency - {config_params}")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Usage Count", rotation=270, labelpad=15)

    plt.tight_layout()
    return fig


def create_oracle_comparison_bars(
    data: Dict, colors: Dict, metadata: Dict
) -> matplotlib.figure.Figure:
    """Create oracle comparison bar chart with success/uptime vs naive/custom."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Categories: oracle success, oracle uptime, naive, custom_controller
    categories = ["top_success", "top_success_small_miss", "naive", "custom_controller"]
    category_labels = ["Oracle Success", "Oracle Uptime", "Naive", "Custom Controller"]

    # Extract data
    success_counts = []
    small_miss_counts = []
    large_miss_counts = []

    for cat in categories:
        if cat in data and data[cat]:
            if cat in ["naive", "custom_controller"]:
                result = data[cat]
            else:
                result = data[cat][0] if isinstance(data[cat], list) else data[cat]

            decision_counts = result.get("decision_counts", {})
            success_counts.append(decision_counts.get("success", 0))
            small_miss_counts.append(decision_counts.get("small_miss", 0))
            large_miss_counts.append(decision_counts.get("large_miss", 0))
        else:
            success_counts.append(0)
            small_miss_counts.append(0)
            large_miss_counts.append(0)

    # Create stacked bars
    width = 0.6
    x = np.arange(len(category_labels))

    ax.bar(x, success_counts, width, label="Success", color="#2E8B57")
    ax.bar(
        x,
        small_miss_counts,
        width,
        bottom=success_counts,
        label="Small Miss",
        color="#FFB347",
    )
    ax.bar(
        x,
        large_miss_counts,
        width,
        bottom=np.array(success_counts) + np.array(small_miss_counts),
        label="Large Miss",
        color="#DC143C",
    )

    # Add percentage labels on segments
    for i, (success, small, large) in enumerate(
        zip(success_counts, small_miss_counts, large_miss_counts)
    ):
        total = success + small + large
        if total > 0:
            # Success percentage
            ax.text(
                i,
                success / 2,
                f"{success / total * 100:.0f}%",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=9,
                color="white",
            )
            # Small miss percentage
            ax.text(
                i,
                success + small / 2,
                f"{small / total * 100:.0f}%",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=9,
                color="black",
            )
            # Large miss percentage
            ax.text(
                i,
                success + small + large / 2,
                f"{large / total * 100:.0f}%",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=9,
                color="white",
            )

    # Add config parameters to title
    config_params = (
        f"Battery: {metadata.get('battery_capacity_wh', 'N/A')}Wh, "
        f"Charge: {metadata.get('charge_rate_hours', 'N/A')}hr, "
        f"Accuracy: {metadata.get('user_accuracy_requirement', 'N/A')}%, "
        f"Latency: {metadata.get('user_latency_requirement', 'N/A')}s"
    )

    ax.set_xlabel("Controller Type")
    ax.set_ylabel("Count")
    ax.set_title(f"Oracle Comparison - {config_params}")
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_clean_energy_comparison(
    data: Dict, colors: Dict, metadata: Dict
) -> matplotlib.figure.Figure:
    """Create clean energy line chart with % difference annotations."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Categories to plot
    categories = ["top_success", "top_most_clean_energy", "naive", "custom_controller"]
    category_labels = [
        "Oracle Success",
        "Oracle Clean Energy",
        "Naive",
        "Custom Controller",
    ]

    # Store data for comparison
    line_data = {}

    for cat, label in zip(categories, category_labels):
        if cat in data and data[cat]:
            if cat in ["naive", "custom_controller"]:
                result = data[cat]
            else:
                result = data[cat][0] if isinstance(data[cat], list) else data[cat]

            action_sequence = result.get("action_sequence", [])
            if action_sequence:
                timesteps = [action.get("timestep", 0) for action in action_sequence]
                clean_energy = [
                    action.get("clean_energy_after", 0) for action in action_sequence
                ]
                line_data[cat] = {
                    "timesteps": timesteps,
                    "clean_energy": clean_energy,
                    "label": label,
                }

                # Plot the line
                ax.plot(
                    timesteps,
                    clean_energy,
                    label=label,
                    color=colors.get(cat, "#808080"),
                    linewidth=2,
                    marker="o",
                    markersize=3,
                )

    # Add % difference annotations for naive and custom_controller
    if "top_success" in line_data and (
        "naive" in line_data or "custom_controller" in line_data
    ):
        oracle_success = line_data["top_success"]
        oracle_clean = line_data.get("top_most_clean_energy")

        for comp_cat in ["naive", "custom_controller"]:
            if comp_cat in line_data:
                comp_data = line_data[comp_cat]

                # Find common timesteps for comparison
                common_indices = []
                for i, ts in enumerate(comp_data["timesteps"]):
                    if ts in oracle_success["timesteps"]:
                        oracle_idx = oracle_success["timesteps"].index(ts)
                        common_indices.append((i, oracle_idx))

                # Add annotations at several points
                for i, (comp_idx, oracle_idx) in enumerate(
                    common_indices[:: max(1, len(common_indices) // 4)]
                ):
                    comp_energy = comp_data["clean_energy"][comp_idx]
                    oracle_success_energy = oracle_success["clean_energy"][oracle_idx]
                    ts = comp_data["timesteps"][comp_idx]

                    # Calculate % differences
                    success_diff = (
                        (
                            (comp_energy - oracle_success_energy)
                            / oracle_success_energy
                            * 100
                        )
                        if oracle_success_energy != 0
                        else 0
                    )

                    clean_diff = 0
                    if oracle_clean and ts in oracle_clean["timesteps"]:
                        clean_idx = oracle_clean["timesteps"].index(ts)
                        oracle_clean_energy = oracle_clean["clean_energy"][clean_idx]
                        clean_diff = (
                            (
                                (comp_energy - oracle_clean_energy)
                                / oracle_clean_energy
                                * 100
                            )
                            if oracle_clean_energy != 0
                            else 0
                        )

                    # Add annotation
                    annotation = f"{success_diff:+.0f}% | {clean_diff:+.0f}%"
                    ax.annotate(
                        annotation,
                        (ts, comp_energy),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7
                        ),
                    )

    # Add config parameters to title
    config_params = (
        f"Battery: {metadata.get('battery_capacity_wh', 'N/A')}Wh, "
        f"Charge: {metadata.get('charge_rate_hours', 'N/A')}hr, "
        f"Accuracy: {metadata.get('user_accuracy_requirement', 'N/A')}%, "
        f"Latency: {metadata.get('user_latency_requirement', 'N/A')}s"
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Clean Energy Amount (Wh)")
    ax.set_title(f"Clean Energy Comparison - {config_params}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_battery_comparison(
    data: Dict, colors: Dict, metadata: Dict
) -> matplotlib.figure.Figure:
    """Create battery percentage line chart with duplicate detection."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # All oracle types + naive + custom_controller
    oracle_categories = [
        "top_success",
        "top_most_clean_energy",
        "top_success_small_miss",
        "top_least_total_energy",
    ]
    other_categories = ["naive", "custom_controller"]

    # Track plotted lines to avoid duplicates
    plotted_sequences = {}

    # Plot oracle types first
    for cat in oracle_categories:
        if cat in data and data[cat]:
            result = data[cat][0] if isinstance(data[cat], list) else data[cat]
            action_sequence = result.get("action_sequence", [])

            if action_sequence:
                timesteps = [action.get("timestep", 0) for action in action_sequence]
                battery_levels = [
                    action.get("battery_after", 0)
                    / metadata.get("battery_capacity_wh", 1)
                    * 100
                    for action in action_sequence
                ]

                # Check for duplicates using first 10 points as signature
                sequence_key = tuple(round(b, 2) for b in battery_levels[:10])
                if sequence_key not in plotted_sequences:
                    plotted_sequences[sequence_key] = cat
                    label = cat.replace("top_", "").replace("_", " ").title()
                    ax.plot(
                        timesteps,
                        battery_levels,
                        label=label,
                        color=colors.get(cat, "#808080"),
                        linewidth=2,
                        alpha=0.8,
                    )

    # Plot naive and custom_controller
    for cat in other_categories:
        if cat in data and data[cat]:
            result = data[cat]
            action_sequence = result.get("action_sequence", [])

            if action_sequence:
                timesteps = [action.get("timestep", 0) for action in action_sequence]
                battery_levels = [
                    action.get("battery_after", 0)
                    / metadata.get("battery_capacity_wh", 1)
                    * 100
                    for action in action_sequence
                ]

                label = cat.replace("_", " ").title()
                ax.plot(
                    timesteps,
                    battery_levels,
                    label=label,
                    color=colors.get(cat, "#808080"),
                    linewidth=2.5,
                    linestyle="--",
                )

    # Add config parameters to title
    config_params = (
        f"Battery: {metadata.get('battery_capacity_wh', 'N/A')}Wh, "
        f"Charge: {metadata.get('charge_rate_hours', 'N/A')}hr, "
        f"Accuracy: {metadata.get('user_accuracy_requirement', 'N/A')}%, "
        f"Latency: {metadata.get('user_latency_requirement', 'N/A')}s"
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Battery Percentage (%)")
    ax.set_title(f"Battery Levels Comparison - {config_params}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    return fig


def create_success_clean_scatter(
    data: Dict, colors: Dict, metadata: Dict
) -> matplotlib.figure.Figure:
    """Create success vs clean energy scatter plot for all 6 categories."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # All 6 categories
    categories = [
        "top_success",
        "top_most_clean_energy",
        "top_success_small_miss",
        "top_least_total_energy",
        "naive",
        "custom_controller",
    ]
    category_labels = [
        "Oracle Success",
        "Oracle Clean Energy",
        "Oracle Success Small Miss",
        "Oracle Least Total Energy",
        "Naive",
        "Custom Controller",
    ]

    for cat, label in zip(categories, category_labels):
        if cat in data and data[cat]:
            if cat in ["naive", "custom_controller"]:
                result = data[cat]
            else:
                result = data[cat][0] if isinstance(data[cat], list) else data[cat]

            # Calculate success percentage
            decision_counts = result.get("decision_counts", {})
            total_decisions = result.get("total_decisions", 1)
            successes = decision_counts.get("success", 0)
            success_pct = (
                (successes / total_decisions * 100) if total_decisions > 0 else 0
            )

            # Get clean energy amount
            clean_energy_amount = result.get("agg_clean_energy", 0)

            ax.scatter(
                success_pct,
                clean_energy_amount,
                label=label,
                color=colors.get(cat, "#808080"),
                s=100,
                alpha=0.8,
                edgecolors="black",
            )

            # Add label for each point
            ax.annotate(
                label,
                (success_pct, clean_energy_amount),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    # Add config parameters to title
    config_params = (
        f"Battery: {metadata.get('battery_capacity_wh', 'N/A')}Wh, "
        f"Charge: {metadata.get('charge_rate_hours', 'N/A')}hr, "
        f"Accuracy: {metadata.get('user_accuracy_requirement', 'N/A')}%, "
        f"Latency: {metadata.get('user_latency_requirement', 'N/A')}s"
    )

    ax.set_xlabel("Success Percentage (%)")
    ax.set_ylabel("Clean Energy Amount (Wh)")
    ax.set_title(f"Success vs Clean Energy Tradeoff - {config_params}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_regional_subplots(
    config_data: Dict[str, Dict], graph_type: str, colors: Dict
) -> matplotlib.figure.Figure:
    """Create 2x2 subplot layout for regions."""
    regions = ["CA", "FL", "NW", "NY"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"{graph_type.replace('_', ' ').title()} - Regional Comparison",
        fontsize=16,
        fontweight="bold",
    )

    for i, region in enumerate(regions):
        ax = axes[i // 2, i % 2]

        # Find data for this region
        region_data = None
        region_metadata = None

        for config_type, config_regions in config_data.items():
            if region in config_regions:
                for season, season_data in config_regions[region].items():
                    region_data = season_data.get("results", {})
                    region_metadata = season_data.get("metadata", {})
                    break
                break

        if region_data and region_metadata:
            if graph_type == "oracle_comparison":
                _create_oracle_subplot(ax, region_data, region_metadata, region, colors)
            elif graph_type == "clean_energy":
                _create_clean_energy_subplot(
                    ax, region_data, region_metadata, region, colors
                )
            elif graph_type == "battery_comparison":
                _create_battery_subplot(
                    ax, region_data, region_metadata, region, colors
                )
            elif graph_type == "success_clean_scatter":
                _create_scatter_subplot(
                    ax, region_data, region_metadata, region, colors
                )

        if region_metadata:
            config_info = (
                f"{region} - {region_metadata.get('battery_capacity_wh', 'N/A')}Wh, "
                f"{region_metadata.get('charge_rate_hours', 'N/A')}hr"
            )
            ax.set_title(config_info, fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _create_oracle_subplot(ax, data: Dict, metadata: Dict, region: str, colors: Dict):
    """Create oracle comparison subplot for a region."""
    categories = ["top_success", "top_success_small_miss", "naive", "custom_controller"]
    category_labels = ["Oracle Success", "Oracle Uptime", "Naive", "Custom Controller"]

    success_counts = []
    small_miss_counts = []
    large_miss_counts = []

    for cat in categories:
        if cat in data and data[cat]:
            if cat in ["naive", "custom_controller"]:
                result = data[cat]
            else:
                result = data[cat][0] if isinstance(data[cat], list) else data[cat]

            decision_counts = result.get("decision_counts", {})
            success_counts.append(decision_counts.get("success", 0))
            small_miss_counts.append(decision_counts.get("small_miss", 0))
            large_miss_counts.append(decision_counts.get("large_miss", 0))
        else:
            success_counts.append(0)
            small_miss_counts.append(0)
            large_miss_counts.append(0)

    width = 0.15
    x = np.arange(len(category_labels))

    ax.bar(x, success_counts, width, label="Success", color="#2E8B57")
    ax.bar(
        x,
        small_miss_counts,
        width,
        bottom=success_counts,
        label="Small Miss",
        color="#FFB347",
    )
    ax.bar(
        x,
        large_miss_counts,
        width,
        bottom=np.array(success_counts) + np.array(small_miss_counts),
        label="Large Miss",
        color="#DC143C",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)


def _create_clean_energy_subplot(
    ax, data: Dict, metadata: Dict, region: str, colors: Dict
):
    """Create clean energy subplot for a region."""
    categories = ["top_success", "top_most_clean_energy", "naive", "custom_controller"]
    category_labels = [
        "Oracle Success",
        "Oracle Clean Energy",
        "Naive",
        "Custom Controller",
    ]

    for cat, label in zip(categories, category_labels):
        if cat in data and data[cat]:
            if cat in ["naive", "custom_controller"]:
                result = data[cat]
            else:
                result = data[cat][0] if isinstance(data[cat], list) else data[cat]

            action_sequence = result.get("action_sequence", [])
            if action_sequence:
                timesteps = [action.get("timestep", 0) for action in action_sequence]
                clean_energy = [
                    action.get("clean_energy_after", 0) for action in action_sequence
                ]

                ax.plot(
                    timesteps,
                    clean_energy,
                    label=label,
                    color=colors.get(cat, "#808080"),
                    linewidth=1.5,
                    marker="o",
                    markersize=2,
                )

    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_ylabel("Clean Energy (Wh)", fontsize=8)
    ax.legend(fontsize=8)


def _create_battery_subplot(ax, data: Dict, metadata: Dict, region: str, colors: Dict):
    """Create battery comparison subplot for a region."""
    categories = [
        "top_success",
        "top_most_clean_energy",
        "top_success_small_miss",
        "top_least_total_energy",
        "naive",
        "custom_controller",
    ]

    # Track duplicates
    plotted_sequences = {}

    for cat in categories:
        if cat in data and data[cat]:
            if cat in ["naive", "custom_controller"]:
                result = data[cat]
            else:
                result = data[cat][0] if isinstance(data[cat], list) else data[cat]

            action_sequence = result.get("action_sequence", [])
            if action_sequence:
                timesteps = [action.get("timestep", 0) for action in action_sequence]
                battery_levels = [
                    action.get("battery_after", 0)
                    / metadata.get("battery_capacity_wh", 1)
                    * 100
                    for action in action_sequence
                ]

                # Check for duplicates
                sequence_key = tuple(round(b, 2) for b in battery_levels[:10])
                is_duplicate = sequence_key in plotted_sequences

                if not is_duplicate:
                    plotted_sequences[sequence_key] = cat
                    label = cat.replace("top_", "").replace("_", " ").title()
                    linestyle = "--" if cat in ["naive", "custom_controller"] else "-"
                    ax.plot(
                        timesteps,
                        battery_levels,
                        label=label,
                        color=colors.get(cat, "#808080"),
                        linewidth=1.5,
                        alpha=0.8,
                        linestyle=linestyle,
                    )

    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_ylabel("Battery %", fontsize=8)
    ax.legend(fontsize=8)


def _create_scatter_subplot(ax, data: Dict, metadata: Dict, region: str, colors: Dict):
    """Create success vs clean energy scatter subplot for a region."""
    categories = [
        "top_success",
        "top_most_clean_energy",
        "top_success_small_miss",
        "top_least_total_energy",
        "naive",
        "custom_controller",
    ]
    category_labels = [
        "Oracle Success",
        "Oracle Clean Energy",
        "Oracle Success Small Miss",
        "Oracle Least Total Energy",
        "Naive",
        "Custom Controller",
    ]

    for cat, label in zip(categories, category_labels):
        if cat in data and data[cat]:
            if cat in ["naive", "custom_controller"]:
                result = data[cat]
            else:
                result = data[cat][0] if isinstance(data[cat], list) else data[cat]

            decision_counts = result.get("decision_counts", {})
            total_decisions = result.get("total_decisions", 1)
            successes = decision_counts.get("success", 0)
            success_pct = (
                (successes / total_decisions * 100) if total_decisions > 0 else 0
            )

            clean_energy_amount = result.get("agg_clean_energy", 0)

            ax.scatter(
                success_pct,
                clean_energy_amount,
                color=colors.get(cat, "#808080"),
                s=50,
                alpha=0.8,
                edgecolors="black",
            )

    ax.set_xlabel("Success %", fontsize=8)
    ax.set_ylabel("Clean Energy (Wh)", fontsize=8)
