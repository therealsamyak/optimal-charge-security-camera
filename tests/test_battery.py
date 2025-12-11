from typing import Dict


def plot_battery_comparison(data: Dict, colors: Dict, metadata: Dict):
    """Create battery percentage line chart with duplicate detection."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

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

                # Check for duplicates
                sequence_key = tuple(
                    battery_levels[:10]
                )  # First 10 points as signature
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
                    timesteps = [
                        action.get("timestep", 0) for action in action_sequence
                    ]
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
        f"Charge: {metadata.get('charge_rate_hours', 'N/A')}hr"
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Battery Percentage (%)")
    ax.set_title(f"Battery Levels Comparison - {config_params}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    return fig
