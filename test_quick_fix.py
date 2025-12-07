#!/usr/bin/env python3
"""Quick test to verify simulation fixes work."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from simulation_engine import SimulationEngine
from controller import NaiveWeakController
from config_loader import SimulationConfig
from power_profiler import PowerProfiler
from energy_data import EnergyData
from metrics_collector import CSVExporter


def main():
    # Quick test simulation
    config = SimulationConfig(
        duration_days=1,
        task_interval_seconds=300,
        user_accuracy_requirement=50.0,
        user_latency_requirement=10.0,
        battery_capacity_wh=10.0,
        charge_rate_watts=50.0,
        time_acceleration=1000,
    )

    profiler = PowerProfiler()
    profiler.load_profiles()
    power_profiles = profiler.get_all_models_data()
    energy_data = EnergyData()

    controller = NaiveWeakController()
    engine = SimulationEngine(
        config=config,
        controller=controller,
        location="CA",
        season="summer",
        week=1,
        power_profiles=power_profiles,
        energy_data=energy_data,
    )

    # Run very short simulation
    engine.config.duration_days = 0.01  # ~14 minutes
    metrics = engine.run()

    print("Key metrics:")
    print(f"  Total tasks: {metrics['total_tasks']}")
    print(f"  Completed tasks: {metrics['completed_tasks']}")
    print(f"  Total energy (Wh): {metrics['total_energy_wh']:.3f}")
    print(f"  Clean energy (Wh): {metrics['clean_energy_wh']:.3f}")
    print(f"  Clean energy (MWh): {metrics['clean_energy_mwh']:.6f}")
    print(f"  Dirty energy (MWh): {metrics['dirty_energy_mwh']:.6f}")
    print(f"  Clean energy %: {metrics['clean_energy_percentage']:.1f}%")
    print(f"  Model selections: {metrics['model_selections']}")
    print(f"  Peak power (mW): {metrics['peak_power_mw']:.3f}")
    print(f"  Battery efficiency score: {metrics['battery_efficiency_score']:.2f}")

    # Test export
    exporter = CSVExporter("results")
    metrics["simulation_id"] = "test_fix"
    metrics["controller"] = "naive_weak"
    csv_path = exporter.export_detailed_results([metrics], "test_fixes.csv")
    print(f"Exported to: {csv_path}")


if __name__ == "__main__":
    main()
