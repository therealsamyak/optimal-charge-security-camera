# Optimal Charge Security Camera - Simulation Framework

An intelligent security camera simulation framework that dynamically selects YOLOv10 models based on battery level and clean energy availability. This system balances accuracy, efficiency, and clean energy usage through comprehensive simulation and comparative analysis.

## Prerequisites

- **uv** - Python package manager: [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
- **Python 3.12+** with required packages:
  - pandas (data processing)
  - loguru (logging)
  - pytest (testing)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/therealsamyak/optimal-charge-security-camera
cd optimal-charge-security-camera

# Install dependencies
uv sync

# Run simulation with default config
uv run python src/main_simulation.py

# Run with custom configuration
uv run python src/main_simulation.py --config custom_config.jsonc --output results.csv
```

## Features

- **Simulation Framework**: 24-hour simulations with configurable parameters
- **Three Controller Types**: Custom (weighted scoring), Oracle (MILP optimization), Benchmark (performance-at-all-costs)
- **Energy-Aware Model Selection**: Intelligent YOLOv10 model switching based on battery and clean energy
- **Real Data Integration**: LDWP energy data and YOLOv10 benchmark data
- **Comprehensive Metrics**: Miss rate analysis, energy consumption tracking, clean energy percentage
- **Seasonal Testing**: Support for winter, spring, summer, fall scenarios
- **Extensible Testing**: Unit, integration, and scenario tests

## Configuration

The simulation is configured through `config.jsonc`:

```jsonc
{
  "accuracy_threshold": 0.4,
  "latency_threshold_ms": 15.0,
  "simulation": {
    "date": "2024-01-05",
    "image_quality": "good",
    "output_interval_seconds": 10,
    "controller_type": "custom"
  },
  "battery": {
    "initial_capacity": 100.0,
    "charging_rate": 0.0035,
    "low_battery_threshold": 20.0
  },
  "model_energy_consumption": {
    "YOLOv10-N": 0.004,
    "YOLOv10-S": 0.007,
    "YOLOv10-M": 0.011,
    "YOLOv10-B": 0.015,
    "YOLOv10-L": 0.019,
    "YOLOv10-X": 0.023
  }
}
```

## Usage Examples

### Single Simulation

```bash
# Run with default config
uv run python src/main_simulation.py

# Run with custom configuration
uv run python src/main_simulation.py --config custom_config.jsonc --output results.csv
```

### Different Controllers

Update `config.jsonc` to set `controller_type` to one of:

- `"custom"` - Weighted scoring controller
- `"oracle"` - MILP optimization controller
- `"benchmark"` - Performance-at-all-costs controller

### Batch Simulations

Run all combinations of seasonal days, controllers, and thresholds:

```bash
./scripts/batch_simulations.sh
```

This script runs:

- 4 seasonal days (Jan 5, Apr 15, Jul 4, Oct 20)
- 3 controller types (custom, oracle, benchmark)
- 4 threshold combinations (high/medium/low performance)
- 2 image qualities (good, bad)

Results are saved in `results/` directory with format:
`{date}_{controller}_{accuracy}_{latency}_{quality}.csv`

## Output

Results are exported to CSV with comprehensive metrics:

```csv
timestamp,battery_level,energy_cleanliness,model_selected,accuracy,latency,miss_type,energy_consumed,clean_energy_consumed
2024-01-05T10:00:00,95.0,0.7,YOLOv10-S,0.92,8.0,none,0.007,0.0049
2024-01-05T10:00:10,94.993,0.75,YOLOv10-N,0.85,5.0,small_miss,0.004,0.003
```

### Performance Summary

The simulation outputs a comprehensive summary including:

- Total inferences and miss rates
- Energy consumption and clean energy percentage
- Model usage distribution
- Charging periods

## Testing

Run the complete test suite:

```bash
# All tests
./scripts/tests.sh

# Unit tests only
uv run pytest tests/test_simulation_units.py -v

# Integration tests
uv run pytest tests/test_simulation_integration.py -v

# Scenario tests
uv run pytest tests/test_simulation_scenarios.py -v
```

## Architecture

```
src/
├── simulation/
│   ├── runner.py          # Main simulation orchestrator
│   ├── controllers.py     # Three controller implementations
│   └── metrics.py         # Performance tracking
├── data/
│   ├── energy_loader.py   # LDWP energy data integration
│   └── model_data.py      # YOLO model data management
├── sensors/
│   └── simulation_sensors.py  # Battery and energy simulation
├── utils/
│   └── config.py          # Configuration management
└── main_simulation.py     # Simulation entry point
```

## Documentation

- **Complete Documentation**: https://therealsamyak.github.io/optimal-charge-security-camera/
- **Configuration Guide**: See `config.jsonc` for all available parameters
- **API Reference**: Detailed documentation in source code docstrings

## Research & Evaluation

This framework enables comprehensive evaluation of:

- Controller effectiveness across seasonal variations
- Trade-offs between accuracy, latency, and energy efficiency
- Clean energy optimization strategies
- Battery management approaches

The simulation supports rigorous comparative analysis between different control strategies under realistic conditions.
