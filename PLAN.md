# Optimal Charge Security Camera - Simulation Framework

## Introduction

This project is an adaptive security camera simulation that intelligently switches between YOLO models based on battery level and clean energy availability. The system balances accuracy, efficiency, and clean energy usage in real-time through a comprehensive simulation framework.

### Key Components

- **Battery Simulation**: Fully software-simulated battery where each inference consumes energy based on selected YOLO model
- **Clean Energy Data**: Uses Los Angeles Department of Water and Power (LDWP) dataset providing 5-minute granularity carbon intensity data for 2024
- **Model Performance**: Utilizes YOLOv10 team benchmark data from `model-data.csv` for accuracy and latency metrics
- **Configuration**: User-defined accuracy and latency thresholds specified in `config.jsonc`

## Architecture

The simulation framework consists of:

```
src/
├── simulation/
│   ├── runner.py          # Main simulation orchestrator
│   ├── controllers.py     # Custom, Oracle, Benchmark controllers
│   └── metrics.py         # Performance tracking and analysis
├── data/
│   ├── energy_loader.py   # LDWP data integration
│   └── model_data.py      # YOLO model data management
├── sensors/
│   └── simulation_sensors.py  # Battery and energy simulation
├── utils/
│   └── config.py          # Configuration management
└── main_simulation.py     # Simulation entry point
```

## Evaluation Methodology

### Simulation Parameters

- **Duration**: 24-hour simulations
- **Seasonal Coverage**: Four representative days:
  - January 5th (Winter)
  - April 15th (Spring) 
  - July 4th (Summer)
  - October 20th (Fall)
- **Image Processing**: Static images with varying quality levels
- **Interval**: Consistent processing intervals (configurable)

### Controller Comparison

Three distinct controller approaches are compared:

#### 1. Custom Controller
- **Algorithm**: Weighted scoring balancing multiple factors
- **Inputs**: User-defined accuracy/latency requirements, energy cleanliness, battery level
- **Weights**: Configurable weights for accuracy, latency, clean energy, and battery conservation
- **Decision**: Real-time model selection and charging decisions

#### 2. Oracle Controller (Omniscient)
- **Algorithm**: Mixed-Integer Linear Programming (MILP) optimization
- **Objective**: Maximize total clean energy consumption over 24-hour period
- **Knowledge**: Complete historical energy data and future awareness
- **Constraints**: Battery capacity, performance thresholds, energy availability

#### 3. Benchmark Controller
- **Algorithm**: Performance-at-all-costs approach
- **Strategy**: Always use largest available model (YOLOv10-X)
- **Charging**: Only when battery below threshold (30%)
- **Energy**: Ignores clean energy considerations

### Performance Metrics

Two primary evaluation metrics:

#### Miss Rate Analysis
- **Small Miss**: Model output fails to meet accuracy/latency thresholds
- **Large Miss**: Battery completely dead, no output possible
- **Tracking**: Separate documentation of both miss types

#### Energy Consumption Analysis
- **Total Energy Used**: Sum of all energy consumption during simulation
- **Clean Energy Used**: Energy consumed during high clean energy periods
- **Clean Energy Percentage**: Ratio of clean energy to total energy
- **Trade-offs**: Analysis of energy quantity vs. cleanliness trade-offs

### Configuration Parameters

The `config.jsonc` file includes:

```jsonc
{
  "accuracy_threshold": 0.9,           // User accuracy requirement (0.0-1.0)
  "latency_threshold_ms": 10.0,       // User latency requirement (milliseconds)
  
  "simulation": {
    "date": "2024-01-05",            // Simulation date
    "image_quality": "good",           // Image quality: "good" or "bad"
    "output_interval_seconds": 10,       // Processing interval
    "controller_type": "custom"         // Controller: "custom", "oracle", "benchmark"
  },
  
  "battery": {
    "initial_capacity": 100.0,          // Starting battery level (0-100%)
    "charging_rate": 0.0035,           // Charging rate (%/second)
    "low_battery_threshold": 20.0       // Forced charging threshold
  },
  
  "model_energy_consumption": {           // Energy cost per inference (%)
    "YOLOv10-N": 0.004,             // ~12 hours battery life
    "YOLOv10-S": 0.007,             // ~7 hours battery life
    "YOLOv10-M": 0.011,             // ~4.5 hours battery life
    "YOLOv10-B": 0.015,             // ~3.2 hours battery life
    "YOLOv10-L": 0.019,             // ~2.5 hours battery life
    "YOLOv10-X": 0.023              // ~2 hours battery life
  }
}
```

## Usage

### Running Simulations

```bash
# Run single simulation with default config
python src/main_simulation.py

# Run with custom configuration
python src/main_simulation.py --config custom_config.jsonc --output results.csv

# Run tests
./scripts/tests.sh
```

### Output Format

Results are exported to CSV with following structure:
```csv
timestamp,battery_level,energy_cleanliness,model_selected,accuracy,latency,miss_type,energy_consumed,clean_energy_consumed
```

## Testing

Comprehensive test suite includes:
- **Unit Tests**: Controller logic, sensor functionality, metrics calculation
- **Integration Tests**: End-to-end simulation execution
- **Scenario Tests**: All seasonal days, image qualities, controller types, and threshold combinations

This framework provides thorough evaluation of controller effectiveness across diverse scenarios and configurations.