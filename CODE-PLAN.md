# Complete Refactoring Plan

Do not worry about backwards compatibility for the existing codebase. This is a complete refactor to focus on simulation and comparison. Obviously some files will be necessary and new files will be created (as defined in requirements below), but we will strip down the codebase significantly.

When a phase / task of a plan is complete, mark the corresponding [ ] with an X to signify its done. Before working on a phase or task of the plan, verify the current plan status and update this first, then once it's up-to-date crack on with your task.

## Phase 1: Configuration System
[X] Update config.jsonc with comprehensive parameters:
```jsonc
{
  // User performance requirements
  "accuracy_threshold": 0.9,  // Unit: decimal (0.0-1.0). Example: 0.9 means 90% accuracy requirement
  "latency_threshold_ms": 10.0,  // Unit: milliseconds. Example: 10.0 means max 10ms latency per inference
  
  // Simulation parameters
  "simulation": {
    "date": "2024-01-05",  // Unit: YYYY-MM-DD string. Example: "2024-01-05" for winter simulation
    "image_quality": "good",  // Unit: string. Options: "good" (image1.png) or "bad" (image2.jpeg)
    "output_interval_seconds": 10,  // Unit: seconds. Example: 10 means output CSV every 10 seconds
    "controller_type": "custom"  // Unit: string. Options: "custom", "oracle", "benchmark"
  },
  
  // Battery simulation parameters
  "battery": {
    "initial_capacity": 100.0,  // Unit: percentage (0-100). Example: 100.0 means full battery at start
    "max_capacity": 100.0,     // Unit: percentage (0-100). Example: 100.0 means max battery is 100%
    "charging_rate": 0.0035,     // Unit: % per second. Full charge in 8 hours (28,800 seconds)
    "low_battery_threshold": 20.0  // Unit: percentage (0-100). Example: 20.0 means forced charging below 20%
  },
  
  // YOLO model energy consumption rates (% per inference)
  "model_energy_consumption": {
    "YOLOv10-N": 0.004,   // Unit: % per inference. ~12 hours battery life
    "YOLOv10-S": 0.007,   // Unit: % per inference. ~7 hours battery life  
    "YOLOv10-M": 0.011,   // Unit: % per inference. ~4.5 hours battery life
    "YOLOv10-B": 0.015,   // Unit: % per inference. ~3.2 hours battery life
    "YOLOv10-L": 0.019,   // Unit: % per inference. ~2.5 hours battery life
    "YOLOv10-X": 0.023    // Unit: % per inference. ~2 hours battery life
  },
  
  // Custom controller weighting parameters (must sum to 1.0)
  "custom_controller_weights": {
    "accuracy_weight": 0.4,      // Unit: decimal (0.0-1.0). Example: 0.4 means 40% importance for accuracy
    "latency_weight": 0.3,       // Unit: decimal (0.0-1.0). Example: 0.3 means 30% importance for latency
    "energy_cleanliness_weight": 0.2,  // Unit: decimal (0.0-1.0). Example: 0.2 means 20% importance for clean energy
    "battery_conservation_weight": 0.1   // Unit: decimal (0.0-1.0). Example: 0.1 means 10% importance for battery
  },
  
  // Oracle controller (MILP) parameters
  "oracle_controller": {
    "optimization_horizon_hours": 24,  // Unit: hours. Example: 24 means optimize for full 24-hour day
    "time_step_minutes": 5,            // Unit: minutes. Example: 5 means 5-minute timesteps (matches LDWP data)
    "clean_energy_bonus_factor": 1.5   // Unit: multiplier. Example: 1.5 means clean energy gets 50% bonus in optimization
  },
  
  // Benchmark controller parameters
  "benchmark_controller": {
    "prefer_largest_model": true,       // Unit: boolean. Example: true means always use YOLOv10-X model
    "charge_when_below": 30.0           // Unit: percentage (0-100). Example: 30.0 means charge only when battery below 30%
  }
}
```

## Phase 2: Core Architecture Simplification
[X] Create new simplified structure:
```
src/
├── simulation/
│   ├── runner.py          # Main simulation orchestrator
│   ├── controllers.py     # Custom, Oracle, Benchmark controllers
│   └── metrics.py         # Miss rate, energy tracking
├── data/
│   ├── energy_loader.py   # LDWP data integration
│   └── model_data.py      # Simplified YOLO data
├── sensors/
│   └── simulation_sensors.py  # Enhanced mock sensors
└── utils/
    └── config.py          # Configuration management
```

[X] Remove unnecessary files:
- [X] `src/controller/ml_controller.py` - ML training not needed
- [X] `src/controller/training_data.py` - No data collection required
- [X] `src/controller/performance_validator.py` - Over-engineered validation
- [X] `src/models/model_loader.py` - Simplify to direct CSV reading
- [X] `src/sensors/battery.py` - Abstract interface unnecessary
- [X] `src/sensors/energy.py` - Abstract interface unnecessary
- [X] `src/controller/hybrid_controller.py` - Replace with 3 specific controllers
- [ ] Real-time webcam processing in `main.py` - Replace with simulation runner

## Phase 3: Controller Implementation
[ ] Implement Custom Controller:
- Weighted scoring algorithm using config weights
- Balances accuracy, latency, energy cleanliness, battery conservation
- Real-time decision making based on current conditions

[ ] Implement Oracle Controller:
- Pyomo MILP solver for full 24-hour optimization
- Variables: model selection and charging decisions for each 5-minute timestep
- Objective: maximize total clean energy consumption over entire day
- Constraints: battery capacity, performance thresholds, energy availability
- Uses complete LDWP data for omniscient decision making

[ ] Implement Benchmark Controller:
- Brute-force approach: always use largest available model (YOLOv10-X)
- Ignore clean energy considerations
- Charge battery only when below threshold (30%)
- Represents performance-at-all-costs approach

## Phase 4: Data Integration
[ ] Implement LDWP Energy Data:
- Load 5-minute granularity carbon intensity data for 2024
- Extract data for 4 seasonal simulation days:
  - January 5 (winter)
  - April 15 (spring) 
  - July 4 (summer)
  - October 20 (fall)
- Convert carbon intensity to clean energy percentage

[ ] Implement YOLO Model Data:
- Load benchmark data from model-data.csv
- Map model versions to energy consumption rates from config
- Use accuracy (mAP) and latency data for threshold validation

[ ] Implement Image Processing:
- Static image processing (no webcam)
- image1.png for "good" quality scenarios
- image2.jpeg for "bad" quality scenarios
- Consistent processing interval throughout simulation

## Phase 5: Simulation Execution
[ ] Implement Main Simulation Loop:
- 24-hour simulation period
- Process image every 10 seconds (configurable)
- Output results to CSV every 10 seconds
- Track battery level, energy usage, model selection, performance metrics

[ ] Implement Single Simulation Mode:
- Program runs one simulation using config.jsonc parameters
- Easy to modify config for different test scenarios
- Separate benchmark script for running multiple simulations

## Phase 6: Metrics & Output
[ ] Implement Performance Metrics:
- **Small Miss**: Model output fails to meet accuracy/latency thresholds
- **Large Miss**: Battery completely dead, no output possible
- **Total Energy Used**: Sum of all energy consumption
- **Clean Energy Used**: Energy consumed during high clean energy periods
- **Clean Energy Percentage**: Clean energy / total energy

[ ] Implement CSV Output Format:
```csv
timestamp,battery_level,energy_cleanliness,model_selected,accuracy,latency,miss_type,energy_consumed,clean_energy_consumed
```

## Phase 7: Testing & Validation
[ ] Implement Unit Tests:
- Controller decision logic validation
- Energy consumption calculations
- Battery simulation accuracy
- LDWP data integration

[ ] Implement Integration Tests:
- End-to-end simulation execution
- CSV output validation
- Performance metrics calculation

[ ] Implement Scenario Tests:
- All 4 seasonal days
- Both image quality levels
- All 3 controller types
- Various accuracy/latency threshold combinations

## Phase 8: Documentation Updates
[ ] Update PLAN.md:
- Reflect simplified architecture
- Document configuration parameters
- Explain controller algorithms
- Specify evaluation metrics

[ ] Update README.md:
- New usage instructions
- Configuration guide
- Simulation examples

[ ] Update AGENTS.md:
- New build/test commands
- Code style guidelines for simplified structure

## Summary
This plan transforms the current over-engineered real-time system into a focused simulation framework that directly addresses PLAN.md requirements while maintaining flexibility through comprehensive configuration and enabling thorough comparative analysis of the three controller approaches.