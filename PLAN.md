# Final Multi-Phase Implementation Plan

## Phase 1: Foundation & Infrastructure

### 1. Project Setup

- [x] Use uv to configure dependencies: ultralytics, pandas, numpy, pulp, psutil, pytest
- [x] Create directory structure: src/, tests/, logs/, results/
- [x] Setup error-focused logging

### 2. Data Models & Classes

- [x] Battery class: 4000 mAh capacity, charge/discharge simulation
- [x] YOLOModel class: wrapper for YOLOv10 models with psutil power profiling
- [x] EnergyData class: load/process clean energy CSVs
- [x] Controller base class: abstract interface

## Phase 2: Power Benchmarking

### 1. Power Measurement System

- [x] Implement psutil-based CPU power monitoring (cpu_percent, battery sensors)
- [x] Create benchmark suite for each YOLOv10 model (N,S,M,B,L,X)
- [x] Run models on benchmark images, measure power consumption
- [x] Store power profiles for 4000 mAh battery simulation

### 2. Battery Simulation

- [x] Implement realistic battery behavior with 100W USB-C charging
- [x] Validate power consumption models

## Phase 3: Controller Implementation

### 1. Base Controllers

- [x] NaiveWeakController: always smallest model, charge only when battery ≤ 20%
- [x] NaiveStrongController: always largest model, charge only when battery ≤ 20%
- [x] OracleController: PuLP MILP solver using future knowledge, charges when clean energy percentage is maximal

### 2. Custom Controller Training

- [x] Create MIPS solver to generate training data with diverse scenarios:
  - Battery levels: 5-100% (step 5%)
  - Clean energy: 0-100% (step 10%)
  - Accuracy requirements: 70-95% (step 5%)
  - Latency requirements: 1000-3000ms (step 250ms)
  - Target: 10,000 training samples
- [ ] Cache MIPS results to JSON file for reuse
- [x] Gradient descent training with dual outputs (model selection + charging decision):
  - Loss: α _ (1 - accuracy) + β _ latency + γ \* non-clean-energy-usage
  - Initial weights: α=0.5, β=0.3, γ=0.2
- [x] Implement CustomController that utilizes trained weights

### 3. Desired Functionality

- [x] A python file exists that solves the MIPS problem and outputs training data to JSON
- [x] A python file exists that trains the CustomController from scratch and saves weights to JSON

## Phase 4: Simulation Engine

### IMPORTANT
Ensure that only one new .py file is added to the root of this repository for this step (the file that runs the full simulation engine). It is fine to abstract code in an object-oriented / functional programming approach, but do that within `src/` directory such that the root only has that one .py file.

### 1. Core Simulation Framework

- [x] Create `SimulationEngine` class with configurable parameters:
  - Duration: 7 days (604,800 seconds)
  - Task interval: 5 seconds (120,960 total tasks)
  - Time acceleration: configurable (1x, 10x, 100x)
- [x] Implement `TaskGenerator` for realistic security camera workload:
  - Random task arrival with configurable frequency
  - Accuracy requirements: 70-95% (uniform distribution)
  - Latency requirements: 1000-3000ms (uniform distribution)
- [x] Integrate `EnergyData` class for 5-minute clean energy updates:
  - Load 4 location datasets (CA, FL, NW, NY)
  - Interpolate between 5-minute data points for 5-second tasks
  - Support seasonal variations (4 weeks per season)

### 2. Configuration System

- [x] Create `config.jsonc` structure:
  ```jsonc
  {
    "simulation": {
      "duration_days": 7,
      "task_interval_seconds": 5,
      "time_acceleration": 1
    },
    "battery": {
      "capacity_wh": 5.0,
      "charge_rate_watts": 100
    },
    "locations": ["CA", "FL", "NW", "NY"],
    "seasons": ["winter", "spring", "summer", "fall"],
    "controllers": ["naive_weak", "naive_strong", "oracle", "custom"],
    "output_dir": "results/"
  }
  ```
- [x] Implement `ConfigLoader` class to parse and validate configuration
- [x] Update `Battery` class to work directly with Wh units:
  - **Power profiles use mW/mWh, convert to W/Wh: `power_w = power_mw / 1000`**
  - **Energy: `energy_wh = energy_mwh / 1000`**
  - **No voltage conversion needed - direct Wh units**
- [x] Implement charging behavior: when controller decides to charge, continue charging until next controller decision (next task interval)

### 3. Metrics Collection System

- [x] Create `MetricsCollector` class with real-time tracking:
  - Small model miss rate: (failed_small_tasks / total_small_tasks) × 100
  - Large model miss rate: (failed_large_tasks / total_large_tasks) × 100
  - Total energy consumption: sum(all_power_usage)
  - Clean energy percentage: (clean_energy / total_energy) × 100
  - Battery level tracking over time
  - Model selection distribution
- [x] Implement `CSVExporter` for results:
  - Per-simulation summary CSV
  - Detailed time-series data (optional)
  - Aggregated statistics across all simulations

### 4. Simulation Orchestrator

- [x] Create `SimulationRunner` class:
  - Execute 192 total simulations (4 locations × 4 seasons × 4 controllers × 3 weeks)
  - Parallel execution support with configurable worker count
  - **Failure handling: terminate immediately on any failure, ignore all results, no CSV output**
  - Progress tracking for successful simulations only
  - Clean error logging to console only
- [x] Implement result aggregation and comparison tools (only for successful runs)

### 5. Phase 4 Testing

- [x] Create `test_simulation_engine.py`:
  - Unit tests for `SimulationEngine` class with short durations (1-5 minutes)
  - Test `TaskGenerator` with deterministic seeds for reproducible results
  - Validate energy interpolation between 5-minute data points
  - Test configuration loading and validation
- [x] Create `test_metrics_collector.py`:
  - Test metrics calculation accuracy with known inputs
  - Validate CSV export format and data integrity
  - Test edge cases (empty simulations, single task scenarios)
- [x] Create `test_battery_integration.py`:
  - Test Wh-based battery operations with power profile conversions
  - Validate charging behavior between controller decisions
  - Test battery depletion and recovery scenarios
- [x] Create `test_integration_short.py`:
  - End-to-end test with 1-hour simulations instead of 7-day
  - Test all controller types with reduced task count
  - Validate failure handling (terminate on error, no CSV output)
  - Performance benchmarking for short simulation runs

## Phase 5: Testing & Validation

### 1. Unit Tests

- [ ] pytest-based individual component testing

### 2. Integration Tests

- [ ] pytest-based full simulation validation

### 3. Performance Tests

- [ ] 192 total simulations

## Unresolved Questions

- None
