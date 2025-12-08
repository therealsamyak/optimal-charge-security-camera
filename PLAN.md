# AI-Optimized Implementation Plan

## Dependencies
- [x] `uv add torch torchvision jsonschema`

## Phase 1: Neural Network Core
- [x] **File**: `src/neural_controller.py`
  - [x] PyTorch Module class
  - [x] Input: 4 features → Hidden[128,64] → Dual outputs (6-softmax, 1-sigmoid)
  - [x] Loss: 0.5*CrossEntropyLoss + 0.5*BCELoss
  - [x] Forward pass returns (model_probs, charge_prob)

## Phase 2: Training Pipeline Rewrite
- [x] **File**: `train_custom_controller.py`
  - [x] Replace linear models with PyTorch training loop
  - [x] Adam optimizer, learning rate scheduler
  - [x] Early stopping with validation monitoring
  - [x] Save/load neural network state dict to JSON

## Phase 3: Controller Integration
- [x] **File**: `src/controller.py`
  - [x] CustomController loads neural network weights
  - [x] select_model() uses neural network forward pass
  - [x] Maintain existing interface signature

## Phase 4: JSON Logging Conversion
- [x] **File**: `src/metrics_collector.py`
  - [x] Remove CSV export methods
  - [x] Add JSON export with hierarchical structure
  - [x] Schema: {metadata, aggregated_metrics, detailed_metrics, time_series}

- [x] **Files**: `batch_simulation.py`, `results.py`
  - [x] Update to read/write JSON instead of CSV
  - [x] Remove all CSV file handling

## Phase 5: Test Suite
- [x] **Directory**: `tests/`

1. [x] **test_neural_training.py** - 5 training scenarios, test forward/backward pass
2. [x] **test_training_data.py** - 3 locations × 2 timestamps, test MIPS generation
3. [x] **test_simulation_power.py** - Use real power_profiles.json, test power calculations
4. [x] **test_json_logging.py** - Test JSON export structure and schema validation
5. [x] **test_results_analysis.py** - Test JSON parsing and metric calculations
6. [x] **test_integration.py** - End-to-end pipeline with 10 scenarios

## Implementation Sequence
- [x] Add dependencies
- [x] Create neural_controller.py
- [x] Rewrite train_custom_controller.py
- [x] Update controller.py integration
- [x] Convert metrics_collector.py to JSON
- [x] Update batch_simulation.py and results.py
- [x] Create all 6 test files
- [x] Run test suite validation

## Key Constraints
- [x] Use existing power_profiles.json (no benchmark rerun)
- [x] Remove all CSV output (no backward compatibility)
- [x] Loss weights: 0.5 accuracy + 0.5 energy
- [x] Test data: minimal (1-10 points each)
- [x] Maintain existing controller interface

## Architecture Requirements
- [x] Object-oriented/functional programming structure
- [x] Clear error throwing with descriptive messages
- [x] Debug logging to console for easy troubleshooting
- [x] Proper separation of concerns
- [x] Type hints only for function inputs (outputs implied)

## Progress Tracking
Mark tasks as completed by checking the [ ] boxes as you implement each component. This plan is designed for AI agents to execute systematically while tracking progress.