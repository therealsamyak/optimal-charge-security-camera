# AI-Optimized Implementation Plan

## Dependencies
- [ ] `uv add torch torchvision jsonschema`

## Phase 1: Neural Network Core
- [ ] **File**: `src/neural_controller.py`
  - [ ] PyTorch Module class
  - [ ] Input: 4 features → Hidden[128,64] → Dual outputs (6-softmax, 1-sigmoid)
  - [ ] Loss: 0.5*CrossEntropyLoss + 0.5*BCELoss
  - [ ] Forward pass returns (model_probs, charge_prob)

## Phase 2: Training Pipeline Rewrite
- [ ] **File**: `train_custom_controller.py`
  - [ ] Replace linear models with PyTorch training loop
  - [ ] Adam optimizer, learning rate scheduler
  - [ ] Early stopping with validation monitoring
  - [ ] Save/load neural network state dict to JSON

## Phase 3: Controller Integration
- [ ] **File**: `src/controller.py`
  - [ ] CustomController loads neural network weights
  - [ ] select_model() uses neural network forward pass
  - [ ] Maintain existing interface signature

## Phase 4: JSON Logging Conversion
- [ ] **File**: `src/metrics_collector.py`
  - [ ] Remove CSV export methods
  - [ ] Add JSON export with hierarchical structure
  - [ ] Schema: {metadata, aggregated_metrics, detailed_metrics, time_series}

- [ ] **Files**: `batch_simulation.py`, `results.py`
  - [ ] Update to read/write JSON instead of CSV
  - [ ] Remove all CSV file handling

## Phase 5: Test Suite
- [ ] **Directory**: `tests/`

1. [ ] **test_neural_training.py** - 5 training scenarios, test forward/backward pass
2. [ ] **test_training_data.py** - 3 locations × 2 timestamps, test MIPS generation
3. [ ] **test_simulation_power.py** - Use real power_profiles.json, test power calculations
4. [ ] **test_json_logging.py** - Test JSON export structure and schema validation
5. [ ] **test_results_analysis.py** - Test JSON parsing and metric calculations
6. [ ] **test_integration.py** - End-to-end pipeline with 10 scenarios

## Implementation Sequence
- [ ] Add dependencies
- [ ] Create neural_controller.py
- [ ] Rewrite train_custom_controller.py
- [ ] Update controller.py integration
- [ ] Convert metrics_collector.py to JSON
- [ ] Update batch_simulation.py and results.py
- [ ] Create all 6 test files
- [ ] Run test suite validation

## Key Constraints
- [ ] Use existing power_profiles.json (no benchmark rerun)
- [ ] Remove all CSV output (no backward compatibility)
- [ ] Loss weights: 0.5 accuracy + 0.5 energy
- [ ] Test data: minimal (1-10 points each)
- [ ] Maintain existing controller interface

## Architecture Requirements
- [ ] If you find shitty architecture, rewrite it to be more object-oriented/functional programming
- [ ] Add clear error throwing with descriptive messages
- [ ] Add debug logging to console for easy troubleshooting
- [ ] Ensure proper separation of concerns
- [ ] Use type hints only for function inputs (outputs implied)

## Progress Tracking
Mark tasks as completed by checking the [ ] boxes as you implement each component. This plan is designed for AI agents to execute systematically while tracking progress.