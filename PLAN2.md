# Complete Pipeline Overhaul Plan

## EXECUTIVE SUMMARY

This document outlines a complete overhaul of the optimal charge security camera simulation pipeline. The current codebase has fundamental architectural flaws requiring a complete rewrite rather than incremental improvements.

## TECHNOLOGY STACK

- **ML Framework**: PyTorch
- **Visualization**: Matplotlib (with optional seaborn)
- **Data Storage**: Massive JSON files (streaming for large datasets)
- **Deployment**: Local only
- **Monitoring**: Console logging + JSON file logging
- **Optimization**: OR-Tools for batch optimization
- **Parallelization**: Process-based multiprocessing (configurable workers)

## CURRENT CODEBASE ISSUES

### 1. Training Data Generation (generate_training_data.py)
- **Duplicated parallel processing blocks** (lines 265-320 and 322-356 are identical)
- **Inefficient MIPS solver** - creates new PuLP problem for every single scenario (100,000 times)
- **Poor data loading** - loads power profiles in every worker process instead of sharing
- **Memory inefficient** - generates all 100,000 scenarios in memory before processing
- **Hardcoded parameters** - magic numbers scattered throughout
- **No data validation** - scenarios can have impossible combinations

### 2. Custom Controller Training (train_custom_controller.py)
- **Fundamentally broken ML architecture** - linear model for complex non-linear decision making
- **Poor gradient computation** - manual gradient descent with incorrect derivatives
- **No proper loss function** - combines incompatible objectives
- **Overfitting prone** - no regularization, dropout, or proper validation
- **Wrong feature scaling** - inconsistent normalization between training and inference

### 3. Batch Simulation Runner (batch_simulation.py)
- **Massive memory leaks** - stores all results in memory without cleanup
- **Poor error recovery** - terminates entire batch on single failure
- **Inefficient resource management** - creates 100 workers regardless of system capacity
- **No checkpointing** - failed batch means complete restart
- **Tight coupling** - simulation logic mixed with orchestration

### 4. Results Reporting (results.py)
- **Hardcoded file paths** - only works with specific timestamp format
- **No data validation** - crashes on malformed CSV files
- **Poor statistical analysis** - basic averages without significance testing
- **Memory inefficient** - loads entire datasets into memory
- **No visualization** - text-only output limits insights

### 5. Core Simulation Components (src/ directory)
- **Tight coupling everywhere** - components can't be tested independently
- **God object anti-pattern** - SimulationEngine does everything
- **Inconsistent state management** - battery state scattered across multiple objects
- **Poor abstraction** - concrete implementations mixed with interfaces
- **No dependency injection** - hard-coded dependencies make testing impossible

## NEW ARCHITECTURE

### Phase 1: Training Data Generation
```
src/
├── training/
│   ├── __init__.py
│   ├── scenario_generator.py     # Clean scenario generation
│   ├── optimization_engine.py    # Efficient batch optimization
│   ├── data_validator.py         # Data quality validation
│   └── training_data_exporter.py # Structured output
```

**Key Improvements:**
- Batch optimization using OR-Tools instead of individual PuLP solves
- Streaming scenario generation (memory efficient)
- Shared data structures for parallel workers
- Configurable workers from config.jsonc
- Comprehensive console logging with progress bars

### Phase 2: Neural Network Training
```
src/
├── ml/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_controller.py      # Abstract base
│   │   ├── neural_controller.py    # PyTorch implementation
│   │   └── ensemble_controller.py  # Ensemble of models
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training orchestration
│   │   ├── data_loader.py         # Efficient data loading
│   │   ├── loss_functions.py      # Proper loss functions
│   │   └── hyperparameter_tuner.py # AutoML
│   └── inference/
│       ├── __init__.py
│       ├── predictor.py           # Inference engine
│       └── model_registry.py      # Model versioning
```

**Neural Network Architecture:**
```python
class CustomController(nn.Module):
    def __init__(self, input_dim=4, num_models=6, hidden_dims=[64, 32, 16]):
        super().__init__()
        
        # Shared feature extractor
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )
        
        # Model selection head (softmax)
        self.model_head = nn.Linear(hidden_dims[2], num_models)
        
        # Charging decision head (sigmoid)
        self.charge_head = nn.Linear(hidden_dims[2], 1)
    
    def forward(self, x):
        features = self.feature_layers(x)
        model_logits = self.model_head(features)
        charge_prob = torch.sigmoid(self.charge_head(features))
        return model_logits, charge_prob.squeeze()
```

**Key Improvements:**
- PyTorch implementation with proper multi-output architecture
- Adam optimizer with learning rate scheduling
- Early stopping with validation monitoring
- Checkpointing every N epochs
- Real-time loss/metric logging to console and JSON

### Phase 3: Simulation Engine
```
src/
├── simulation/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── simulation_engine.py    # Clean simulation engine
│   │   ├── time_manager.py        # Time abstraction
│   │   ├── event_system.py        # Event-driven architecture
│   │   └── state_manager.py       # Centralized state
│   ├── components/
│   │   ├── __init__.py
│   │   ├── battery.py            # Battery component
│   │   ├── task_generator.py     # Task generation
│   │   ├── energy_manager.py     # Energy tracking
│   │   └── metrics_collector.py  # Comprehensive metrics
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── base_controller.py    # Abstract interface
│   │   ├── naive_controllers.py  # Simple baselines
│   │   ├── oracle_controller.py  # Optimal baseline
│   │   └── custom_controller.py  # Learned controller
│   └── orchestration/
│       ├── __init__.py
│       ├── batch_runner.py       # Batch orchestration
│       ├── resource_manager.py   # Process/resource management
│       ├── checkpointing.py      # Resume capability
│       └── progress_tracker.py  # Real-time progress
```

**Key Improvements:**
- Event-driven architecture with clean component separation
- Proper battery state management with energy tracking
- Configurable parallel workers for batch execution
- Checkpointing system for resume capability
- Detailed console logging for each simulation

### Phase 4: Results Analytics
```
src/
├── analytics/
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── data_loader.py        # Smart file discovery
│   │   ├── data_validator.py     # Data quality checks
│   │   ├── data_transformer.py   # Feature engineering
│   │   └── streaming_processor.py # Large dataset handling
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── statistical_analyzer.py # Advanced stats
│   │   ├── trend_analyzer.py     # Temporal analysis
│   │   ├── performance_analyzer.py # Controller comparison
│   │   └── efficiency_analyzer.py # Energy efficiency
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── chart_generator.py    # Matplotlib charts
│   │   ├── dashboard.py          # Interactive dashboards
│   │   └── report_generator.py   # HTML/PDF reports
│   └── export/
│       ├── __init__.py
│       ├── csv_exporter.py       # Structured data export
│       ├── json_exporter.py      # Detailed data export
│       └── html_exporter.py      # Interactive reports
```

**Key Improvements:**
- Streaming JSON processing for massive datasets
- Matplotlib visualization with statistical analysis
- Automated insight generation
- Export to structured JSON and CSV
- Comprehensive console reporting

## CONFIGURATION SYSTEM

### New config.jsonc Structure
```jsonc
{
  "global": {
    "workers": 100,
    "random_seed": null,
    "output_dir": "results/",
    "log_level": "INFO",
    "log_to_file": true,
    "log_to_console": true
  },
  
  "training_data": {
    "num_scenarios": 100000,
    "batch_size": 1000,
    "scenario_generation": {
      "battery_range": {"min": 1, "max": 100},
      "accuracy_range": {"min": 0.3, "max": 1.0},
      "latency_options": [1, 2, 3, 5, 8, 10, 15, 20, 25, 30],
      "use_real_energy_data": true
    },
    "optimization": {
      "solver": "ortools",
      "parallel_workers": 100
    },
    "output": {
      "format": "json",
      "compression": false,
      "split_files": true,
      "max_file_size_mb": 100
    }
  },
  
  "controller_training": {
    "model": {
      "architecture": "pytorch_neural_network",
      "hidden_layers": [64, 32, 16],
      "activation": "relu",
      "dropout": 0.2,
      "input_features": 4,
      "num_models": 6
    },
    "training": {
      "epochs": 1000,
      "batch_size": 512,
      "learning_rate": 0.001,
      "optimizer": "adam",
      "early_stopping_patience": 50,
      "validation_split": 0.2,
      "checkpoint_interval": 50
    },
    "loss_weights": {
      "model_selection": 1.0,
      "charging_decision": 1.0
    }
  },
  
  "simulation": {
    "duration_days": 1,
    "task_interval_seconds": 5,
    "time_acceleration": 1,
    "battery": {
      "capacity_wh": 5.0,
      "charge_rate_watts": 100,
      "discharge_efficiency": 0.95
    },
    "logging": {
      "log_every_task": false,
      "log_battery_levels": true,
      "log_energy_usage": true,
      "log_model_selections": true
    }
  },
  
  "batch_simulation": {
    "variations": {
      "num_variations": 4,
      "accuracy_range": {"min": 30, "max": 80},
      "latency_range": {"min": 2, "max": 15},
      "battery_capacity_range": {"min": 2, "max": 15},
      "charge_rate_range": {"min": 50, "max": 200}
    },
    "execution": {
      "parallel_workers": 100,
      "checkpoint_interval": 100,
      "resume_on_failure": true,
      "max_retries": 3
    }
  },
  
  "data_sources": {
    "energy_data_dir": "energy-data/",
    "power_profiles_file": "model-data/power_profiles.json",
    "locations": ["CA", "FL", "NW", "NY"],
    "seasons": ["winter", "spring", "summer", "fall"]
  },
  
  "results": {
    "output_formats": ["json", "csv"],
    "visualization": {
      "enabled": true,
      "library": "matplotlib",
      "save_plots": true,
      "plot_formats": ["png", "pdf"]
    },
    "analysis": {
      "statistical_tests": true,
      "confidence_level": 0.95,
      "trend_analysis": true,
      "performance_comparison": true
    },
    "export": {
      "detailed_json": true,
      "summary_csv": true,
      "split_large_files": true,
      "max_file_size_mb": 50
    }
  }
}
```

## IMPLEMENTATION PHASES

### Phase 1: Foundation (Days 1-2)
- Delete all existing src/ files and main scripts
- Implement new config system with JSONC parsing
- Create logging infrastructure (console + JSON)
- Set up project structure with proper modules
- Implement dependency injection container

### Phase 2: Training Pipeline (Days 3-4)
- Implement streaming training data generation
- Build PyTorch neural network controller
- Create training pipeline with checkpointing
- Add comprehensive progress logging
- Implement batch optimization with OR-Tools

### Phase 3: Simulation Engine (Days 5-6)
- Build event-driven simulation with clean components
- Implement all controller types (naive, oracle, custom)
- Create batch runner with checkpointing
- Add detailed simulation logging
- Implement proper resource management

### Phase 4: Analytics & Reporting (Days 7-8)
- Implement streaming JSON processing
- Create Matplotlib visualization system
- Build statistical analysis engine
- Add comprehensive result reporting
- Create automated insight generation

## NEW FILE STRUCTURE
```
├── config.jsonc                    # Complete configuration
├── main.py                         # Pipeline orchestrator
├── generate_training_data.py        # Clean training data generation
├── train_controller.py              # PyTorch training
├── run_simulation.py                # Single simulation runner
├── run_batch_simulation.py          # Batch simulation runner
├── analyze_results.py               # Results analysis
├── logs/                           # JSON log files
├── results/                        # Output files
├── model-data/                     # UNCHANGED - Power profiles
│   └── power_profiles.json
├── energy-data/                    # UNCHANGED - Energy data
│   ├── US-CAL-LDWP_2024_5_minute.csv
│   ├── US-FLA-FPL_2024_5_minute.csv
│   ├── US-NW-PSEI_2024_5_minute.csv
│   └── US-NY-NYIS_2024_5_minute.csv
└── src/
    ├── core/                       # Core infrastructure
    ├── training/                   # Training pipeline
    ├── ml/                         # Machine learning components
    ├── simulation/                 # Simulation engine
    ├── analytics/                  # Results analysis
    └── utils/                      # Utilities
```

## KEY BENEFITS

1. **10x Performance**: Batch optimization, proper parallelization, efficient algorithms
2. **100x Reliability**: Proper error handling, checkpointing, validation
3. **Complete Observability**: Comprehensive logging, monitoring, analytics
4. **Easy Maintenance**: Clean architecture, separation of concerns, dependency injection
5. **Scalable Design**: Event-driven architecture, resource management, distributed computing

## PIPELINE FLOW

### Step 1: Training Data Generation
- Load energy data from all 4 locations
- Generate realistic scenarios with proper temporal distribution
- Use batch optimization to determine optimal model selection and charging decisions
- Export structured training data with features and labels
- Log all progress to console and JSON files

### Step 2: Custom Controller Training
- Load training data with proper train/validation/test split
- Train PyTorch neural network with multi-output architecture
- Use cross-entropy for model selection, binary cross-entropy for charging
- Implement early stopping and checkpointing
- Save trained model with metadata

### Step 3: Batch Simulation Runner
- Load trained custom controller alongside baseline controllers
- Run simulations across all locations, seasons, and parameter variations
- Use configurable parallel workers for efficient execution
- Implement checkpointing for resume capability
- Log every data point to JSON files with comprehensive metrics

### Step 4: Results Reporting
- Process massive JSON result files with streaming
- Generate statistical analysis with confidence intervals
- Create Matplotlib visualizations for key insights
- Export structured results in multiple formats
- Provide comprehensive console reporting

## UNRESOLVED QUESTIONS (RESOLVED)

1. ✅ **Neural network framework**: PyTorch
2. ✅ **Visualization library**: Matplotlib
3. ✅ **Data storage**: Massive JSON files with streaming
4. ✅ **Cloud deployment**: Local only
5. ✅ **Real-time monitoring**: Console logging + JSON file logging

## IMPORTANT CONSTRAINTS

### UNCHANGED DIRECTORIES
- **model-data/**: Contains power_profiles.json with YOLO model specifications
- **energy-data/**: Contains CSV files with real energy data from 4 locations

These directories must remain completely unchanged as they contain the core data sources for the pipeline.

## NEXT STEPS

1. **Backup existing codebase** (if needed for reference)
2. **Begin Phase 1 implementation** - Foundation and infrastructure
3. **Proceed through phases sequentially** with testing at each step
4. **Comprehensive testing** of complete pipeline
5. **Documentation and deployment**

**Note**: All implementation must preserve the existing model-data/ and energy-data/ directories exactly as they are, as they contain the essential data sources for the entire pipeline.

This plan provides a complete overhaul addressing all identified issues while maintaining configurability and performance requirements.