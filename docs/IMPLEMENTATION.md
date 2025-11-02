# Implementation Documentation

## Overview

The Optimal Charge Security Camera (OCS Camera) system is an intelligent security camera platform that dynamically selects YOLOv10 models based on battery level, energy cleanliness, and user requirements. The system balances performance with energy efficiency through intelligent controller logic.

## Architecture

### Core Components

- **Main System** (`src/main.py`): Entry point and real-time processing loop
- **Configuration** (`src/config/`): YAML-based configuration management
- **Controllers** (`src/controller/`): Model selection and decision logic
- **Models** (`src/models/`): YOLOv10 model management and inference
- **Sensors** (`src/sensors/`): Battery and energy monitoring interfaces
- **Utilities** (`src/utils/`): Helper functions and common utilities

## Source Implementation

### Main System (`src/main.py`)

The `SecurityCameraSystem` class orchestrates the entire application:

- **Initialization**: Sets up webcam, controllers, model manager, and sensors
- **Processing Loop**: Captures frames, makes controller decisions, runs inference
- **Display**: Shows real-time overlay with system status and detection results
- **Resource Management**: Handles battery consumption and charging control

Key methods:

- `_process_frame()`: Core processing pipeline
- `_display_frame()`: GUI overlay with system metrics
- `run()`: Main processing loop with frame rate control

### Configuration Management (`src/config/`)

`ConfigManager` class provides centralized configuration:

- **YAML Loading**: Loads from `configuration.yml` with fallback defaults
- **Validation**: Ensures configuration values are within valid ranges
- **Access**: Provides dot-notation access to nested config values
- **Runtime Updates**: Supports configuration reloading

Default configuration includes:

- Model availability and performance profiles
- User requirements (accuracy, latency)
- Controller thresholds and behavior
- Sensor settings and mock data options

### Controllers (`src/controller/`)

#### Intelligent Controller (`intelligent_controller.py`)

Rule-based controller using heuristic scoring:

- **Model Profiles**: Performance characteristics for each YOLOv10 variant
- **Scoring Algorithm**: Multi-factor scoring based on:
  - Battery level (high/medium/low/critical)
  - Energy cleanliness (clean energy bonus)
  - User requirements (accuracy, latency constraints)
  - Charging state (opportunistic charging)
- **Decision Logic**: Selects optimal model and charging decisions

#### ML Controller (`ml_controller.py`)

Machine learning interface for future enhancement:

- **Feature Extraction**: Converts system state to ML features
- **Temporal Encoding**: Cyclical encoding for time-based features
- **Training Pipeline**: Structure for collecting and training on data
- **Fallback Logic**: Rule-based fallback when ML model unavailable

#### Hybrid Controller (`hybrid_controller.py`)

Combines rule-based and ML approaches:

- **Controller Types**: Rule-based, ML-based, or hybrid
- **Decision Fusion**: Combines predictions based on confidence
- **Training Data Collection**: Records decisions for future ML training
- **Dynamic Switching**: Can switch between controller types

### Model Management (`src/models/yolo_manager.py`)

`YOLOv10Manager` handles YOLOv10 model operations:

- **Model Caching**: LRU cache for loaded models with memory management
- **Performance Tracking**: Records latency, accuracy, battery consumption
- **Inference Pipeline**: Runs inference with comprehensive metrics
- **Profile Management**: Expected performance characteristics per model

Model variants supported:

- `yolov10n`: Smallest, most efficient (65% accuracy, 15ms)
- `yolov10s`: Small balanced (75% accuracy, 25ms)
- `yolov10m`: Medium (82% accuracy, 40ms)
- `yolov10b`: Large (87% accuracy, 60ms)
- `yolov10l`: Extra large (91% accuracy, 85ms)
- `yolov10x`: Largest, most accurate (94% accuracy, 120ms)

### Sensors (`src/sensors/`)

#### Battery Interface (`battery.py`)

Abstract interface for hardware integration:

```python
class BatterySensor(ABC):
    def get_battery_percentage() -> float
    def consume_battery(amount: float) -> None
    def is_charging() -> bool
    def start_charging() -> None
    def stop_charging() -> None
```

#### Mock Battery (`mock_battery.py`)

Simulation implementation for testing:

- Configurable initial battery level
- Consumption tracking per model
- Charging state management
- Realistic discharge rates

#### Energy Monitoring (`energy.py`, `mock_energy.py`)

Energy cleanliness tracking:

- Time-based energy source simulation
- Clean energy availability (solar/wind patterns)
- Grid energy cleanliness modeling

## Testing Implementation

### Test Structure (`tests/`)

#### Test Runner (`run_tests.py`)

Orchestrates all test execution:

- Runs individual test files with pytest
- Provides detailed output and timing
- Generates comprehensive summary
- Handles test discovery and execution

#### Unit Tests

**Controller Tests** (`test_controller.py`):

- `TestModelController`: Core controller logic validation
- `TestModelProfile`: Data structure validation
- `TestControllerDecision`: Decision object validation

Test scenarios:

- Model scoring with different battery levels
- Clean energy bonus calculations
- Charging decision logic
- User requirement compliance
- Decision reasoning validation

**Mock Sensor Tests** (`test_mock_sensors.py`):

- Sensor data generation and validation
- Performance metrics simulation
- Scenario-based testing (normal, low battery, clean energy)

**Configuration Tests** (`test_config.py`):

- YAML loading and validation
- Default configuration fallback
- Configuration merging and overrides

**Utility Tests** (`test_utils.py`):

- Helper function validation
- Data processing utilities
- Common functionality testing

#### Integration Tests (`test_integration.py`)

End-to-end system validation:

**Component Integration**:

- Config-to-controller integration
- Sensor-to-controller data flow
- Performance validation integration

**Scenario Testing**:

- Low battery scenarios
- Clean energy optimization
- Performance requirement validation
- Configuration variations

**Error Handling**:

- Invalid sensor values
- Edge case handling
- Graceful degradation

**End-to-End Simulation**:

- Multi-cycle processing simulation
- Model selection consistency
- Performance tracking validation

### Test Data Generation

**MockSensorDataGenerator**: Generates realistic sensor readings

- Normal operation scenarios
- Low battery conditions
- Clean energy periods
- Configurable randomness with seeds

**PerformanceDataGenerator**: Simulates inference performance

- Model-specific performance characteristics
- Detection confidence simulation
- Latency and battery consumption modeling

## Key Design Patterns

### Strategy Pattern

Controllers implement different selection strategies:

- Rule-based: Heuristic scoring
- ML-based: Learned decisions
- Hybrid: Confidence-weighted fusion

### Factory Pattern

Model manager handles model instantiation and caching

- Dynamic model loading
- Memory management
- Performance tracking

### Observer Pattern

Performance validation monitors system behavior

- Real-time metrics collection
- Requirement validation
- Performance trend analysis

### Abstract Factory

Sensor interfaces allow multiple implementations

- Hardware sensors (future)
- Mock sensors (testing)
- Simulation environments

## Data Flow

1. **Sensor Input**: Battery level, energy cleanliness, charging state
2. **Controller Decision**: Model selection and charging logic
3. **Model Execution**: YOLOv10 inference with performance tracking
4. **Performance Validation**: Metrics collection and requirement checking
5. **System Update**: Battery consumption, charging control, training data

## Configuration-Driven Behavior

The system is highly configurable through YAML:

- Model availability and performance profiles
- User requirements and thresholds
- Controller behavior parameters
- Sensor simulation settings
- Output and logging configuration

## Extensibility

The architecture supports several extension points:

- New YOLOv10 models via profile configuration
- Alternative controller implementations
- Hardware sensor integration
- Additional performance metrics
- Custom validation rules

## Performance Considerations

- **Model Caching**: LRU cache limits memory usage
- **Lazy Loading**: Models loaded on-demand
- **Performance Tracking**: Minimal overhead metrics collection
- **Batch Processing**: Efficient sensor data handling
- **Memory Management**: Explicit cleanup and garbage collection

This implementation provides a robust, extensible foundation for intelligent security camera operation with optimal resource utilization.
