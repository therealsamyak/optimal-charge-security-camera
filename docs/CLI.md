# CLI Reference

Complete command line interface documentation for the Optimal Charge Security Camera system.

## Basic Usage

```bash
# Default webcam processing
./scripts/start.sh

# With custom options
./scripts/start.sh --controller hybrid --verbose --interval-ms 1000
```

## Command Line Options

### Runtime Options

```bash
-i, --interval-ms MS       Processing interval in milliseconds
                          Default: 2000

-v, --verbose              Enable verbose logging

--no-display              Disable GUI display (headless mode)
```

### Controller Options

```bash
-c, --controller TYPE      Controller type
                          Options: rule_based, ml_based, hybrid
                          Default: rule_based

--enable-charging          Enable battery charging control
                          Default: true

--disable-charging         Disable battery charging control
```

### Performance Requirements

```bash
--min-accuracy ACC         Minimum accuracy requirement (0-100)
                          Default: 80.0

--max-latency MS           Maximum latency requirement in ms
                          Default: 100.0
```

### Sensor Options

```bash
--mock-battery LEVEL       Initial mock battery level (0-100)
                          Default: 80.0
```

### Utility Options

```bash
-h, --help                 Show help message
```

## Usage Examples

### Basic Real-time Processing

```bash
# Default webcam processing
./scripts/start.sh

# Faster processing (1-second interval)
./scripts/start.sh --interval-ms 1000

# Headless mode (no display)
./scripts/start.sh --no-display
```

### Performance Tuning

```bash
# High-performance mode (relaxed latency, high accuracy)
./scripts/start.sh --max-latency 150 --min-accuracy 90

# Energy-saving mode (strict latency, lower accuracy)
./scripts/start.sh --max-latency 50 --min-accuracy 70

# Verbose logging for debugging
./scripts/start.sh --verbose
```

### Controller Configuration

```bash
# Hybrid controller (rule-based with ML augmentation)
./scripts/start.sh --controller hybrid

# ML-based controller (requires trained model)
./scripts/start.sh --controller ml_based

# Disable charging control
./scripts/start.sh --disable-charging
```

### Battery Simulation

```bash
# Start with low battery to test conservation
./scripts/start.sh --mock-battery 25

# Start with full battery
./scripts/start.sh --mock-battery 95
```

## Model Selection Logic

The system intelligently selects YOLOv10 models based on resource availability:

### High Resources (80%+ battery, 80%+ clean energy)

- **Selected Models**: `yolov10l`, `yolov10x`
- **Priority**: Maximum accuracy and capability
- **Use Case**: When power is abundant and energy is clean

### Medium Resources (50-80% battery, 60%+ clean energy)

- **Selected Models**: `yolov10m`, `yolov10b`
- **Priority**: Balance between performance and efficiency
- **Use Case**: Normal operating conditions

### Low Resources (<30% battery)

- **Selected Models**: `yolov10n`, `yolov10s`
- **Priority**: Battery conservation
- **Use Case**: Power-constrained situations

### Charging with Clean Energy

- **Selected Models**: Higher-tier models allowed
- **Priority**: Take advantage of clean energy
- **Use Case**: Opportunistic high-performance processing

## Available Models

| Model      | Accuracy | Latency | Battery Use | Size        |
| ---------- | -------- | ------- | ----------- | ----------- |
| `yolov10n` | 65.0%    | 15ms    | 0.1         | Smallest    |
| `yolov10s` | 75.0%    | 25ms    | 0.2         | Small       |
| `yolov10m` | 82.0%    | 40ms    | 0.4         | Medium      |
| `yolov10b` | 87.0%    | 60ms    | 0.6         | Large       |
| `yolov10l` | 91.0%    | 85ms    | 0.8         | Extra Large |
| `yolov10x` | 94.0%    | 120ms   | 1.0         | Largest     |

## Output and Monitoring

### Real-time Display

The system displays a live camera feed with overlay information:

- Currently selected YOLOv10 model
- Battery level and charging status
- Energy cleanliness percentage
- Detection results and confidence
- Processing latency

### Metrics Logging

The system logs performance metrics to `src/data/output/metrics.csv`:

- Timestamp and configuration
- Selected model and controller score
- Detection confidence and latency
- Battery level and energy cleanliness
- Validation results and violations

### Performance Validation

Real-time validation checks:

- Accuracy requirements compliance
- Latency threshold monitoring
- Battery consumption tracking
- Detection quality assessment

## Troubleshooting

### Common Issues

1. **Webcam not accessible**: Ensure OpenCV is installed and camera permissions are granted
2. **Model loading fails**: Check internet connection for first-time model downloads
3. **High latency**: Try smaller models (`yolov10n`, `yolov10s`) or increase interval
4. **Battery drain**: Use `--disable-charging` or set higher `--min-battery`

### Debug Mode

```bash
# Enable verbose logging
./scripts/start.sh --verbose

# Test without display
./scripts/start.sh --no-display --verbose
```

### Keyboard Controls

When running with display:

- **q** or **ESC**: Quit the application
- **c**: Toggle charging on/off

## Testing

Run the comprehensive test suite:

```bash
./scripts/tests.sh
```

This script tests all command line options, combinations, and edge cases with proper error handling and timeout management.

## Configuration

The system uses `src/config/configuration.yml` for default settings. Command line options override configuration file values.

Key configuration sections:

- `source`: Input source settings (webcam by default)
- `runtime`: Processing interval and verbosity
- `requirements`: Performance thresholds
- `controller`: Control strategy and charging logic
- `models`: Available YOLOv10 variants
- `sensors`: Mock sensor configuration
