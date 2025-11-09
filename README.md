# Optimal Charge Security Camera

An intelligent security camera system with optimal charge management that dynamically selects YOLOv10 models based on battery level and energy cleanliness.

## Prerequisites

- **uv** - Python manager: [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
  - No need for `conda` or `pip`.
- OpenCV (webcam support):

  ```bash
  # macOS
  brew install opencv

  # Ubuntu/Debian
  sudo apt install python3-opencv libopencv-dev
  ```

## Quick Start

```bash
# Clone and run
git clone https://github.com/therealsamyak/optimal-charge-security-camera
cd optimal-charge-security-camera
./scripts/start.sh
```

## Features

- Real-time webcam processing with intelligent model selection
- Energy-aware YOLOv10 model switching
- Multiple controller types (rule-based, ML-based, hybrid)
- Battery management with clean energy prioritization
- Performance monitoring and CSV logging

## Documentation

- **Complete Documentation**: https://therealsamyak.github.io/optimal-charge-security-camera/ - CLI reference, implementation details, and more
- **Configuration**: [src/config/configuration.yml](src/config/configuration.yml) - Default settings

## Testing

Run the entire test suite:

```bash
./scripts/tests.sh
```

## Troubleshooting

See https://therealsamyak.github.io/optimal-charge-security-camera/ for common issues and debug options.
