#!/bin/bash

# Optimal Charge Security Camera - Start Script
# Usage: ./scripts/start.sh [OPTIONS]
# This script configures and runs the OCS camera system with various options

set -e

# Sync dependencies
echo "Syncing dependencies..."
uv sync

# Ensure lefthook git hooks are installed
if command -v uvx &> /dev/null && uvx --help | grep -q "lefthook"; then
    echo "Checking/installing git hooks..."
    uvx lefthook install
fi

# Default values
SOURCE_TYPE="webcam"
INTERVAL_MS=2000
CONTROLLER_TYPE="rule_based"
ENABLE_CHARGING=true
MIN_BATTERY=20.0
MAX_BATTERY=90.0
MIN_ACCURACY=80.0
MAX_LATENCY=100.0
MOCK_BATTERY=80.0
VERBOSE=false
HELP=false
NO_DISPLAY=false

# Function to display help
show_help() {
    cat << 'EOF'
Optimal Charge Security Camera - Start Script

USAGE:
    ./scripts/start.sh [OPTIONS]

OPTIONS:
    Source Options:
    -s, --source TYPE         Input source type (webcam) [default: webcam]
    
    Runtime Options:
    -t, -i, --interval, --interval-ms MS  Processing interval in milliseconds [default: 2000]
    -v, --verbose              Enable verbose logging
    --no-display              Disable GUI display (headless mode)
    
    Controller Options:
    -c, --controller TYPE      Controller type (rule_based|ml_based|hybrid) [default: rule_based]
    --enable-charging          Enable battery charging control [default: true]
    --disable-charging         Disable battery charging control
    --min-battery LEVEL        Minimum battery threshold (0-100) [default: 20.0]
    --max-battery LEVEL        Maximum battery threshold (0-100) [default: 90.0]
    
    Performance Requirements:
    --min-accuracy ACC         Minimum accuracy requirement (0-100) [default: 80.0]
    --max-latency MS           Maximum latency requirement in ms [default: 100.0]
    
    Sensor Options:
    --mock-battery LEVEL       Initial mock battery level (0-100) [default: 80.0]
    
    Utility Options:
    -h, --help                 Show this help message
    --test-scenarios           Run model selection test scenarios (for testing)

EXAMPLES:
    # Default webcam mode
    ./scripts/start.sh

    # Webcam with 1-second interval
    ./scripts/start.sh --interval-ms 1000

    # ML-based controller with verbose logging
    ./scripts/start.sh --controller ml_based --verbose

    # High-performance mode with relaxed battery constraints
    ./scripts/start.sh --min-battery 10 --max-latency 150 --enable-charging

    # Test model selection scenarios (for testing)
    ./scripts/start.sh --test-scenarios

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--source)
            SOURCE_TYPE="$2"
            shift 2
            ;;
        -t|-i|--interval|--interval-ms)
            INTERVAL_MS="$2"
            shift 2
            ;;
        -c|--controller)
            CONTROLLER_TYPE="$2"
            shift 2
            ;;
        --enable-charging)
            ENABLE_CHARGING=true
            shift
            ;;
        --disable-charging)
            ENABLE_CHARGING=false
            shift
            ;;
        --min-battery)
            MIN_BATTERY="$2"
            shift 2
            ;;
        --max-battery)
            MAX_BATTERY="$2"
            shift 2
            ;;
        --min-accuracy)
            MIN_ACCURACY="$2"
            shift 2
            ;;
        --max-latency)
            MAX_LATENCY="$2"
            shift 2
            ;;
        --mock-battery)
            MOCK_BATTERY="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --no-display)
            NO_DISPLAY=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --test-scenarios)
            echo "Running model selection test scenarios..."
            uv run python tests/test_scenarios.py
            exit 0
            ;;
        
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate source type
case $SOURCE_TYPE in
    webcam)
        ;;
    *)
        echo "Error: Invalid source type '$SOURCE_TYPE'. Use: webcam"
        exit 1
        ;;
esac

# Display configuration
echo "Optimal Charge Security Camera Configuration:"
echo "  Source: $SOURCE_TYPE"
echo "  Interval: ${INTERVAL_MS}ms"
echo "  Controller: $CONTROLLER_TYPE"
echo "  Charging: $ENABLE_CHARGING"
echo "  Battery Range: ${MIN_BATTERY}% - ${MAX_BATTERY}%"
echo "  Requirements: ≥${MIN_ACCURACY}% accuracy, ≤${MAX_LATENCY}ms latency"
echo "  Mock Battery: ${MOCK_BATTERY}%"
echo "  Verbose: $VERBOSE"
echo ""

# Cleanup function
cleanup_start() {
    echo "Cleaning up processes..."
    pkill -f "src/main.py" || true
    pkill -f "opencv" || true
    pkill -f "python.*main.py" || true
    pkill -9 -f "python" 2>/dev/null || true
}

# Set up signal handlers for cleanup
trap cleanup_start EXIT INT TERM

# Run main application
echo "Starting OCS Camera..."
uv run python src/main.py \
    --source "$SOURCE_TYPE" \
    --interval-ms "$INTERVAL_MS" \
    --controller "$CONTROLLER_TYPE" \
    $([ "$ENABLE_CHARGING" = true ] && echo "--enable-charging" || echo "--disable-charging") \
    --min-battery "$MIN_BATTERY" \
    --max-battery "$MAX_BATTERY" \
    --min-accuracy "$MIN_ACCURACY" \
    --max-latency "$MAX_LATENCY" \
    --mock-battery "$MOCK_BATTERY" \
    $([ "$VERBOSE" = true ] && echo "--verbose") \
    $([ "$NO_DISPLAY" = true ] && echo "--no-display")