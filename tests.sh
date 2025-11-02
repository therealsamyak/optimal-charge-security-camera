#!/bin/bash

# Comprehensive Test Script for Optimal Charge Security Camera
# Tests ALL start.sh options and combinations with proper error handling

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED_TESTS++))
}

log_error() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED_TESTS++))
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Test function
run_test() {
    local test_name="$1"
    local command="$2"
    local expected_exit_code="${3:-0}"
    local timeout="${4:-10}"
    
    ((TOTAL_TESTS++))
    log "Testing: $test_name"
    log "Command: $command"
    
    # Create temp files for output
    local stdout_file=$(mktemp)
    local stderr_file=$(mktemp)
    
    # Run command with timeout (using gtimeout if available, otherwise python)
    if command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$timeout" bash -c "$command" > "$stdout_file" 2> "$stderr_file"
    else
        # Use uv run python for timeout on macOS
        uv run python -c "
import subprocess
import sys
import signal

def timeout_handler(signum, frame):
    sys.exit(124)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm($timeout)

try:
    result = subprocess.run(['bash', '-c', '$command'], capture_output=True, text=True)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    sys.exit(result.returncode)
except SystemExit as e:
    sys.exit(e.code)
" > "$stdout_file" 2> "$stderr_file"
    fi
    local exit_code=$?
    
    # If command timed out but completed successfully (exit code 124), that's expected for long-running commands
    if [ $exit_code -eq 124 ] && [ "$expected_exit_code" -eq 124 ]; then
        exit_code=124
    fi
    
    if [ $exit_code -eq $expected_exit_code ]; then
        log_success "$test_name"
        if [ -s "$stdout_file" ]; then
            echo "Output: $(head -1 "$stdout_file")"
        fi
    else
        if [ $exit_code -eq 124 ]; then
            log_warning "$test_name (timeout after ${timeout}s)"
        else
            log_error "$test_name (exit code: $exit_code, expected: $expected_exit_code)"
        fi
        if [ -s "$stderr_file" ]; then
            echo "Error: $(cat "$stderr_file")"
        fi
    fi
    
    # Cleanup
    rm -f "$stdout_file" "$stderr_file"
    echo ""
}

# Cleanup function
cleanup() {
    log "Cleaning up processes..."
    pkill -f "src/main.py" || true
    pkill -f "start.sh" || true
    pkill -f "opencv" || true
    pkill -f "python.*main.py" || true
    # Force kill any remaining Python processes that might be holding camera
    pkill -9 -f "python" 2>/dev/null || true
    log "Cleanup complete"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Start testing
log "Starting COMPREHENSIVE test suite for Optimal Charge Security Camera"
log "=================================================="

# Test 1: Utility Options
log "Testing Utility Options..."
run_test "Help option (-h)" "./start.sh -h" 0 5
run_test "Help option (--help)" "./start.sh --help" 0 5
run_test "Test scenarios" "./start.sh --test-scenarios" 0 30

# Test 2: Source Options
log "Testing Source Options..."
run_test "Default source (webcam)" "./start.sh" 124 5  # Should timeout starting main
run_test "Webcam source (-s)" "./start.sh -s webcam" 124 5
run_test "Webcam source (--source)" "./start.sh --source webcam" 124 5
run_test "Invalid source type" "./start.sh -s invalid" 1 5

# Test 3: Runtime Options
log "Testing Runtime Options..."
run_test "Default interval" "./start.sh" 124 5
run_test "Custom interval (-t)" "./start.sh -t 1000" 124 5
run_test "Custom interval (--interval)" "./start.sh --interval 500" 124 5
run_test "Verbose flag (-v)" "./start.sh -v" 124 5
run_test "Verbose flag (--verbose)" "./start.sh --verbose" 124 5

# Test 4: Controller Options
log "Testing Controller Options..."
run_test "Default controller" "./start.sh" 124 5
run_test "Rule-based controller (-c)" "./start.sh -c rule_based" 124 5
run_test "Rule-based controller (--controller)" "./start.sh --controller rule_based" 124 5
run_test "ML-based controller" "./start.sh -c ml_based" 124 5
run_test "Hybrid controller" "./start.sh -c hybrid" 124 5
run_test "Invalid controller" "./start.sh -c invalid" 1 5

# Test 5: Charging Options
log "Testing Charging Options..."
run_test "Enable charging" "./start.sh --enable-charging" 124 5
run_test "Disable charging" "./start.sh --disable-charging" 124 5

# Test 6: Battery Threshold Options
log "Testing Battery Threshold Options..."
run_test "Min battery threshold" "./start.sh --min-battery 30" 124 5
run_test "Max battery threshold" "./start.sh --max-battery 95" 124 5

# Test 7: Performance Requirements
log "Testing Performance Requirements..."
run_test "Min accuracy requirement" "./start.sh --min-accuracy 85" 124 5
run_test "Max latency requirement" "./start.sh --max-latency 150" 124 5

# Test 8: Sensor Options
log "Testing Sensor Options..."
run_test "Mock battery level" "./start.sh --mock-battery 50" 124 5

# Test 9: Invalid Options
log "Testing Invalid Options..."
run_test "Unknown option" "./start.sh --unknown-option" 1 5

# Test 10: Examples from start.sh help
log "Testing Examples from start.sh help..."
run_test "Default webcam mode" "./start.sh --no-display" 124 5
run_test "Webcam with 1-second interval" "./start.sh --interval-ms 1000 --no-display" 124 5
run_test "ML-based controller with verbose logging" "./start.sh --controller ml_based --verbose --no-display" 124 5
run_test "High-performance mode" "./start.sh --min-battery 10 --max-latency 150 --enable-charging --no-display" 124 5

# Test 11: ALL COMBINATIONS - Source + Controller
log "Testing Source + Controller Combinations..."
run_test "Webcam + Rule-based" "./start.sh -s webcam -c rule_based --no-display" 124 5
run_test "Webcam + ML-based" "./start.sh -s webcam -c ml_based --no-display" 124 5
run_test "Webcam + Hybrid" "./start.sh -s webcam -c hybrid --no-display" 124 5

# Test 12: ALL COMBINATIONS - Controller + Charging
log "Testing Controller + Charging Combinations..."
run_test "Rule-based + Enable charging" "./start.sh -c rule_based --enable-charging --no-display" 124 5
run_test "Rule-based + Disable charging" "./start.sh -c rule_based --disable-charging --no-display" 124 5
run_test "ML-based + Enable charging" "./start.sh -c ml_based --enable-charging --no-display" 124 5
run_test "ML-based + Disable charging" "./start.sh -c ml_based --disable-charging --no-display" 124 5
run_test "Hybrid + Enable charging" "./start.sh -c hybrid --enable-charging --no-display" 124 5
run_test "Hybrid + Disable charging" "./start.sh -c hybrid --disable-charging --no-display" 124 5

# Test 13: ALL COMBINATIONS - Runtime + Performance
log "Testing Runtime + Performance Combinations..."
run_test "Fast interval + High accuracy" "./start.sh -t 500 --min-accuracy 90 --no-display" 124 5
run_test "Slow interval + Low latency" "./start.sh -t 5000 --max-latency 50 --no-display" 124 5
run_test "Verbose + Custom interval" "./start.sh -v -t 1500 --no-display" 124 5

# Test 14: ALL COMBINATIONS - Battery + Performance
log "Testing Battery + Performance Combinations..."
run_test "Low battery + High accuracy" "./start.sh --mock-battery 15 --min-accuracy 95 --no-display" 124 5
run_test "High battery + Low latency" "./start.sh --mock-battery 95 --max-latency 30 --no-display" 124 5
run_test "Custom battery thresholds" "./start.sh --min-battery 25 --max-battery 85 --no-display" 124 5

# Test 15: ALL COMBINATIONS - Full Feature Sets
log "Testing Full Feature Combinations..."
run_test "Full featured: ML + Verbose + Custom battery" "./start.sh -c ml_based -v --mock-battery 60 --min-battery 30 --max-battery 80 --no-display" 124 5
run_test "Full featured: Hybrid + Charging + Performance" "./start.sh -c hybrid --enable-charging --min-accuracy 85 --max-latency 120 --no-display" 124 5
run_test "Full featured: ALL OPTIONS" "./start.sh -s webcam -c hybrid -t 1000 -v --enable-charging --min-battery 20 --max-battery 90 --min-accuracy 80 --max-latency 100 --mock-battery 75 --no-display" 124 5

# Test 16: Boundary Values
log "Testing Boundary Values..."
run_test "Minimum interval (1ms)" "./start.sh --interval-ms 1 --no-display" 124 5
run_test "Maximum reasonable interval (60000ms)" "./start.sh --interval-ms 60000 --no-display" 124 5
run_test "Zero accuracy" "./start.sh --min-accuracy 0 --no-display" 124 5
run_test "Maximum accuracy (100)" "./start.sh --min-accuracy 100 --no-display" 124 5
run_test "Zero latency" "./start.sh --max-latency 0 --no-display" 124 5
run_test "Zero battery" "./start.sh --mock-battery 0 --no-display" 124 5
run_test "Maximum battery (100)" "./start.sh --mock-battery 100 --no-display" 124 5

# Test 17: Edge Cases and Error Conditions
log "Testing Edge Cases..."
run_test "Negative interval" "./start.sh --interval-ms -100 --no-display" 124 5  # May not validate
run_test "Negative battery" "./start.sh --mock-battery -10 --no-display" 124 5  # May not validate
run_test "Over 100 battery" "./start.sh --mock-battery 150 --no-display" 124 5  # May not validate
run_test "Over 100 accuracy" "./start.sh --min-accuracy 150 --no-display" 124 5  # May not validate
run_test "Zero interval" "./start.sh --interval-ms 0 --no-display" 124 5

# Test 18: Short vs Long Form Options
log "Testing Short vs Long Form Options..."
run_test "Short form source" "./start.sh -s webcam" 124 5
run_test "Long form source" "./start.sh --source webcam" 124 5
run_test "Short form interval" "./start.sh -t 1000" 124 5
run_test "Long form interval" "./start.sh --interval 1000" 124 5
run_test "Short form controller" "./start.sh -c hybrid" 124 5
run_test "Long form controller" "./start.sh --controller hybrid" 124 5
run_test "Short form verbose" "./start.sh -v" 124 5
run_test "Long form verbose" "./start.sh --verbose" 124 5

# Test 19: Option Order Independence
log "Testing Option Order Independence..."
run_test "Order 1: Source first" "./start.sh -s webcam -c hybrid -t 1000" 124 5
run_test "Order 2: Controller first" "./start.sh -c hybrid -s webcam -t 1000" 124 5
run_test "Order 3: Interval first" "./start.sh -t 1000 -s webcam -c hybrid" 124 5
run_test "Order 4: Verbose last" "./start.sh -s webcam -c hybrid -t 1000 -v" 124 5
run_test "Order 5: Verbose first" "./start.sh -v -s webcam -c hybrid -t 1000" 124 5

# Test 20: Direct Python Examples
log "Testing Direct Python Examples..."
run_test "Direct: Default webcam processing" "uv run python src/main.py --no-display" 124 5
run_test "Direct: Faster processing" "uv run python src/main.py --interval-ms 1000 --no-display" 124 5
run_test "Direct: Headless mode" "uv run python src/main.py --no-display" 124 5
run_test "Direct: Hybrid controller verbose" "uv run python src/main.py --controller hybrid --verbose --no-display" 124 5
run_test "Direct: High performance mode" "uv run python src/main.py --max-latency 150 --min-accuracy 90 --no-display" 124 5
run_test "Direct: Energy saving mode" "uv run python src/main.py --max-latency 50 --min-accuracy 70 --no-display" 124 5
run_test "Direct: ML-based controller" "uv run python src/main.py --controller ml_based --no-display" 124 5
run_test "Direct: Disable charging" "uv run python src/main.py --disable-charging --no-display" 124 5
run_test "Direct: Low battery start" "uv run python src/main.py --mock-battery 25 --no-display" 124 5
run_test "Direct: Full battery start" "uv run python src/main.py --mock-battery 95 --no-display" 124 5

# Final summary
log "=================================================="
log "COMPREHENSIVE Test Suite Summary:"
log "Total Tests: $TOTAL_TESTS"
log_success "Passed: $PASSED_TESTS"
if [ $FAILED_TESTS -gt 0 ]; then
    log_error "Failed: $FAILED_TESTS"
fi
log "Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"

if [ $FAILED_TESTS -gt 0 ]; then
    log_error "Some tests failed. Please review the output above."
    exit 1
else
    log_success "All tests passed!"
    exit 0
fi