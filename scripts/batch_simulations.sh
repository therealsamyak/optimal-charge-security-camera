#!/bin/bash

# Batch simulation script for running all combinations of:
# - 4 seasonal days
# - 3 controller types
# - Multiple accuracy/latency threshold combinations
# - 2 image qualities
# - Consistent 10-second intervals for 24 hours

# Don't exit on error - continue with remaining simulations
set +e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log "=" | tr '=' '='
log "Starting Batch Simulation Script"
log "=" | tr '=' '='

log "Step 1: Checking prerequisites..."
if ! command -v uv &> /dev/null; then
    log_error "uv is not installed or not in PATH"
    exit 1
fi
log_success "uv found"

if [ ! -f "src/main_simulation.py" ]; then
    log_error "src/main_simulation.py not found"
    exit 1
fi
log_success "main_simulation.py found"

if [ ! -f "US-CAL-LDWP_2024_5_minute.csv" ]; then
    log_error "US-CAL-LDWP_2024_5_minute.csv not found"
    exit 1
fi
log_success "Energy data CSV found"

if [ ! -f "model-data.csv" ]; then
    log_error "model-data.csv not found"
    exit 1
fi
log_success "Model data CSV found"

log "Step 2: Creating results directory..."
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"
log_success "Results directory: $RESULTS_DIR"

# Seasonal dates
DATES=("2024-01-05" "2024-04-15" "2024-07-04" "2024-10-20")

# Controller types
CONTROLLERS=("custom" "oracle" "benchmark")

# Threshold combinations: (accuracy, latency_ms)
THRESHOLDS=(
    "0.95 5.0"
    "0.90 10.0"
    "0.85 20.0"
    "0.80 30.0"
)

# Image qualities
QUALITIES=("good" "bad")

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Counter
TOTAL=0
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

# Calculate total combinations
TOTAL_COMBOS=$((${#DATES[@]} * ${#CONTROLLERS[@]} * ${#THRESHOLDS[@]} * ${#QUALITIES[@]}))
log "Step 3: Configuration setup"
log "  Seasonal dates: ${#DATES[@]} (${DATES[*]})"
log "  Controller types: ${#CONTROLLERS[@]} (${CONTROLLERS[*]})"
log "  Threshold combinations: ${#THRESHOLDS[@]}"
log "  Image qualities: ${#QUALITIES[@]} (${QUALITIES[*]})"
log "  Output interval: 10 seconds (8640 timesteps per simulation)"
log "  Total combinations: $TOTAL_COMBOS"

# Create temporary config file
log "Step 4: Creating temporary config file..."
TEMP_CONFIG=$(mktemp)
trap "rm -f $TEMP_CONFIG" EXIT
log_success "Temporary config: $TEMP_CONFIG"

log "Step 5: Starting simulations..."
log "=" | tr '=' '='

for date in "${DATES[@]}"; do
    for controller in "${CONTROLLERS[@]}"; do
        for threshold in "${THRESHOLDS[@]}"; do
            read -r accuracy latency <<< "$threshold"
            for quality in "${QUALITIES[@]}"; do
                TOTAL=$((TOTAL + 1))
                
                # Create output filename
                output_file="${RESULTS_DIR}/${date}_${controller}_${accuracy}_${latency}_${quality}.csv"
                
                log "[$TOTAL/$TOTAL_COMBOS] Starting simulation: date=$date, controller=$controller, accuracy=$accuracy, latency=$latency, quality=$quality"
                
                # Create temporary config
                log "  Creating config for this simulation..."
                cat > "$TEMP_CONFIG" <<EOF
{
  "accuracy_threshold": $accuracy,
  "latency_threshold_ms": $latency,
  "simulation": {
    "date": "$date",
    "image_quality": "$quality",
    "output_interval_seconds": 10,
    "controller_type": "$controller"
  },
  "battery": {
    "initial_capacity": 100.0,
    "max_capacity": 100.0,
    "charging_rate": 0.0035,
    "low_battery_threshold": 20.0
  },
  "model_energy_consumption": {
    "YOLOv10-N": 0.004,
    "YOLOv10-S": 0.007,
    "YOLOv10-M": 0.011,
    "YOLOv10-B": 0.015,
    "YOLOv10-L": 0.019,
    "YOLOv10-X": 0.023
  },
  "custom_controller_weights": {
    "accuracy_weight": 0.4,
    "latency_weight": 0.3,
    "energy_cleanliness_weight": 0.2,
    "battery_conservation_weight": 0.1
  },
  "oracle_controller": {
    "optimization_horizon_hours": 24,
    "time_step_minutes": 5,
    "clean_energy_bonus_factor": 1.5
  },
  "benchmark_controller": {
    "prefer_largest_model": true,
    "charge_when_below": 30.0
  }
}
EOF
                
                # Run simulation with timing
                log "  Running simulation command..."
                SIM_START=$(date +%s)
                if uv run python src/main_simulation.py --config "$TEMP_CONFIG" --output "$output_file" > /tmp/sim_output_${TOTAL}.log 2>&1; then
                    SIM_END=$(date +%s)
                    SIM_DURATION=$((SIM_END - SIM_START))
                    COMPLETED=$((COMPLETED + 1))
                    
                    # Calculate progress and ETA
                    ELAPSED=$((SIM_END - START_TIME))
                    AVG_TIME=$((ELAPSED / TOTAL))
                    REMAINING=$((TOTAL_COMBOS - TOTAL))
                    ETA=$((REMAINING * AVG_TIME))
                    ETA_MIN=$((ETA / 60))
                    PROGRESS_PCT=$((TOTAL * 100 / TOTAL_COMBOS))
                    
                    log_success "[$TOTAL/$TOTAL_COMBOS] ($PROGRESS_PCT%) Completed in ${SIM_DURATION}s: $output_file"
                    log "  ETA: ~${ETA_MIN} minutes remaining"
                else
                    SIM_END=$(date +%s)
                    SIM_DURATION=$((SIM_END - SIM_START))
                    FAILED=$((FAILED + 1))
                    log_error "[$TOTAL/$TOTAL_COMBOS] Failed after ${SIM_DURATION}s: $output_file"
                    log_error "  Check /tmp/sim_output_${TOTAL}.log for details"
                fi
                log "  Moving to next simulation..."
            done
        done
    done
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

log "=" | tr '=' '='
log "Batch Simulation Summary"
log "=" | tr '=' '='
log "Total combinations: $TOTAL_COMBOS"
log "Completed: $COMPLETED"
log "Failed: $FAILED"
log "Success rate: $((COMPLETED * 100 / TOTAL_COMBOS))%"
log "Total duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
if [ $COMPLETED -gt 0 ]; then
    AVG_TIME=$((TOTAL_DURATION / COMPLETED))
    log "Average time per simulation: ${AVG_TIME}s"
fi
log "Results saved in: $RESULTS_DIR"

# Log cache statistics if cache utility is available
if command -v python &> /dev/null; then
    log "Cache statistics:"
    python3 -c "
from src.utils.cache import get_cache_stats
stats = get_cache_stats()
print(f'  Energy cache files: {stats[\"energy_cache_files\"]}')
print(f'  Oracle cache files: {stats[\"oracle_cache_files\"]}')
print(f'  Total cache size: {stats[\"total_cache_size_mb\"]:.2f} MB')
" 2>/dev/null || log "  (Cache stats unavailable)"
fi

log "=" | tr '=' '='

