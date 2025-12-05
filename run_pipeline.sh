#!/bin/bash

# Optimal Charge Security Camera - Complete Pipeline Script
# This script runs the full training and simulation pipeline

set -e  # Exit on any error

echo "🚀 Starting Optimal Charge Security Camera Pipeline"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install uv first."
        echo "Visit: https://docs.astral.sh/uv/"
        exit 1
    fi
    print_success "uv is installed"
}

# Check if required files exist
check_files() {
    print_status "Checking required files..."
    
    required_files=(
        "results/power_profiles.json"
        "energy-data/US-CAL-LDWP_2024_5_minute.csv"
        "energy-data/US-FLA-FPL_2024_5_minute.csv"
        "energy-data/US-NW-PSEI_2024_5_minute.csv"
        "energy-data/US-NY-NYIS_2024_5_minute.csv"
        "model-data/model-data.csv"
        "config.jsonc"
    )
    
    missing_files=()
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        print_error "Missing required files:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        exit 1
    fi
    
    print_success "All required files found"
}

# Kill existing tmux sessions
cleanup_tmux() {
    print_status "Cleaning up existing tmux sessions..."
    tmux list-sessions 2>/dev/null | cut -d: -f1 | xargs -I {} tmux kill-session -t {} 2>/dev/null || true
    print_success "Cleaned up tmux sessions"
}

# Step 1: Generate Training Data
generate_training_data() {
    print_status "Step 1: Generating training data with real energy patterns..."
    
    # Kill any existing training session
    tmux kill-session -t training 2>/dev/null || true
    
    # Start training data generation in tmux
    tmux new-session -d -s training "uv run python generate_training_data.py"
    
    print_status "Training data generation started in tmux session 'training'"
    print_status "Monitor with: tmux attach -t training"
    
    # Wait for completion or timeout (30 minutes)
    timeout=1800
    elapsed=0
    
    while [[ $elapsed -lt $timeout ]]; do
        if ! tmux has-session -t training 2>/dev/null; then
            print_success "Training data generation completed!"
            return 0
        fi
        
        # Check progress
        output=$(tmux capture-pane -t training -p 2>/dev/null | tail -5)
        if echo "$output" | grep -q "Training data saved to"; then
            print_success "Training data generation completed!"
            tmux kill-session -t training 2>/dev/null || true
            return 0
        fi
        
        if echo "$output" | grep -q "Error\|Failed\|Exception"; then
            print_error "Training data generation failed!"
            echo "Last output:"
            echo "$output"
            tmux kill-session -t training 2>/dev/null || true
            return 1
        fi
        
        sleep 10
        elapsed=$((elapsed + 10))
        echo -ne "\rProgress: ${elapsed}s / ${timeout}s"
    done
    
    print_error "Training data generation timed out!"
    tmux kill-session -t training 2>/dev/null || true
    return 1
}

# Step 2: Train Custom Controller
train_controller() {
    print_status "Step 2: Training Custom Controller..."
    
    # Check if training data exists
    if [[ ! -f "results/training_data.json" ]]; then
        print_error "Training data not found! Run Step 1 first."
        return 1
    fi
    
    # Kill any existing controller training session
    tmux kill-session -t controller 2>/dev/null || true
    
    # Start controller training in tmux
    tmux new-session -d -s controller "uv run python train_custom_controller.py"
    
    print_status "Controller training started in tmux session 'controller'"
    print_status "Monitor with: tmux attach -t controller"
    
    # Wait for completion or timeout (20 minutes)
    timeout=1200
    elapsed=0
    
    while [[ $elapsed -lt $timeout ]]; do
        if ! tmux has-session -t controller 2>/dev/null; then
            print_success "Controller training completed!"
            return 0
        fi
        
        # Check progress
        output=$(tmux capture-pane -t controller -p 2>/dev/null | tail -5)
        if echo "$output" | grep -q "Training complete"; then
            print_success "Controller training completed!"
            tmux kill-session -t controller 2>/dev/null || true
            return 0
        fi
        
        if echo "$output" | grep -q "Error\|Failed\|Exception"; then
            print_error "Controller training failed!"
            echo "Last output:"
            echo "$output"
            tmux kill-session -t controller 2>/dev/null || true
            return 1
        fi
        
        sleep 10
        elapsed=$((elapsed + 10))
        echo -ne "\rProgress: ${elapsed}s / ${timeout}s"
    done
    
    print_error "Controller training timed out!"
    tmux kill-session -t controller 2>/dev/null || true
    return 1
}

# Step 3: Run Basic Simulation
run_basic_simulation() {
    print_status "Step 3: Running Basic Simulation..."
    
    # Check if controller weights exist
    if [[ ! -f "results/custom_controller_weights.json" ]]; then
        print_error "Controller weights not found! Run Step 2 first."
        return 1
    fi
    
    # Kill any existing simulation session
    tmux kill-session -t simulation 2>/dev/null || true
    
    # Start basic simulation in tmux
    tmux new-session -d -s simulation "uv run python simulation_runner.py"
    
    print_status "Basic simulation started in tmux session 'simulation'"
    print_status "Monitor with: tmux attach -t simulation"
    
    # Wait for completion or timeout (15 minutes)
    timeout=900
    elapsed=0
    
    while [[ $elapsed -lt $timeout ]]; do
        if ! tmux has-session -t simulation 2>/dev/null; then
            print_success "Basic simulation completed!"
            return 0
        fi
        
        # Check progress
        output=$(tmux capture-pane -t simulation -p 2>/dev/null | tail -5)
        if echo "$output" | grep -q "All.*basic simulations completed successfully"; then
            print_success "Basic simulation completed!"
            tmux kill-session -t simulation 2>/dev/null || true
            return 0
        fi
        
        if echo "$output" | grep -q "Error\|Failed\|Exception"; then
            print_error "Basic simulation failed!"
            echo "Last output:"
            echo "$output"
            tmux kill-session -t simulation 2>/dev/null || true
            return 1
        fi
        
        sleep 10
        elapsed=$((elapsed + 10))
        echo -ne "\rProgress: ${elapsed}s / ${timeout}s"
    done
    
    print_error "Basic simulation timed out!"
    tmux kill-session -t simulation 2>/dev/null || true
    return 1
}

# Step 4: Run Batch Simulation (Optional)
run_batch_simulation() {
    print_status "Step 4: Running Batch Simulation (Optional)..."
    
    read -p "Run batch simulation? This takes much longer. (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Skipping batch simulation."
        return 0
    fi
    
    # Kill any existing batch session
    tmux kill-session -t batch 2>/dev/null || true
    
    # Start batch simulation in tmux
    tmux new-session -d -s batch "uv run python batch_simulation.py"
    
    print_status "Batch simulation started in tmux session 'batch'"
    print_status "Monitor with: tmux attach -t batch"
    print_warning "Batch simulation can take 30+ minutes!"
    
    # Wait for completion or timeout (60 minutes)
    timeout=3600
    elapsed=0
    
    while [[ $elapsed -lt $timeout ]]; do
        if ! tmux has-session -t batch 2>/dev/null; then
            print_success "Batch simulation completed!"
            return 0
        fi
        
        # Check progress
        output=$(tmux capture-pane -t batch -p 2>/dev/null | tail -5)
        if echo "$output" | grep -q "All batch simulations completed successfully"; then
            print_success "Batch simulation completed!"
            tmux kill-session -t batch 2>/dev/null || true
            return 0
        fi
        
        if echo "$output" | grep -q "Error\|Failed\|Exception"; then
            print_error "Batch simulation failed!"
            echo "Last output:"
            echo "$output"
            tmux kill-session -t batch 2>/dev/null || true
            return 1
        fi
        
        sleep 30
        elapsed=$((elapsed + 30))
        echo -ne "\rProgress: ${elapsed}s / ${timeout}s"
    done
    
    print_error "Batch simulation timed out!"
    tmux kill-session -t batch 2>/dev/null || true
    return 1
}

# Show results
show_results() {
    print_status "Pipeline Results Summary:"
    echo "=============================================="
    
    # Training data
    if [[ -f "results/training_data.json" ]]; then
        samples=$(python -c "import json; print(len(json.load(open('results/training_data.json'))))" 2>/dev/null || echo "Unknown")
        print_success "Training Data: $samples samples generated"
    else
        print_error "Training Data: Not found"
    fi
    
    # Controller weights
    if [[ -f "results/custom_controller_weights.json" ]]; then
        print_success "Controller Weights: Trained and saved"
    else
        print_error "Controller Weights: Not found"
    fi
    
    # Simulation results
    if ls results/*aggregated-results.csv 1> /dev/null 2>&1; then
        print_success "Simulation Results: CSV files generated"
        echo "  Latest: $(ls -t results/*aggregated-results.csv | head -1)"
    else
        print_error "Simulation Results: Not found"
    fi
    
    echo "=============================================="
    print_status "All results saved in 'results/' directory"
}

# Main execution
main() {
    echo "Optimal Charge Security Camera - Complete Pipeline"
    echo "=============================================="
    echo
    
    # Pre-flight checks
    check_uv
    check_files
    cleanup_tmux
    echo
    
    # Run pipeline steps
    if generate_training_data; then
        echo
        if train_controller; then
            echo
            if run_basic_simulation; then
                echo
                run_batch_simulation
                echo
                show_results
            else
                print_error "Basic simulation failed. Stopping pipeline."
                exit 1
            fi
        else
            print_error "Controller training failed. Stopping pipeline."
            exit 1
        fi
    else
        print_error "Training data generation failed. Stopping pipeline."
        exit 1
    fi
    
    print_success "🎉 Pipeline completed successfully!"
}

# Handle script interruption
trap 'print_warning "Pipeline interrupted by user"; cleanup_tmux; exit 130' INT

# Run main function
main "$@"