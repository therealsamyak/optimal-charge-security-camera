#!/bin/bash

# Test script for simulation framework

set -e

echo "Running simulation framework tests..."

# Run unit tests
echo "Running unit tests..."
uv run pytest tests/test_simulation_units.py -v

# Run integration tests
echo "Running integration tests..."
uv run pytest tests/test_simulation_integration.py -v

# Run scenario tests
echo "Running scenario tests..."
uv run pytest tests/test_simulation_scenarios.py -v

echo "All tests completed!"
