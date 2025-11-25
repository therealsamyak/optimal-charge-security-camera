#!/bin/bash

# CI test script for simulation framework

set -e

echo "Running CI tests for simulation framework..."

# Run all tests
uv run pytest tests/ -v --tb=short

echo "CI tests completed!"
