#!/bin/bash

LOCATION=$1
OUTPUT=$2

if [ -z "$LOCATION" ]; then
    echo "Usage: $0 <location> [output_folder]"
    exit 1
fi

# Set default output folder if not provided
if [ -z "$OUTPUT" ]; then
    OUTPUT="results2"
fi

for config in config3/config*.jsonc; do
    echo "Running tree_search for $LOCATION with $config..."
    uv run tree_search.py --location "$LOCATION" --parallel --config "$config" --output "$OUTPUT"
done