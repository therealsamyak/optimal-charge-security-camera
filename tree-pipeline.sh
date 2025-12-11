#!/bin/bash

LOCATION=$1

if [ -z "$LOCATION" ]; then
    echo "Usage: $0 <location>"
    exit 1
fi

for config in config2/config*.jsonc; do
    echo "Running tree_search for $LOCATION with $config..."
    uv run tree_search.py --location "$LOCATION" --parallel --config "$config"
done