#!/usr/bin/env bash
#
# Run the project's test suite via offload (parallel on Modal).
# Requires: offload (cargo install offload@0.3.0), Modal CLI + credentials
#
set -euo pipefail

if ! command -v offload &> /dev/null; then
    echo "Error: 'offload' not installed. Install with: cargo install offload@0.3.0"
    exit 1
fi

cd "$(git rev-parse --show-toplevel)"
exec offload run --copy-dir ".:/app" "$@"
