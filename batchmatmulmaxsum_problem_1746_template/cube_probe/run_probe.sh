#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "ERROR: ASCEND_HOME_PATH is not set. Please run:"
    echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "or set ASCEND_HOME_PATH to your CANN toolkit path."
    exit 1
fi

echo "=== Set CANN env ==="
source "${ASCEND_HOME_PATH}/set_env.sh"

echo "=== Configure + Build (isolated probe, does not touch v1 build) ==="
rm -rf build
cmake -S . -B build
cmake --build build -j4

echo "=== Run ==="
./build/cube_probe
