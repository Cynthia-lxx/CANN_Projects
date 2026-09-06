#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="batch_matmul_max_sum_custom"

# Optional case filter: run.sh [cid ...] runs only the given case ids
# (the binary and verify_result.py both honour a single id when passed).

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "ERROR: ASCEND_HOME_PATH is not set. Please run:"
    echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "or set ASCEND_HOME_PATH to your CANN toolkit path."
    exit 1
fi

echo "=== [1/4] Set CANN env ==="
source "${ASCEND_HOME_PATH}/set_env.sh"

echo "=== [2/4] Build ==="
rm -rf build
mkdir -p build
cd build
cmake ..
make -j4
cd ..

echo "=== [3/4] Gen test data ==="
cd build
python3 ../scripts/gen_data.py

echo "=== [4/4] Run + Verify ==="
overall_pass=1
if [ "$#" -eq 0 ]; then
    echo "Running ALL cases from input/cases.txt ..."
    if timeout 600 "./${OP_NAME}"; then
        if python3 ../scripts/verify_result.py; then
            echo "=== ALL PASSED ==="
        else
            echo "=== FAILED (verify) ==="
            exit 1
        fi
    else
        echo "=== FAILED (kernel exited non-zero or timed out) ==="
        exit 1
    fi
else
    for cid in "$@"; do
        echo "--- case ${cid} ---"
        if timeout 600 "./${OP_NAME}" "${cid}"; then
            if python3 ../scripts/verify_result.py "${cid}"; then
                echo "=== case ${cid}: PASSED ==="
            else
                echo "=== case ${cid}: FAILED (verify) ==="
                overall_pass=0
            fi
        else
            echo "=== case ${cid}: FAILED (kernel exited non-zero or timed out) ==="
            overall_pass=0
        fi
    done
    if [ "${overall_pass}" -eq 1 ]; then
        echo "=== ALL PASSED ==="
    else
        echo "=== SOME CASES FAILED ==="
        exit 1
    fi
fi
