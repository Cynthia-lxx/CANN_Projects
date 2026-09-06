#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="batch_matmul_max_sum_custom"

# Case ids known to the data chain. MUST stay in sync with main.asc
# (CASE_SHAPES / NUM_CASES) and scripts/gen_data.py (CASES).
ALL_CASES="0 1 2 3 4 5"

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
if [ "$#" -eq 0 ]; then
    RUN_CASES="0"
elif [ "$1" = "all" ]; then
    RUN_CASES="${ALL_CASES}"
else
    RUN_CASES="$*"
fi

overall_pass=1
for cid in ${RUN_CASES}; do
    case_dir="output/case${cid}"
    rm -f "${case_dir}/y.bin" 2>/dev/null || true
    echo "--- case ${cid} ---"
    if timeout 120 "./${OP_NAME}" "${cid}"; then
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
    exit 0
else
    echo "=== SOME CASES FAILED ==="
    exit 1
fi
