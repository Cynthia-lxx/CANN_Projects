#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ASCEND_HOME_PATH auto-detect (CANN Lab / CANN_Projects template convention):
# this sandbox installs CANN under $HOME/Ascend/cann-9.0.0; /usr/local/Ascend/...
# is NOT present here, so do not hard-require it.
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    for _cand in "$HOME/Ascend/cann-9.0.0" "/usr/local/Ascend/ascend-toolkit/latest" "/usr/local/Ascend/ascend-toolkit"; do
        if [ -f "${_cand}/set_env.sh" ]; then
            ASCEND_HOME_PATH="${_cand}"
            break
        fi
    done
fi

if [ -z "${ASCEND_HOME_PATH:-}" ] || [ ! -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    echo "ERROR: cannot locate CANN toolkit. Tried \$ASCEND_HOME_PATH then:"
    echo "  \$HOME/Ascend/cann-9.0.0, /usr/local/Ascend/ascend-toolkit/{latest,}"
    echo "Set ASCEND_HOME_PATH=/path/to/ascend-toolkit (its set_env.sh must exist)."
    exit 1
fi

echo "=== [1/4] Set CANN env (ASCEND_HOME_PATH=${ASCEND_HOME_PATH}) ==="
source "${ASCEND_HOME_PATH}/set_env.sh"

echo "=== [2/4] Configure + Build (isolated probe, does not touch v1 build) ==="
rm -rf build
cmake -S . -B build
cmake --build build -j4

echo "=== Run cube_probe ==="
./build/cube_probe || true

echo ""
echo "=== Run copy_sem_probe (DataCopy gap semantics direct measurement) ==="
./build/copy_sem_probe || true

echo ""
echo "=== Run off_matmul_abs (F1/F2/base: SetDim, numBlocks) sweep ==="
./build/off_matmul_abs || true

echo ""
echo "=== Run shape_sweep (F4: route1 SetDim==numBlocks, shape grid incl tail/tiny) ==="
./build/shape_sweep || true

echo ""
echo "=== Run cube_iterall (F5: official pure-cube __cube__ IterateAll GM-C, non-constant golden) ==="
./build/cube_iterall || true
