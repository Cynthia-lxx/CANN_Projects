import numpy as np
import os
import sys

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

sys.path.insert(0, os.path.dirname(__file__))
from BatchMatmulMaxSum import impl

# ---------------------------------------------------------------------------
# CASE TABLE — single source of truth for shape/data generation.
#   id: (B, M, N, K, label, note)
# IMPORTANT: keep in sync with the case table inside kernel.asc (run_kernel's
# caller switch) and main.asc. Shapes respect the problem domain:
#   B in [1,64], M/N in [1,8192], K in [32,8192] with K % 8 == 0.
# K=40 covers the "8-multiple but not 16-multiple" tail segment; M=1000 / N=1000
# cover non-16-aligned tail blocks.
# ---------------------------------------------------------------------------
CASES = {
    0: (1, 64, 64, 64, "anchor64", "P2b regression anchor: B=1 M=N=K=64"),
    1: (1, 1, 1, 64, "singlepoint", "P2a-style single point M=N=1 (vector dot path)"),
    2: (8, 64, 64, 64, "batch8", "B=8 > 1, 16-aligned"),
    3: (1, 64, 64, 40, "k8x5", "K=40 (8-multiple, not 16-multiple) tail segment"),
    4: (1, 1000, 64, 64, "mtail1000", "M=1000 non-16-aligned tail rows"),
    5: (1, 64, 1000, 64, "ntail1000", "N=1000 non-16-aligned tail columns"),
}

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

transposeX1 = False
transposeX2 = False

for cid, (B, M, N, K, label, note) in sorted(CASES.items()):
    np.random.seed(42 + cid)
    in_dir = os.path.join("input", f"case{cid}")
    out_dir = os.path.join("output", f"case{cid}")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    x1 = np.random.uniform(low=-1.0, high=1.0, size=(B, M, K)).astype(np.float16)
    x1.tofile(os.path.join(in_dir, "x1.bin"))
    x2 = np.random.uniform(low=-1.0, high=1.0, size=(B, K, N)).astype(np.float16)
    x2.tofile(os.path.join(in_dir, "x2.bin"))

    golden = impl(x1, x2, transposeX1=transposeX1, transposeX2=transposeX2)
    if golden is not None:
        golden.tofile(os.path.join(out_dir, "golden_y.bin"))

    print(
        f"case{cid} [{label}]: B={B} M={M} N={N} K={K} x1={x1.shape} x2={x2.shape} golden={golden.shape} — {note}"
    )

print(f"Generated test data and golden output for {len(CASES)} cases.")
