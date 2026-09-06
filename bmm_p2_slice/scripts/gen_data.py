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
# IMPORTANT: keep in sync with the case table inside main.asc (CASE_SHAPES).
# Shapes respect the problem domain:
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

# ---------------------------------------------------------------------------
# DISK (device) layout vs LOGICAL shape — route ③ (bounded-C dual kernel).
#
# kernel.asc reads x1/x2 from GM with ZERO-PADDED strides:
#   x1.bin  : (B, Mpad, Kpad)  fp16  — Mpad = ceil(M/64)*64 rows (tail rows 0)
#             Kpad = ceil(K/16)*16 halves per row (tail columns 0)
#   x2.bin  : (B, Kpad, N)     fp16  — K rows padded to Kpad (tail rows 0),
#             N is NOT padded (only every row's length matters there).
#
# The pad values are exact zeros, so a padded element contributes exactly 0 to
# the fp32 accumulation and the result equals the LOGICAL computation. golden
# below is therefore computed on the LOGICAL arrays. The pad multiples must
# match kernel.asc's kPadM (64) / kPadK (16).
# ---------------------------------------------------------------------------
def pad_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


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

    # Logical (truth) tensors — these drive the golden below.
    x1 = np.random.uniform(low=-1.0, high=1.0, size=(B, M, K)).astype(np.float16)
    x2 = np.random.uniform(low=-1.0, high=1.0, size=(B, K, N)).astype(np.float16)

    golden = impl(x1, x2, transposeX1=transposeX1, transposeX2=transposeX2)
    if golden is not None:
        golden.tofile(os.path.join(out_dir, "golden_y.bin"))

    # Padded disk layout (route ③). Tail rows/columns are exact zeros.
    Mpad = pad_up(M, 64)
    Kpad = pad_up(K, 16)
    x1p = np.zeros((B, Mpad, Kpad), dtype=np.float16)
    x1p[:, :M, :K] = x1
    x1p.tofile(os.path.join(in_dir, "x1.bin"))
    x2p = np.zeros((B, Kpad, N), dtype=np.float16)
    x2p[:, :K, :] = x2
    x2p.tofile(os.path.join(in_dir, "x2.bin"))

    print(
        f"case{cid} [{label}]: B={B} M={M} N={N} K={K} "
        f"(disk Mpad={Mpad} Kpad={Kpad}) x1={x1p.shape} x2={x2p.shape} golden={golden.shape} — {note}"
    )

print(f"Generated test data and golden output for {len(CASES)} cases.")
