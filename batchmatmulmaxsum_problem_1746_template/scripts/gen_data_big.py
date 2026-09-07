import numpy as np
import os
import sys
import time

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

sys.path.insert(0, os.path.dirname(__file__))
from BatchMatmulMaxSum import impl

# ---------------------------------------------------------------------------
# Big-shape smoke cases for LOCAL/cloud timing only (NOT the judge set).
#   row: (B, M, N, K, dtype_name, label, note)
# LOGICAL row-major layout, transposeX1 = transposeX2 = False.
# Sizes are kept moderate so each file pair fits comfortably in device GM.
# cids start at 100 so they never collide with the small gen_data.py set.
# ---------------------------------------------------------------------------
BIG_CASES = [
    (100, 1, 128, 128, 128, "float16",  "c128",   "v1 estimate ~0.2s"),
    (101, 1, 256, 256, 256, "float16",  "c256",   "v1 estimate ~1-3s"),
    (102, 1, 512, 512, 512, "float16",  "c512",   "v1 estimate ~30s+"),
    (103, 1, 256, 256, 512, "bfloat16", "c256k512", "bf16 larger K"),
    (104, 2, 256, 256, 256, "float16",  "b2c256", "multi-batch 256^3"),
    (105, 1, 1024, 1024, 128, "float16", "m1024n1024k128", "wide M/N thin K"),
]

transposeX1 = False
transposeX2 = False

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

manifest_lines = []
t0 = time.time()
for cid, B, M, N, K, dtype_name, label, note in BIG_CASES:
    rng = np.random.default_rng(20260907 + cid)
    in_dir = os.path.join("input", f"case{cid}")
    out_dir = os.path.join("output", f"case{cid}")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    x1 = rng.uniform(-1.0, 1.0, size=(B, M, K)).astype(np.float32)
    x2 = rng.uniform(-1.0, 1.0, size=(B, K, N)).astype(np.float32)
    if dtype_name == "float16":
        x1 = x1.astype(np.float16)
        x2 = x2.astype(np.float16)
        dtype_code = 1
    elif dtype_name == "bfloat16":
        x1 = x1.astype(bfloat16)
        x2 = x2.astype(bfloat16)
        dtype_code = 2
    else:
        raise ValueError(f"unsupported dtype: {dtype_name}")

    x1.tofile(os.path.join(in_dir, "x1.bin"))
    x2.tofile(os.path.join(in_dir, "x2.bin"))

    golden = impl(x1, x2, transposeX1=transposeX1, transposeX2=transposeX2)
    golden.tofile(os.path.join(out_dir, "golden_y.bin"))

    manifest_lines.append(f"{cid} {B} {M} {N} {K} {dtype_code} 0 0")
    print(
        f"case{cid:03d} [{label}]: B={B} M={M} N={N} K={K} {dtype_name} "
        f"macs={B*M*N*K:.3e} x1={x1.shape} x2={x2.shape} — {note}"
    )

with open(os.path.join("input", "cases.txt"), "w") as f:
    f.write("\n".join(manifest_lines) + "\n")

print(f"Generated {len(BIG_CASES)} BIG cases -> input/cases.txt in {time.time()-t0:.1f}s")
print("Next (cloud CANN lab):  time ./build/batch_matmul_max_sum_custom <cid>")
