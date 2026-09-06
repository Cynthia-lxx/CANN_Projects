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
# M1 case matrix — single source of truth for shape/data generation.
#   row: (B, M, N, K, dtype_name, x1_range, x2_range, label, note)
# Shapes respect the problem domain: B in [1,64], M/N in [1,8192],
# K in [32,8192] with K % 8 == 0.  All tensors are stored in LOGICAL
# row-major layout (x1 (B,M,K), x2 (B,K,N)) — identical to what the judge
# injects. The kernel does its own zero padding internally.
# dtype_name: "float16" or "bfloat16".  x1_range/x2_range give the uniform
# sampling interval of each input (independent so a row-max-negative test can
# use opposite signs: positive x1 x negative x2 -> all products/row sums < 0).
# ---------------------------------------------------------------------------
CASES = [
    # cid, B,  M,   N,    K,  dtype_name, x1_range,          x2_range,          label,       note
    (0, 1, 64, 64, 64, "float16", [-1.0, 1.0], [-1.0, 1.0], "anchor64", "cube anchor B=1 M=N=K=64 (fp16)"),
    (1, 3, 64, 64, 128, "float16", [-1.0, 1.0], [-1.0, 1.0], "batch3", "multi-batch cube fp16 K=128"),
    (2, 1, 64, 64, 64, "bfloat16", [-1.0, 1.0], [-1.0, 1.0], "bf16cube", "bf16 cube anchor 64^3"),
    (3, 1, 1, 1, 32, "float16", [-1.0, 1.0], [-1.0, 1.0], "singlepoint", "M=N=1 single point (1x1 cube pad)"),
    (4, 1, 2, 3, 40, "float16", [-1.0, 1.0], [-1.0, 1.0], "tinyN3", "tiny N=3 (<8) + K=40 (8-mult not 16)"),
    (5, 1, 100, 64, 64, "float16", [-1.0, 1.0], [-1.0, 1.0], "mtail100", "M=100 tail M-block (M%64=36)"),
    (6, 1, 64, 1000, 64, "float16", [-1.0, 1.0], [-1.0, 1.0], "ntail1000", "N=1000 not 16-aligned tail cols"),
    (7, 1, 3, 4, 32, "float16", [0.1, 0.9], [-0.9, -0.1], "allnegative", "opposite signs -> every C row max is negative (needs -FLT_MAX init, not 0)"),
    (8, 2, 2, 5, 32, "bfloat16", [-1.0, 1.0], [-1.0, 1.0], "bf16tiny", "bf16 non-aligned tiny B=2 M=2 N=5"),
    (9, 1, 64, 64, 40, "float16", [-1.0, 1.0], [-1.0, 1.0], "k40mid", "K=40 medium shape (8-mult not 16)"),
]

transposeX1 = False
transposeX2 = False

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

manifest_lines = []
for cid, B, M, N, K, dtype_name, x1_range, x2_range, label, note in CASES:
    rng = np.random.default_rng(20260906 + cid)
    in_dir = os.path.join("input", f"case{cid}")
    out_dir = os.path.join("output", f"case{cid}")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    lo1, hi1 = x1_range
    lo2, hi2 = x2_range
    x1 = rng.uniform(lo1, hi1, size=(B, M, K)).astype(np.float32)
    x2 = rng.uniform(lo2, hi2, size=(B, K, N)).astype(np.float32)
    if dtype_name == "float16":
        x1 = x1.astype(np.float16)
        x2 = x2.astype(np.float16)
        dtype_code = 1  # TensorInfo.dtype: 1 = fp16
    elif dtype_name == "bfloat16":
        x1 = x1.astype(bfloat16)
        x2 = x2.astype(bfloat16)
        dtype_code = 2  # TensorInfo.dtype: 2 = bf16
    else:
        raise ValueError(f"unsupported dtype: {dtype_name}")

    # Logical row-major binaries (byte count = shape * 2).
    x1.tofile(os.path.join(in_dir, "x1.bin"))
    x2.tofile(os.path.join(in_dir, "x2.bin"))

    # Golden: impl's FP64 matmul -> row max -> sum, cast to fp32.
    golden = impl(x1, x2, transposeX1=transposeX1, transposeX2=transposeX2)
    golden.tofile(os.path.join(out_dir, "golden_y.bin"))

    # Defensive: the allnegative case must really produce negative scores
    # (guards the -FLT_MAX row-max initialisation on the kernel side).
    if label == "allnegative":
        assert np.all(golden < 0), f"case{cid} [{label}] expected negative golden, got min={golden.min()}"

    manifest_lines.append(f"{cid} {B} {M} {N} {K} {dtype_code} 0 0")
    print(
        f"case{cid:02d} [{label}]: B={B} M={M} N={N} K={K} {dtype_name} "
        f"x1={x1.shape} x2={x2.shape} y_range=[{golden.min():.6f}, {golden.max():.6f}] — {note}"
    )

with open(os.path.join("input", "cases.txt"), "w") as f:
    f.write("\n".join(manifest_lines) + "\n")

print(f"Generated test data and golden output for {len(CASES)} cases -> input/cases.txt")
