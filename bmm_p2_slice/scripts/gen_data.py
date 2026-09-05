import numpy as np
import os
import sys

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

sys.path.insert(0, os.path.dirname(__file__))
from BatchMatmulMaxSum import impl

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


# --- Case 0 (P2b: real M/N vertical slice, cube MatMul + vector reduction) ---
# B=1, M=N=K=64, all 16-aligned; fp16 x fp16 -> fp32 y[b]=max_i(sum_j C[b][i][j]).
CASE0_M = 64
CASE0_N = 64
CASE0_K = 64
np.random.seed(42 + 0)
os.makedirs("input/case0", exist_ok=True)
x1 = np.random.uniform(low=-1.0, high=1.0, size=(1, CASE0_M, CASE0_K)).astype(np.float16)
x1.tofile("input/case0/x1.bin")
x2 = np.random.uniform(low=-1.0, high=1.0, size=(1, CASE0_K, CASE0_N)).astype(np.float16)
x2.tofile("input/case0/x2.bin")

transposeX1 = False
transposeX2 = False

os.makedirs("output/golden_case0", exist_ok=True)
golden = impl(x1, x2, transposeX1=transposeX1, transposeX2=transposeX2)
if golden is not None:
    golden.tofile("output/golden_case0/golden_y.bin")

print(f"Generated test data and golden output for 1 cases.")
