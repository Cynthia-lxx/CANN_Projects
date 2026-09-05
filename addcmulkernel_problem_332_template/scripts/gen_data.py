import numpy as np
import os
import sys

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

sys.path.insert(0, os.path.dirname(__file__))
from addcmul import impl

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


# --- Case 0 ---
np.random.seed(42 + 0)
os.makedirs("input/case0", exist_ok=True)
input_data = np.random.uniform(low=-10, high=10, size=(32)).astype(np.float16)
input_data.tofile("input/case0/input_data.bin")
x1 = np.random.uniform(low=-10, high=10, size=(32)).astype(np.float16)
x1.tofile("input/case0/x1.bin")
x2 = np.random.uniform(low=-10, high=10, size=(32)).astype(np.float16)
x2.tofile("input/case0/x2.bin")
value = np.random.uniform(low=-10, high=10, size=(1)).astype(np.float16)
value.tofile("input/case0/value.bin")



os.makedirs("output/golden_case0", exist_ok=True)
golden = impl(input_data, x1, x2, value)
if golden is not None:
    golden.tofile("output/golden_case0/golden_y.bin")

print(f"Generated test data and golden output for 1 cases.")
