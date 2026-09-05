import os
import re
from pathlib import Path

import numpy as np


def parse_case_shape():
    header = Path("case_config.h").read_text(encoding="utf-8")
    m = re.search(r"CASE_SHAPE\s*=\s*\{([^}]*)\}", header)
    dims = [int(x) for x in m.group(1).split(",")]
    return tuple(dims)


def clean_bin_files():
    for name in ("inputA.bin", "inputB.bin", "golden.bin", "output.bin"):
        p = Path(name)
        if p.exists():
            p.unlink()


def gen_case_data():
    shape = parse_case_shape()
    rng = np.random.default_rng(1)
    a = rng.uniform(-100.0, 100.0, size=shape).astype(np.float32)
    b = rng.uniform(-100.0, 100.0, size=shape).astype(np.float32)
    golden = (a + b).astype(np.float32)
    a.tofile("inputA.bin")
    b.tofile("inputB.bin")
    golden.tofile("golden.bin")
    print(f"generated inputA.bin/inputB.bin/golden.bin, shape={shape}, dtype=float32")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    clean_bin_files()
    gen_case_data()
