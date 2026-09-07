import numpy as np
import sys
import os

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

# Per-case output specs. Tolerances follow the problem's fp16/bf16 criterion
# (relative & absolute error both 1e-3); the fp16 anchor keeps the tighter
# 1e-4 band from P2b so regressions surface earlier.
# Each spec: (name, dtype, rtol, atol, tol) — tol = max mismatch fraction.
case_output_specs = {
    0: [("y", np.float32, 1e-4, 1e-4, 0.0)],
    1: [("y", np.float32, 1e-3, 1e-3, 0.0)],
    2: [("y", np.float32, 1e-3, 1e-3, 0.0)],
    3: [("y", np.float32, 1e-3, 1e-3, 0.0)],
    4: [("y", np.float32, 1e-3, 1e-3, 0.0)],
    5: [("y", np.float32, 1e-3, 1e-3, 0.0)],
    6: [("y", np.float32, 1e-3, 1e-3, 0.0)],
    7: [("y", np.float32, 1e-3, 1e-3, 0.0)],
    8: [("y", np.float32, 1e-3, 1e-3, 0.0)],
    9: [("y", np.float32, 1e-3, 1e-3, 0.0)],
}


def _big_case_spec(case_id):
    """Generic spec for ad-hoc BIG smoke cases (cid >= 100): y is fp32,
    tolerance uses the fp16/bf16 band per the dtype stored in cases.txt."""
    manifest = "input/cases.txt"
    if os.path.exists(manifest):
        for line in open(manifest):
            tok = line.split()
            if tok and int(tok[0]) == case_id and len(tok) >= 6:
                return [("y", np.float32, 1e-3, 1e-3, 0.0)]
    return None


def known_case_ids():
    """Prefer input/cases.txt; fall back to the spec dict keys."""
    manifest = "input/cases.txt"
    ids = []
    if os.path.exists(manifest):
        for line in open(manifest):
            line = line.strip()
            if not line:
                continue
            ids.append(int(line.split()[0]))
        if ids:
            return sorted(ids)
    return sorted(case_output_specs.keys())


def verify_result(output_path, golden_path, dtype, rtol, atol, tol=0.0):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)
    total_size = golden.size

    if output.size != golden.size:
        if output.size < golden.size:
            golden = golden[:output.size]
            print(f"WARNING: output {output.size} < golden {total_size} elements, truncated golden to match")
        else:
            print(f"FAILED: output has {output.size} elements, golden has {total_size} — size mismatch")
            return False

    missing = total_size - output.size

    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        if missing == 0 and np.array_equal(output, golden):
            print(f"PASSED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
            return True
        diff = np.abs(output.astype(np.int64) - golden.astype(np.int64))
        errors = np.sum(output != golden) + missing
        error_rate = errors / total_size if total_size > 0 else 0.0
        if error_rate <= tol:
            print(f"PASSED (tol={tol}): {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
            print(f"  Mismatched: {errors}/{total_size} ({error_rate*100:.2f}%)")
            return True
        print(f"FAILED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
        print(f"  Mismatched: {errors}/{total_size} ({error_rate*100:.2f}%), tol={tol}")
        print(f"  Max diff: {np.max(diff) if diff.size > 0 else 0.0}")
        return False

    cmp_output = output.astype(np.float32) if dtype == bfloat16 else output
    cmp_golden = golden.astype(np.float32) if dtype == bfloat16 else golden
    isclose = np.isclose(cmp_output, cmp_golden, rtol=rtol, atol=atol, equal_nan=True)
    errors = np.sum(~isclose) + missing
    error_rate = errors / total_size if total_size > 0 else 0.0
    diff = np.abs(cmp_output - cmp_golden)
    if error_rate <= tol:
        print(f"PASSED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
        if errors > 0:
            print(f"  Mismatched: {errors}/{total_size} ({error_rate*100:.2f}%), tol={tol}")
        print(f"  Max diff: {np.max(diff) if diff.size > 0 else 0.0}")
        return True
    else:
        print(f"FAILED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
        print(f"  Mismatched: {errors}/{total_size} ({error_rate*100:.2f}%), tol={tol}")
        print(f"  Max diff: {np.max(diff) if diff.size > 0 else 0.0}")
        return False


def run_case(case_id):
    specs = case_output_specs.get(case_id)
    if specs is None:
        specs = _big_case_spec(case_id)
    if specs is None:
        print(f"Unknown case_id {case_id}. Available: {sorted(case_output_specs.keys())}")
        return False
    output_dir = os.path.join("output", f"case{case_id}")
    all_pass = True
    for name, dtype, rtol, atol, tol in specs:
        output_path = os.path.join(output_dir, name + ".bin")
        golden_path = os.path.join(output_dir, "golden_" + name + ".bin")
        if not os.path.exists(output_path):
            print(f"FAILED: output {name}.bin not found")
            all_pass = False
            continue
        if not os.path.exists(golden_path):
            print(f"FAILED: golden_{name}.bin not found")
            all_pass = False
            continue
        if not verify_result(output_path, golden_path, dtype, rtol, atol, tol):
            all_pass = False
    return all_pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            ids = [int(x) for x in sys.argv[1:]]
        except ValueError:
            print(f"Invalid case ids: {sys.argv[1:]}")
            sys.exit(1)
    else:
        ids = known_case_ids()

    all_pass = True
    for cid in ids:
        print(f"--- verify case {cid} ---")
        if not run_case(cid):
            all_pass = False
    sys.exit(0 if all_pass else 1)
