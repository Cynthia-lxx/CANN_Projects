# -*- coding: utf-8 -*-
"""BatchMatmulMaxSum golden 参考实现 —— 纯标准库、FP64 计算。

语义事实源：docs/semantic_baseline.md（唯一权威为 stage_goal_batchmatmul_maxsum.md）。
语义：y[b] = sum_m max_n sum_k x1[b,m,k]*x2[b,k,n]
标准 golden（题面原文）：使用输入「实际存储值」在 FP64 精度下计算，最后转换为 FP32。

本文件职责：
1) 读取 cases/case_matrix.json；
2) 对每个用例生成逻辑值（fixed 直接取题面值 / random 用确定性种子）；
3) 按 dtype（fp16/bf16）量化得到「实际存储回读值」，并依 transposeX1/X2 派生存储序列；
4) FP64 bmm -> amax(N) -> sum(M)；同时给出 FP64 精确值 y_fp64 与 FP32 golden y_fp32；
5) 断言官方三示例（ex1/ex2/ex3）输出 == 题面 expect；
6) layout_equiv 组同逻辑内容四布局输出逐位一致断言。

零第三方依赖（仅标准库）。本地 penv 可直接运行；cloud 大用例由 golden_torch_ref.py 承担。
"""

import argparse
import json
import math
import os
import random
import struct
import sys

DEFAULT_MATRIX = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases", "case_matrix.json")
)
DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_outputs")

ANCHOR_IDS = ("ex1_anchor", "ex2_anchor", "ex3_anchor")


# ---------------------------------------------------------------------------
# dtype 量化（实际存储位模式 -> 回读的精确 float64 值）
# ---------------------------------------------------------------------------

def quantize_fp16(value: float) -> float:
    """fp16 量化：IEEE binary16 round-to-nearest，回读为 float64。"""
    return struct.unpack("<e", struct.pack("<e", float(value)))[0]


def quantize_bf16(value: float) -> float:
    """bf16 量化：fp32 高 16 位 round-to-nearest-even 截断，回读为 float64。

    依据：对 fp32 位 u，RNE 到高 16 位 = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000。
    对 NaN/Inf 直接原样返回（本题目用例不含，仅防御）。
    """
    f = float(value)
    bits = struct.unpack("<I", struct.pack("<f", f))[0]
    if (bits & 0x7F800000) == 0x7F800000:  # Inf / NaN
        return f
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


QUANTIZERS = {"fp16": quantize_fp16, "bf16": quantize_bf16}


def f64_to_f32(value: float) -> float:
    """FP64 -> FP32（最终 golden 输出；题目输出 dtype 恒 FP32）。"""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


# ---------------------------------------------------------------------------
# 逻辑值生成与 layout 派生
# ---------------------------------------------------------------------------

def _logical_values(case):
    """返回 (x1, x2) 各为扁平逻辑行序 list[float]（已按 dtype 量化回读）。"""
    gen = case["gen"]
    dtype = case["dtype"]
    q = QUANTIZERS[dtype]
    b, m, n, k = case["b"], case["m"], case["n"], case["k"]
    len1, len2 = b * m * k, b * k * n
    if gen["kind"] == "fixed":
        raw1, raw2 = gen["x1"], gen["x2"]
        if len(raw1) != len1 or len(raw2) != len2:
            raise ValueError(
                "case %s: fixed x1 len=%d expected=%d, x2 len=%d expected=%d"
                % (case["id"], len(raw1), len1, len(raw2), len2)
            )
    else:  # random
        rng = random.Random(gen["seed"])
        a1, b1 = gen["x1_domain"]
        a2, b2 = gen["x2_domain"]
        raw1 = [rng.uniform(a1, b1) for _ in range(len1)]
        raw2 = [rng.uniform(a2, b2) for _ in range(len2)]
    return [q(v) for v in raw1], [q(v) for v in raw2]


def logical_as_cube(flat, b, outer, inner):
    """扁平行序 -> 3D 嵌套 list [b][outer][inner]。"""
    out = []
    it = iter(flat)
    for _ in range(b):
        out.append([[next(it) for _ in range(inner)] for _ in range(outer)])
    return out


def storage_sequences(case, x1_flat, x2_flat):
    """把逻辑扁平值按 transposeX1/X2 重排为「存储序列」（输入字节实际顺序）。

    规则（题面 3.3/3.5）：
      tx1=false -> storage row-major (B,M,K)（不变）
      tx1=true  -> storage (B,K,M)：storage[b,k,m] = logic_x1[b,m,k]
      tx2=false -> storage row-major (B,K,N)（不变）
      tx2=true  -> storage (B,N,K)：storage[b,n,k] = logic_x2[b,k,n]
    返回两个扁平 list[float]（已量化回读值）。
    """
    b, m, n, k = case["b"], case["m"], case["n"], case["k"]
    s1, s2 = [], []
    if not case["tx1"]:
        s1 = list(x1_flat)
    else:
        x1c = logical_as_cube(x1_flat, b, m, k)  # [b][m][k]
        for bb in range(b):
            for kk in range(k):
                for mm in range(m):
                    s1.append(x1c[bb][mm][kk])
    if not case["tx2"]:
        s2 = list(x2_flat)
    else:
        x2c = logical_as_cube(x2_flat, b, k, n)  # [b][k][n]
        for bb in range(b):
            for nn in range(n):
                for kk in range(k):
                    s2.append(x2c[bb][kk][nn])
    return s1, s2


# ---------------------------------------------------------------------------
# FP64 计算
# ---------------------------------------------------------------------------

def compute_y64(case, x1_flat, x2_flat):
    """FP64：bmm -> amax(dim=-1) -> sum(dim=-1)。返回 (y64 list per batch, sim3d 可省略)。

    K 维按浮点加法顺序累加（Python float = IEEE double）；两个输入为已量化回读值，
    与题面「输入实际存储值在 FP64 下计算」语义一致。
    """
    b, m, n, k = case["b"], case["m"], case["n"], case["k"]
    x1 = logical_as_cube(x1_flat, b, m, k)
    x2 = logical_as_cube(x2_flat, b, k, n)
    y64 = []
    for bb in range(b):
        total = 0.0
        for mm in range(m):
            row_max = None
            for nn in range(n):
                acc = 0.0
                for kk in range(k):
                    acc += x1[bb][mm][kk] * x2[bb][kk][nn]
                if row_max is None or acc > row_max:
                    row_max = acc
            total += row_max
        y64.append(total)
    return y64


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_matrix(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_case(case):
    """单用例完整管线，返回结果 dict。"""
    x1f, x2f = _logical_values(case)
    s1, s2 = storage_sequences(case, x1f, x2f)
    y64 = compute_y64(case, x1f, x2f)
    y32 = [f64_to_f32(v) for v in y64]
    return {"x1_storage": s1, "x2_storage": s2, "y_fp64": y64, "y_fp32": y32}


def check_anchor(case, result, errors):
    expect = case.get("expect")
    if expect is None:
        return
    y32 = result["y_fp32"]
    if len(y32) != len(expect):
        errors.append("case %s: 输出长度 %d != expect %d" % (case["id"], len(y32), len(expect)))
        return
    for i, (got, exp) in enumerate(zip(y32, expect)):
        if got != exp:
            errors.append("case %s: y_fp32[%d]=%r != expect %r" % (case["id"], i, got, exp))


def check_layout_equiv(matrix, results, errors):
    """layout_equiv 组：同 shape/seed/dtype 的四种 tx 组合输出须逐位一致（FP64 层）。"""
    groups = {}
    for case in matrix["cases"]:
        if case["group"] == "layout_equiv":
            groups.setdefault(case["dtype"], []).append(case)
    for dtype, cases in groups.items():
        if len(cases) != 4:
            errors.append("layout_equiv: dtype %s 应含 4 用例，实际 %d" % (dtype, len(cases)))
            continue
        first = results[cases[0]["id"]]["y_fp64"]
        for case in cases[1:]:
            other = results[case["id"]]["y_fp64"]
            if first != other:
                errors.append(
                    "layout_equiv %s: %s != %s 输出不一致" % (dtype, cases[0]["id"], case["id"])
                )


def select_cases(matrix, scope):
    cases = matrix["cases"]
    if scope == "smoke":
        picked = [c for c in cases if c["id"] in ANCHOR_IDS]
    elif scope == "all_local":
        picked = [c for c in cases if c["runtime"] == "local"]
    elif scope == "all":
        picked = list(cases)
    else:  # 单个 id
        picked = [c for c in cases if c["id"] == scope]
        if not picked:
            raise SystemExit("未找到用例 id: %s" % scope)
    return picked


def main(argv=None):
    parser = argparse.ArgumentParser(description="BatchMatmulMaxSum FP64 stdlib golden")
    parser.add_argument("--matrix", default=DEFAULT_MATRIX, help="case_matrix.json 路径")
    parser.add_argument("--out", default=os.path.join(DEFAULT_OUTDIR, "results.json"))
    parser.add_argument(
        "--scope",
        default="smoke",
        help="smoke(锚点三例) | all_local | all | 单个用例 id；cloud 用例在 stdlib 下会被跳过",
    )
    parser.add_argument("--quiet", action="store_true", help="不打印逐例摘要")
    args = parser.parse_args(argv)

    matrix = load_matrix(args.matrix)
    cases = select_cases(matrix, args.scope)

    results = {}
    errors = []
    skipped = []
    for case in cases:
        if case["runtime"] == "cloud" and args.scope != "all":
            # all 模式下列出但提示云端
            pass
        if case["runtime"] == "cloud":
            skipped.append(case["id"])
            continue
        results[case["id"]] = run_case(case)
        check_anchor(case, results[case["id"]], errors)
        if not args.quiet:
            y32 = results[case["id"]]["y_fp32"]
            print("  [ok] %-16s y_fp32=%s" % (case["id"], y32))

    if any(c["group"] == "layout_equiv" for c in cases):
        check_layout_equiv(matrix, results, errors)

    payload = {"scope": args.scope, "cases": results, "skipped_cloud": skipped}
    outdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(outdir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    if skipped:
        print("  [skip] cloud 用例（stdlib 不跑）: %s" % ", ".join(skipped))

    if errors:
        for e in errors:
            print("  [FAIL] %s" % e, file=sys.stderr)
        raise SystemExit(1)
    print("[PASS] %d 用例通过（含锚点/布局等价断言），结果写入 %s" % (len(results), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
