#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""problem_1746 case_matrix 全量本地回归用例生成器。

数据来源: bmm_maxsum_assets/cases/case_matrix.json（23 例，本脚本生成其中
runtime=local 的 20 例 + anchors 3 例 = 21 例；排除 cloud 组 s28a/s28b 大
bf16 规模例，由 gen_data_big.py 方向单独测）。

与 gen_data.py 的关键差异：
  1. 文件里写的是 PHYSICAL(storage) 布局字节 —— 与判题注入完全一致：
     tx1=true 时 x1 按 (B,K,M) 落盘、tx2=true 时 x2 按 (B,N,K) 落盘，
     逻辑 shape 不变 (x1=(B,M,K), x2=(B,K,N))。
  2. cases.txt 每行末尾补 tx1 tx2 两列，main.asc 已支持读取并传给
     run_kernel 的 transposeX1/transposeX2。
  3. golden 由 BatchMatmulMaxSum.impl(物理数组, tx 标志) 计算，保证与
     kernel 侧「先归一化回逻辑布局再算」的语义一致。

验证: 先跑 run.sh 的 10 例回归，再 `python3 scripts/gen_data_casematrix.py`
后跑 ./batch_matmul_max_sum_custom 与 scripts/verify_result.py。
"""
import os
import sys

import numpy as np

try:
    from ml_dtypes import bfloat16
except ImportError:  # pragma: no cover
    bfloat16 = None

sys.path.insert(0, os.path.dirname(__file__))
from BatchMatmulMaxSum import impl


# ---------------------------------------------------------------------------
# cid: 单例字段依次为
#   (cid, B, M, N, K, dtype_name, tx1, tx2, kind, gen_extra, label, note)
#   kind="fixed" : gen_extra = {"x1": <logical flat>, "x2": <logical flat>, "expect": [...]}
#   kind="random": gen_extra = {"seed": int, "x1_domain": [lo, hi], "x2_domain": [lo, hi]}
# 固定值一律按 LOGICAL 展平给出；脚本内部负责按 tx 落成 storage 布局文件。
# ---------------------------------------------------------------------------
def _fixed(*flat, shape):
    a = np.array(list(flat), dtype=np.float32)
    return a.reshape(shape)


CASES = [
    # ---- anchor: 官方示例（固定值，K=2 非 8 倍数允许在演示域）----
    (200, 1, 2, 3, 2, "float16", False, False, "fixed",
     {"x1": [1.0, 0.0, 0.0, 1.0], "x2": [1.0, 0.0, -1.0, 0.0, 1.0, 0.0]},
     "ex1_anchor_ff", "官方示例1 (tx=ff)"),
    (201, 1, 2, 3, 2, "float16", True, True, "fixed",
     {"x1": [1.0, 0.0, 0.0, 1.0], "x2": [1.0, 0.0, -1.0, 0.0, 1.0, 0.0]},
     "ex2_anchor_tt", "官方示例2 (tx=tt, 2x2 对称方阵 x1)"),
    (202, 1, 1, 2, 2, "float16", False, False, "fixed",
     {"x1": [1.0, 0.0], "x2": [-1.0, -2.0, 0.0, 0.0]},
     "ex3_anchor_ff_fullneg", "官方示例3 全负相似度 (MaxSim 初值禁 0)"),
    # ---- layout_equiv: 非方阵 B=2 M=2 K=3 N=4，四 tx 组合 x fp16/bf16 ----
    (203, 2, 2, 4, 3, "float16", False, False, "random",
     {"seed": 1001, "x1_domain": [-2.0, 2.0], "x2_domain": [-2.0, 2.0]},
     "lay_ff_fp16", "layout_equiv ff fp16"),
    (204, 2, 2, 4, 3, "float16", False, True, "random",
     {"seed": 1001, "x1_domain": [-2.0, 2.0], "x2_domain": [-2.0, 2.0]},
     "lay_ft_fp16", "layout_equiv ft (仅 x2 转置) fp16"),
    (205, 2, 2, 4, 3, "float16", True, False, "random",
     {"seed": 1001, "x1_domain": [-2.0, 2.0], "x2_domain": [-2.0, 2.0]},
     "lay_tf_fp16", "layout_equiv tf (仅 x1 转置) fp16"),
    (206, 2, 2, 4, 3, "float16", True, True, "random",
     {"seed": 1001, "x1_domain": [-2.0, 2.0], "x2_domain": [-2.0, 2.0]},
     "lay_tt_fp16", "layout_equiv tt (双转置) fp16"),
    (207, 2, 2, 4, 3, "bfloat16", False, False, "random",
     {"seed": 1001, "x1_domain": [-2.0, 2.0], "x2_domain": [-2.0, 2.0]},
     "lay_ff_bf16", "layout_equiv ff bf16"),
    (208, 2, 2, 4, 3, "bfloat16", False, True, "random",
     {"seed": 1001, "x1_domain": [-2.0, 2.0], "x2_domain": [-2.0, 2.0]},
     "lay_ft_bf16", "layout_equiv ft bf16"),
    (209, 2, 2, 4, 3, "bfloat16", True, False, "random",
     {"seed": 1001, "x1_domain": [-2.0, 2.0], "x2_domain": [-2.0, 2.0]},
     "lay_tf_bf16", "layout_equiv tf bf16"),
    (210, 2, 2, 4, 3, "bfloat16", True, True, "random",
     {"seed": 1001, "x1_domain": [-2.0, 2.0], "x2_domain": [-2.0, 2.0]},
     "lay_tt_bf16", "layout_equiv tt bf16"),
    # ---- tail_edge: 非对齐尾块 + 单元素归约路径 ----
    (211, 2, 17, 16, 32, "float16", False, False, "random",
     {"seed": 2001, "x1_domain": [-2.0, 2.0], "x2_domain": [-2.0, 2.0]},
     "tail_m_fp16", "M=17 尾块 + K=32 下界"),
    (212, 1, 16, 33, 64, "bfloat16", False, False, "random",
     {"seed": 2002, "x1_domain": [-2.0, 2.0], "x2_domain": [-2.0, 2.0]},
     "tail_n_bf16", "N=33 尾块 (bf16)"),
    (213, 3, 15, 19, 40, "float16", False, False, "random",
     {"seed": 2003, "x1_domain": [-2.0, 2.0], "x2_domain": [-2.0, 2.0]},
     "tail_both_fp16", "M=15 N=19 双尾块 + K=40"),
    (214, 3, 32, 1, 128, "bfloat16", False, False, "random",
     {"seed": 2004, "x1_domain": [-2.0, 2.0], "x2_domain": [-2.0, 2.0]},
     "n1_bf16", "N=1 单元素取 max"),
    (215, 4, 1, 16, 64, "float16", False, False, "random",
     {"seed": 2005, "x1_domain": [-2.0, 2.0], "x2_domain": [-2.0, 2.0]},
     "m1_fp16", "M=1 单行求和"),
    (216, 2, 6, 8, 32, "float16", False, False, "random",
     {"seed": 2006, "x1_domain": [0.5, 2.0], "x2_domain": [-2.0, -0.5]},
     "fullneg_fp16", "x1 正 x2 负 → 行 max 全负 (初值禁 0)"),
    # ---- scale 的 local 组：覆盖 K/M/B 上界，但均会被 v1 快速处理 ----
    (217, 1, 16, 16, 8192, "float16", False, False, "random",
     {"seed": 3001, "x1_domain": [-1.0, 1.0], "x2_domain": [-1.0, 1.0]},
     "k8192_fp16", "K=8192 上界长点积"),
    (218, 1, 8192, 16, 32, "bfloat16", False, False, "random",
     {"seed": 3002, "x1_domain": [-1.0, 1.0], "x2_domain": [-1.0, 1.0]},
     "m8192_bf16", "M=8192 行方向 (bf16/v1)"),
    (219, 64, 16, 16, 32, "float16", False, False, "random",
     {"seed": 3003, "x1_domain": [-1.0, 1.0], "x2_domain": [-1.0, 1.0]},
     "b64_fp16", "B=64 batch 上界"),
]


def _to_dtype(a, dtype_name):
    if dtype_name == "float16":
        return a.astype(np.float16)
    if dtype_name == "bfloat16":
        if bfloat16 is None:
            raise RuntimeError("ml_dtypes 未安装，无法生成 bf16 用例")
        return a.astype(bfloat16)
    raise ValueError(f"unsupported dtype: {dtype_name}")


def _build_case(row):
    cid, B, M, N, K, dtype_name, tx1, tx2, kind, extra, label, note = row
    if kind == "fixed":
        x1_logical = np.array(extra["x1"], dtype=np.float32).reshape(B, M, K)
        x2_logical = np.array(extra["x2"], dtype=np.float32).reshape(B, K, N)
    else:
        rng = np.random.default_rng(extra["seed"])
        lo1, hi1 = extra["x1_domain"]
        lo2, hi2 = extra["x2_domain"]
        x1_logical = rng.uniform(lo1, hi1, size=(B, M, K)).astype(np.float32)
        x2_logical = rng.uniform(lo2, hi2, size=(B, K, N)).astype(np.float32)

    x1_logical = _to_dtype(x1_logical, dtype_name)
    x2_logical = _to_dtype(x2_logical, dtype_name)

    # 物理(storage)布局落盘：与判题注入一致。
    x1_storage = np.swapaxes(x1_logical, -1, -2) if tx1 else x1_logical
    x2_storage = np.swapaxes(x2_logical, -1, -2) if tx2 else x2_logical
    return x1_storage, x2_storage


def main():
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    layout_group = {}  # key = (dtype, seed) -> [golden of ff/ft/tf/tt]
    manifest_lines = []
    for row in CASES:
        cid, B, M, N, K, dtype_name, tx1, tx2, kind, extra, label, note = row
        in_dir = os.path.join("input", f"case{cid}")
        out_dir = os.path.join("output", f"case{cid}")
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        x1_storage, x2_storage = _build_case(row)
        x1_storage.tofile(os.path.join(in_dir, "x1.bin"))
        x2_storage.tofile(os.path.join(in_dir, "x2.bin"))

        golden = impl(x1_storage, x2_storage, transposeX1=tx1, transposeX2=tx2)
        golden.tofile(os.path.join(out_dir, "golden_y.bin"))

        # 防御性检查：全负样例的 y 必须 < 0；布局等价组的四个 tx 输出须逐位一致。
        if label.startswith("fullneg"):
            assert np.all(golden < 0), f"case{cid} [{label}] expected negative golden, got min={golden.min()}"
        if label.startswith("lay_"):
            key = (dtype_name, kind, extra.get("seed", 0), B, M, N, K)
            layout_group.setdefault(key, []).append((label, golden))

        dtype_code = 1 if dtype_name == "float16" else 2
        manifest_lines.append(f"{cid} {B} {M} {N} {K} {dtype_code} {1 if tx1 else 0} {1 if tx2 else 0}")
        print(
            f"case{cid} [{label}]: B={B} M={M} N={N} K={K} {dtype_name} "
            f"tx=({int(tx1)},{int(tx2)}) y_range=[{golden.min():.6f}, {golden.max():.6f}] — {note}"
        )

    for key, items in layout_group.items():
        if len(items) < 4:
            continue
        ref = None
        for label, g in items:
            if ref is None:
                ref = g
            else:
                assert np.array_equal(ref, g), (
                    f"layout_equiv 组 {key} 输出不一致: {items[0][0]} vs {label}"
                )
        print(f"layout_equiv 组 {key[0]}: 四 tx 输出逐位一致 ✓ ({items[0][0]})")

    with open(os.path.join("input", "cases.txt"), "w") as f:
        f.write("\n".join(manifest_lines) + "\n")
    print(f"Generated {len(CASES)} case_matrix local cases -> input/cases.txt")


if __name__ == "__main__":
    main()
