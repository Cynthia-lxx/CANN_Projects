# -*- coding: utf-8 -*-
"""BatchMatmulMaxSum torch 对拍参考脚本（需 torch 环境）。

用途：
1. 用与 golden_fp64_stdlib.py 完全一致的逻辑值生成/量化/布局派生，把「输入实际存储值」
   喂给 torch 做 FP64 bmm->amax->sum，得到 torch 参照 y_fp64 / y_fp32；
2. 与 stdlib 输出（--compare-stdlib-out）做 isclose 交叉对拍；
3. cloud 大 shape 用例（stdlib 纯 Python 太慢）在此环境生成 golden。

本地 penv 无 torch：脚本友好提示后以退出码 0 跳过（不报错）。
"""

import argparse
import json
import os
import sys

try:
    import torch
except ImportError:
    print(
        "[skip] 本机无 torch：golden_torch_ref.py 需在装有 PyTorch 的环境运行"
        "（例如云端 CANN Lab / 你的开发机）。stdlib golden 不受影响。",
        file=sys.stderr,
    )
    raise SystemExit(0)

# 复用 stdlib 的确定性生成/量化/布局派生，保证两脚本输入字节一致。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_fp64_stdlib as base  # noqa: E402


def run_case_torch(case):
    """返回 dict：x1/x2 存储序列与 torch 计算的 y_fp64/y_fp32。"""
    dtype = case["dtype"]
    b, m, n, k = case["b"], case["m"], case["n"], case["k"]
    x1f, x2f = base._logical_values(case)
    s1, s2 = base.storage_sequences(case, x1f, x2f)
    # 存储回读值 -> float64 tensor；逻辑形状由 layout 映射决定（s1 已含 tx 语义：
    # 直接以逻辑 shape 重塑即可，因为 _logical_values 输出的就是逻辑值，
    # storage_sequences 仅给出「字节顺序」，此处仍需从逻辑序列重塑。
    x1 = torch.tensor(x1f, dtype=torch.float64).view(b, m, k)
    x2 = torch.tensor(x2f, dtype=torch.float64).view(b, k, n)
    sim = torch.bmm(x1, x2)
    row_max = torch.amax(sim, dim=-1)
    y64 = torch.sum(row_max, dim=-1, dtype=torch.float64)
    y32 = y64.to(torch.float32)
    return {
        "x1_storage": s1,
        "x2_storage": s2,
        "y_fp64": [float(v) for v in y64.tolist()],
        "y_fp32": [float(v) for v in y32.tolist()],
    }


def isclose_lists(a, b, rtol=1e-12, atol=1e-12):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if not (abs(x - y) <= atol + rtol * abs(y)):
            return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="BatchMatmulMaxSum torch FP64 参照/对拍")
    parser.add_argument("--matrix", default=base.DEFAULT_MATRIX, help="case_matrix.json 路径")
    parser.add_argument("--out", default=os.path.join(base.DEFAULT_OUTDIR, "torch_results.json"))
    parser.add_argument("--scope", default="smoke", help="smoke | all_local | all | 单个 id")
    parser.add_argument("--compare-stdlib-out", default=None, help="与 stdlib results.json 交叉对拍")
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args(argv)

    matrix = base.load_matrix(args.matrix)
    cases = base.select_cases(matrix, args.scope)

    results = {}
    for case in cases:
        if case["runtime"] == "cloud" and args.scope != "all":
            continue
        results[case["id"]] = run_case_torch(case)
        print("  [ok] %-16s y_fp32=%s" % (case["id"], results[case["id"]]["y_fp32"]))

    outdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(outdir, exist_ok=True)
    payload = {"scope": args.scope, "cases": results}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    status = "[PASS]"
    if args.compare_stdlib_out:
        with open(args.compare_stdlib_out, "r", encoding="utf-8") as fh:
            std = json.load(fh)
        common = sorted(set(results) & set(std.get("cases", {})))
        for cid in common:
            if not isclose_lists(
                results[cid]["y_fp64"],
                std["cases"][cid]["y_fp64"],
                rtol=args.rtol,
                atol=args.atol,
            ):
                status = "[DIFF]"
                print("  %s %s: torch=%s stdlib=%s"
                      % (status, cid, results[cid]["y_fp64"], std["cases"][cid]["y_fp64"]))
        print("  %s torch vs stdlib 对拍：公共用例 %d 个" % (status, len(common)))
    print("%s 完成，结果写入 %s" % (status, args.out))
    return 0 if status == "[PASS]" else 1


if __name__ == "__main__":
    raise SystemExit(main())
