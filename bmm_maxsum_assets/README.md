# BatchMatmulMaxSum 资产目录（bmm_maxsum_assets）

> 本目录是「BatchMatmulMaxSum 赛题（2026 CANN 挑战赛·上合初赛）」的**语义/验证资产区**，对应父计划 P0 阶段产出。
> 存放只读事实源复述、官方三示例推导、用例矩阵与 golden 参考实现；**不含任何算子工程代码**（工程见 `..\batchmatmulmaxsum_problem_1746_template\`）。

## 与父计划的关系

- 父计划：`p:\Dev\CANN_Learning_Refs\.codebuddy\plans\PARENT_PLAN_batchmatmulmaxsum.md`
- 唯一事实源：`p:\Dev\CANN_Learning_Refs\.codebuddy\memory\stage_goal_batchmatmul_maxsum.md`（题目原文一字不差存档）
- 本目录 = 父计划 **P0 地基与语义基线** 的落地物；P1–P5 阶段将引用本目录的用例矩阵与 golden。

## 目录结构

```
bmm_maxsum_assets/
├── README.md                    # 本文件：用途、索引、扩展方式
├── docs/
│   ├── semantic_baseline.md     # 语义基线：公式 / torch 对标映射 / 四存储布局 / 精度判据 / 三示例推导
│   ├── cube_datapath_notes.md   # Cube 数据流与范式理解记录（官方 04/05 章研读产物）
│   └── resources_index.md       # 资源引用索引（教程三镜像 / answer 源码 / ascendc 记忆 / skills）
├── cases/
│   ├── case_matrix.md           # 用例矩阵（人读表格）
│   └── case_matrix.json         # 用例矩阵（机器可读）
└── golden/
    ├── golden_fp64_stdlib.py    # 纯标准库 FP64 参考实现（本地可跑，中小 shape）
    ├── golden_torch_ref.py      # torch 对拍脚本（需 torch 环境）
    └── smoke_outputs/           # 本地冒烟留档
```

## 关键语义速记（详见 semantic_baseline.md）

- 数学式：`y[b] = Σ_m max_n Σ_k x1[b,m,k]·x2[b,k,n]`；PyTorch 等价 `bmm(fp32)→amax(-1)→sum(-1, fp32)`。
- 逻辑 shape 固定 `x1:(B,M,K)`、`x2:(B,K,N)`；`transposeX1/X2` 仅声明 storage shape，无实际转置执行。
- dtype：x1/x2 同为 FP16/BF16，输出 y 恒 FP32；K 点积与归约 FP32 累加或等效精度。
- 输出恒 FP32；精度判据按题面原文：float16/bfloat16 → 相对/绝对误差 < 1e-3（双千分之一）。

## 如何使用 / 如何扩展

1. **语义速查**：先读 `docs/semantic_baseline.md`；任何内核推导若有歧义回到 stage_goal 原文裁定。
2. **选用例**：查 `cases/case_matrix.md`；机器可读版本 `case_matrix.json`（字段见其文件头注释）。
3. **生成 golden**：本地跑 `penv\Scripts\python.exe golden\golden_fp64_stdlib.py`（零依赖）；大 shape / torch 语义对拍用 `golden_torch_ref.py`（需有 torch 的环境，本地无 torch 会友好跳过）。
4. **扩展**：新用例按 `case_matrix.json` 既有字段追加后重跑 golden；三官方示例是**锚点用例**，任何实现变更必须保持其输出不变（Ex1 `y=[2.0]`、Ex2 `y=[2.0]`、Ex3 `y=[-1.0]`）。

## 边界声明

- 本目录**不进入** `batchmatmulmaxsum_problem_1746_template\`（题目工程保持原样）。
- 本目录不属于 CANN_OpHelper 仓库；其代码为**独立参考资产**，不随仓库分发。
- 本地 penv 无 numpy/torch（不代装）；stdlib golden 为纯标准库实现，可用于本地任意回归。
