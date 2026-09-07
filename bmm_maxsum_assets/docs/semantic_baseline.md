# BatchMatmulMaxSum 语义基线（semantic baseline）

> **唯一事实源**：`p:\Dev\CANN_Learning_Refs\.codebuddy\memory\stage_goal_batchmatmul_maxsum.md`（2026-09-05 题目全文一字不差存档）。本文件是该原文的**语义复述与可执行化**，供用例矩阵、golden 实现与后续 kernel 判定引用；两者冲突时以原文为准。
> 创建：2026-09-05 · 属于父计划 P0（`bmm_maxsum_assets\docs\`）。

---

## 1. 目标与角色定位

本算子融合三段计算：**BatchMatMul → MaxSim（沿 N 取 max）→ SumReduction（沿 M 求和）**，输出 B 个标量。融合动机（题面原文）：减少 Kernel Launch 与中间数据搬运。本文件把该语义以「机器可判定的形式」钉死：`cases/case_matrix.json` 定义用例，`golden/` 给出参照输出，任一实现必须在此基准上通过判据。

## 2. 数学定义（题面公式）

逻辑矩阵固定：`x1 ∈ R^{B×M×K}`、`x2 ∈ R^{B×K×N}`。

- 阶段一（批量矩阵乘）：`A[b,m,n] = Σ_{k=0}^{K-1} x1[b,m,k]·x2[b,k,n]`
- 阶段二（MaxSim）：`R[b,m] = max_{0≤n<N} A[b,m,n]`
- 阶段三（求和归约）：`y[b] = Σ_{m=0}^{M-1} R[b,m]`

约束要点：
- 第 b 个 x1 只与第 b 个 x2 配对（**无 batch broadcast**、无跨 batch 笛卡尔积）；两输入的 B、K 必须相等。
- **归约顺序固定**：Max(N) → Sum(M)，不可交换（先求和再取最大结果不同）。
- 输出 `y.shape = (B,)`，dtype 恒 FLOAT32。

## 3. PyTorch 对标（题面 3.1 原文，即参考算子）

```python
import torch

def batch_matmul_max_sum(x1_logical, x2_logical):
    # x1_logical: [B, M, K]
    # x2_logical: [B, K, N]
    similarity = torch.bmm(
        x1_logical.to(torch.float32),
        x2_logical.to(torch.float32),
    )
    max_sim = torch.amax(similarity, dim=-1)
    return torch.sum(max_sim, dim=-1, dtype=torch.float32)
```

- 输入先升 FP32 再 bmm；amax 沿最后一维（N）；sum 沿 M 且指定 `dtype=torch.float32`。
- **标准 golden**（题面原文）："使用输入实际存储值在 **FP64 精度**下计算，最后转换为 FP32"。
- 因此本地 golden 参考实现 = `FP64 全精度 bmm→max→sum → 结果转 FP32`（严格性 > 题面 fp32 累加要求，作基准；kernel 用 FP32 累加满足判据即可）。

## 4. 布局语义：transposeX1 / transposeX2（只声明 storage shape）

- 逻辑 shape **固定**：x1 `(B,M,K)`、x2 `(B,K,N)`；transpose 属性**不改变**逻辑 shape，也**不要求算子执行转置**，仅说明输入张量在内存中的物理排布。
- 等价观点：读取输入时，`transpose=true` 意味着把 storage 的 (…) 布局按转置后取值映射为逻辑矩阵（对非方阵，storage 每行的步长不同，必须在用例中覆盖非方阵以真正暴露布局）。

| transposeX1 | transposeX2 | x1 storage shape | x2 storage shape | 逻辑取值映射（b 省略） |
|---|---|---|---|---|
| false | false | `(B,M,K)` | `(B,K,N)` | `x1[m][k]=s1[m][k]`；`x2[k][n]=s2[k][n]` |
| false | true | `(B,M,K)` | `(B,N,K)` | `x1[m][k]=s1[m][k]`；`x2[k][n]=s2[n][k]` |
| true | false | `(B,K,M)` | `(B,K,N)` | `x1[m][k]=s1[k][m]`；`x2[k][n]=s2[k][n]` |
| true | true | `(B,K,M)` | `(B,N,K)` | `x1[m][k]=s1[k][m]`；`x2[k][n]=s2[n][k]` |

- 四种组合的输出 y 应**完全一致**（存储布局不参与数学计算，仅影响访存方式）。用例矩阵必须含**非方阵 + 非 1 的 batch** 的四布局对照，才能暴露取值/步长错误（官方 Ex2 使用 2×2 方阵，逻辑值转置后恰好相同，不足以暴露实现错误）。

## 5. dtype / 数值规格

- x1、x2：**同 dtype**，仅 FLOAT16 或 BFLOAT16（题目不支持其它输入 dtype；无 NaN/±Inf/空 tensor/mask）。
- 输入元素**允许为负数**。
- K 维点积、MaxSim、SumReduction：FLOAT32 累加或**等效精度**实现（避免低精度长归约误差）。
- 输出 y：恒 FLOAT32，且**不得含 NaN/±Inf**（输入有限 + 运算有限 ⇒ 理论上必有限；kernel 需防溢出类实现错误）。
- 算子不得修改输入；仅支持连续 ND tensor；多次执行结果一致（确定性）。

## 6. 精度判据（题面第五节原文，逐数据档位）

- float32：相对误差 < 1e-4 **且** 绝对误差 < 1e-4（双万分之一）。
- float16 / bfloat16：相对误差 < 1e-3 **且** 绝对误差 < 1e-3（双千分之一）。
- int32：完全准确。

> **生效档位说明**：BMM 输入固定为 FP16/BF16，故**实际生效判据为双千分之一（1e-3）**；输出虽恒为 FP32，但题面按输入数据类型的误差分档判定，不适用 float32 的 1e-4 档（后者针对"输入即 float32"的算子场景）。文档、脚本与用例矩阵统一按 **1e-3（相对+绝对）** 判定。

## 7. 官方三示例推导（锚点用例，任何实现不得破坏）

### 示例 1：基础计算（transposeX1=false, transposeX2=false）

- `x1: shape (1,2,2)` = `[[[1,0],[0,1]]]`；`x2: shape (1,2,3)` = `[[[1,0,-1],[0,1,0]]]`。
- bmm：`row0 = [1·1+0·0, 1·0+0·1, 1·(-1)+0·0] = [1,0,-1]`；`row1 = [0·1+1·0, 0·0+1·1, 0·(-1)+1·0] = [0,1,0]` ⇒ `A=[[[1,0,-1],[0,1,0]]]`。
- MaxSim（沿 N=3）：`[max(1,0,-1), max(0,1,0)] = [1,1]`。
- Sum（沿 M=2）：`1+1=2` ⇒ **`y=[2.0]`**。✓（题目原文结果）

### 示例 2：转置存储布局（transposeX1=true, transposeX2=true）

- 逻辑内容与示例 1 完全相同（x1 逻辑 `(1,2,2)`、x2 逻辑 `(1,2,3)`）。
- x1 按 storage `(B,K,M)=(1,2,2)` 物理布局输入 = `[[[1,0],[0,1]]]`（2×2 方阵下转置数值相同）。
- x2 按 storage `(B,N,K)=(1,3,2)` 物理布局输入 = `[[[1,0],[0,1],[-1,0]]]`（即逻辑 x2 的转置逐行存放）。
- 计算语义与示例 1 相同 ⇒ **`y=[2.0]`**。✓

### 示例 3：全负相似度（MaxSim 初值不得为 0）

- `x1: shape (1,1,2)` = `[[[1,0]]]`；`x2: shape (1,2,2)` = `[[[-1,-2],[0,0]]]`。
- bmm：`A[0,0,:] = [1·(-1)+0·0, 1·(-2)+0·0] = [-1,-2]`。
- MaxSim：若初值误设为 0 会得 `max(0,-1,-2)=0`；正确为取首元素/负无穷 ⇒ `max(-1,-2) = -1`。
- Sum：单行 M=1 ⇒ **`y=[-1.0]`**。✓（题目原文：不能因错误地将 MaxSim 初值设为 0 而输出 0）

## 8. 需要持续注意的语义陷阱（为 kernel 与验证设计留档）

1. **全负行**：MaxSim 初值 = 负无穷或行首元素，**禁止 0**（示例 3 直击）。
2. **非对齐尾块**：M/N 建议 16 倍数但可能不整除；尾块归约不得引入越界值/初值污染。
3. **同一 m 行被切多个 n 块**（性能路径必然发生）：N 方向块级部分 max 后需对同 m 跨块再取 max（两级 max 结构）。
4. **K 长归约**：FP32/等效精度累加；golden 用 FP64 因而更严格。
5. **布局取数**：transpose 组合必须用非方阵暴露步长错误（见 §4 结论）。
6. **B 维与规模上限**：`1≤B≤64`、`B×M×K≤2²⁸`、`B×N×K≤2²⁸`；B=1 与 B=64、M/N 极端与 K=8 倍数、K=32 下界都要在用例矩阵出现。
7. **fp16 表示范围**：用例数值需在 fp16 有限范围（约 ±65504）内，避免生成即溢出（golden 生成器按 dtype 量化后回读，保证输入本身可表示）。

## 9. 与其它资产的衔接

- 用例全集定义：`cases/case_matrix.md`（人读）/ `cases/case_matrix.json`（机器读）。
- golden 实现：`golden/golden_fp64_stdlib.py`（纯标准库，本地可跑）与 `golden/golden_torch_ref.py`（torch 对拍，需 torch 环境）。
- 硬件侧执行范式：`docs/cube_datapath_notes.md`。
- P3 阶段将以本文件 §6 判据 + `case_matrix.json` + golden 输出组成对标测试集。
