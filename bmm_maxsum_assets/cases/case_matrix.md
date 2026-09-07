# BatchMatmulMaxSum 用例矩阵（人读版）

> 机器可读同源文件：`cases/case_matrix.json`（本表由此生成，字段一一对应；以 JSON 为准）。
> 用途：P3 对标测试集的**用例全集**。判定依据见 `docs/semantic_baseline.md` §6（fp16/bf16 生效档：相对+绝对 < 1e-3）。
> golden 生成：`golden/golden_fp64_stdlib.py`（本地可跑 local 档用例）+ `golden/golden_torch_ref.py`（torch 对拍 / cloud 大用例）。

## 1. 覆盖维度说明

- **锚点（anchor）**：官方三示例，预期输出固定（`expect` 字段），任何实现必须保持；其中示例 K=2 与「32≤K」维度域冲突，为官方演示特例，仍须正确支持（tag `anchor_k_out_of_domain`）。
- **布局等价（layout_equiv）**：同一逻辑内容（M=2,K=3,N=4,B=2，seed 1001）在 transposeX1/X2 **四组合**下的输出必须**完全一致**（存储布局不影响数学结果）；用**非方阵 + batch=2** 暴露取值/步长错误（官方 Ex2 为方阵不足鉴别）。
- **尾块/边界（tail_edge）**：M/N 非 16 倍数、M=1 / N=1、K 下界、全负行结构。
- **规模（scale）**：B/M/K 维度域上下界与 2²⁸ 规模上限代表用例；`runtime=cloud` 用例由 torch 脚本在云端生成（本地 stdlib 不建议跑）。

## 2. 用例表

| id | group | B | M | N | K | dtype | tx1 | tx2 | tags | runtime |
|---|---|---|---|---|---|---|---|---|---|---|
| ex1_anchor | anchor | 1 | 2 | 3 | 2 | fp16 | false | false | anchor; anchor_k_out_of_domain; expect=[2.0] | local |
| ex2_anchor | anchor | 1 | 2 | 3 | 2 | fp16 | true | true | anchor; anchor_k_out_of_domain; expect=[2.0] | local |
| ex3_anchor | anchor | 1 | 1 | 2 | 2 | fp16 | false | false | anchor; anchor_k_out_of_domain; full_negative; expect=[-1.0] | local |
| lay_ff_fp16 | layout_equiv | 2 | 2 | 4 | 3 | fp16 | false | false | layout_ff; align_check | local |
| lay_ff_bf16 | layout_equiv | 2 | 2 | 4 | 3 | bf16 | false | false | layout_ff; align_check | local |
| lay_ft_fp16 | layout_equiv | 2 | 2 | 4 | 3 | fp16 | false | true | layout_ft | local |
| lay_ft_bf16 | layout_equiv | 2 | 2 | 4 | 3 | bf16 | false | true | layout_ft | local |
| lay_tf_fp16 | layout_equiv | 2 | 2 | 4 | 3 | fp16 | true | false | layout_tf | local |
| lay_tf_bf16 | layout_equiv | 2 | 2 | 4 | 3 | bf16 | true | false | layout_tf | local |
| lay_tt_fp16 | layout_equiv | 2 | 2 | 4 | 3 | fp16 | true | true | layout_tt | local |
| lay_tt_bf16 | layout_equiv | 2 | 2 | 4 | 3 | bf16 | true | true | layout_tt | local |
| tail_m_fp16 | tail_edge | 2 | 17 | 16 | 32 | fp16 | false | false | tail_m; k_min | local |
| tail_n_bf16 | tail_edge | 1 | 16 | 33 | 64 | bf16 | false | false | tail_n | local |
| tail_both_fp16 | tail_edge | 3 | 15 | 19 | 40 | fp16 | false | false | tail_m; tail_n | local |
| n1_bf16 | tail_edge | 3 | 32 | 1 | 128 | bf16 | false | false | max_n_single | local |
| m1_fp16 | tail_edge | 4 | 1 | 16 | 64 | fp16 | false | false | max_m_single | local |
| fullneg_fp16 | tail_edge | 2 | 6 | 8 | 32 | fp16 | false | false | full_negative; k_min | local |
| k8192_fp16 | scale | 1 | 16 | 16 | 8192 | fp16 | false | false | k_max | local |
| m8192_bf16 | scale | 1 | 8192 | 16 | 32 | bf16 | false | false | m_max; k_min | local |
| b64_fp16 | scale | 64 | 16 | 16 | 32 | fp16 | false | false | b_max; k_min | local |
| s28a_fp16 | scale | 1 | 4096 | 512 | 256 | fp16 | false | false | scale_2p28 | cloud |
| s28b_bf16 | scale | 8 | 1024 | 1024 | 512 | bf16 | false | false | scale_2p28 | cloud |

> 规模上限校验（题面约束 `B×M×K≤2²⁸`、`B×N×K≤2²⁸`）：
> - `k8192_fp16`：1×16×8192=131072 ✓；`m8192_bf16`：1×8192×32=262144 ✓
> - `b64_fp16`：64×16×32=32768 ✓；`s28a_fp16`：1×4096×256=1048576=2²⁰ ✓
> - `s28b_bf16`：8×1024×512=4194304=2²² ✓（另 B×N×K 同理 ✓）

## 3. 生成规则（golden 实现据此可复现）

1. 逻辑内容生成（`x1:[B,M,K]`、`x2:[B,K,N]` 的**逻辑值序列**）：
   - anchor：采用题面固定值（见 `semantic_baseline.md` §7），不随机。
   - random：seed 固定（layout 组统一 1001），默认 domain `[-2.0, 2.0)`；`fullneg_fp16` 用 `x1∈[0.5,2.0)`、`x2∈[-2.0,-0.5)`（乘积恒负，保证整行全负）。
2. 布局映射：按 `tx1/tx2` 把逻辑值按 storage shape 重排为**存储序列**（即该用例实际输入字节值）；golden 读取时再映射回逻辑值。
3. 量化：存储序列按 dtype（fp16/bf16）取位模式并回读，作为「输入实际存储值」；FP64 计算（bmm→amax→sum）→ FP32 golden。
4. layout_equiv 组：同 shape/seed/dtype 的四布局用例输出必须逐位相等（同一逻辑值集 + 同一 FP64 计算序）。

## 4. 与后续阶段的衔接

- P2（最小正确版）用：ex1_anchor、lay_ff_fp16、k 小对齐用例子集；
- P3（全约束正确性）用：本矩阵全部 local 用例 + 云端 s28 用例；
- P4（性能）用 scale 组（含 k8192_fp16、b64_fp16、s28*）+ 追加与拆分基线同型的大 shape 性能用例（P4 单独扩展）。
