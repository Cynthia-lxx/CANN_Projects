# 判题平台首跑路线与差距核查（BatchMatmulMaxSum / problem_1746）

> 状态：初版（2026-09-06）。目标：诚实地把「当前 M1 本地自测全 PASS」推进到「CANNJudge 判题平台第一次可见提交/跑分」。
> 关联：计划 `PLAN_BatchMatMulMaxSum.md`（M0–M5）、父计划（CANN_Learning_Refs，只读）。

## 1. 事实盘点（证据链）

- 赛题：2026 CANN 挑战赛·上合赛区初赛，题名含「算子核函数工程(beta)」，唯一事实源 `memory/stage_goal_batchmatmul_maxsum.md`（相邻区只读）。
- 判题平台：CANNJudge（https://cannjudge.cn）。每日提交 50 次上限，取最后一次成绩；在线训练期提交不计评奖、用于调试反馈。
- 平台判题机制（据 `CANN_Learning_Hub_(for_dev)/skills/cannjudge-submit/SKILL.md`，属通用 skill，2016-05 更新）：
  - 标准题提交 = 四个文本文件内容：`op_kernel/{op}_tiling.h`、`op_kernel/tiling_key_{op}.h`、`op_host/{op}.cpp`、`op_kernel/{op}.cpp`；平台以 aclnn/标准算子方式编译并判题；每 testcase 返回 `precision_ratio`（精度比例）与 `time`（毫秒）；状态 Running/Accepted/Wrong Answer/Compile Error/Runtime Error/TLE。
  - 平台**不开放测试用例**，必须设计泛化算子（动态 shape/dtype/属性/对齐/边界）。
- 本工程形态（已核实代码）：**Ascend C Direct Invocation「核函数直调」工程**，非 op_host/op_kernel 标准算子工程：
  - `CMakeLists.txt`：`add_executable(batch_matmul_max_sum_custom main.asc)` + 链接 register/tiling_api。
  - `main.asc`：自写 manifest 驱动本地测试 runner（`./input/cases.txt` → 逐 case → `./output/case*/y.bin`）。
  - `kernel.asc`：被 main include，暴露 `run_kernel(x1, info_x1, x2, info_x2, y, info_y, availableCoreNum, stream, transposeX1, transposeX2)`（注意：**run_kernel 已带 transposeX1/X2 参数**，与 332 模板壳一致；M1 对 tx!=false 守卫拒绝）。
  - `scripts/BatchMatmulMaxSum.py` 为**官方提供的 PyTorch golden**（含 transposeX1/X2，逻辑 shape 固定 (B,M,K)/(B,K,N)，物理布局 swap 最后两维）→ 判题用例必然覆盖四存储布局组合。
  - 姊妹题 `addcmulkernel_problem_332_template/` 保留官方直调模板壳（kernel.asc 的 run_kernel 签名含 TensorGroupInfo/availableCoreNum/stream —— 即「核函数工程 beta」题的上传/调用契约骨架）。
- 正确性现状（2026-09-06 云验）：M1 = fp16/bf16 × (B,M,N,K) 中小 shape × **仅 tx=false** × 非对齐 N/M × 全负行 × 多 batch(B=3)，10/10 PASS（diff ≤6.1e-05，判据双 1e-3 达标）。

## 2. 形态假设（需平台页面确认，非本地可判）

- **H1（最可能）**：判题 = 编译本「核函数直调」工程（CMakeLists + main/kernel/data_utils），以隐藏用例按官方模板约定生成 `input/` 数据 → 运行 binary → 读取输出并 verify + 计 time。此时**自写的 manifest main.asc 必须替换/对齐官方直调 main 的数据流约定**（官方 main 从 `./input/*.bin` 单组数据读入、调 `run_kernel(...)`、写 `./output/*.bin`，见 332 模板壳）。
- **H2**：beta 题其实也走 cannjudge 四文件（op_host/op_kernel）——与本直调工程不兼容，需把算法迁成标准算子工程。
- 待用户从平台（题目页/下载包/在线训练提交入口）确认：提交入口形态、输入注入与输出/时序约定、有无“在线训练先行”。

## 3. 差距清单（按阻塞度）

| ID | 差距 | 现状 | 到判题首跑所需 | 预计轮次(云验/迭代) |
|---|---|---|---|---|
| G1 | 判题入口/数据流事实 | 未知（H1/H2 未定） | 平台页面确认；按 H1 对齐 main 数据流 | 0.5–1 天（用户侧） |
| G2 | transpose 四布局 | 守卫拒绝（tx!=false return） | 读入端按 storage layout 解转置 + pad；**大概率需把 cube 层从「手写 tiling 直调」升级到官方高阶 Matmul 模板（transA/transB 由框架加载）**——架构决策 D | 设计 1 轮 + 云验 3–6 轮 |
| G3 | 全维度域 + 预算挡板 | 守卫上限 rowCopies≤2^18、pad≤1GiB；大 B×M 会诚实拒绝（判题视为失败） | 消除「诚实拒绝」：域内任何合法 shape 必须算对。host 逐行 D2D pad 在 B×M 大时行拷贝数爆炸 → 需 kernel 内 pad/DataCopyPad 或整体单 launch 重构 | 2–4 轮（含 8192 级真机） |
| G4 | 多核/单 launch 与测速 | 每 (b,64行块) 3 次 launch×1 核（usedCoreNum==1 断言），串行 | 首跑不需高分，但 TLE 风险与可读 time 需要；与 G2 的 Matmul 升级天然合并（多核网格 B×blocks） | 并入 G2/G3 主线，性能细调为 P4 |
| G5 | 判题账号/上传 | 无 | 用户在平台注册/登录（RSA 密文可后置，模式 A 手工上传亦可） | 0.5 天（用户侧） |
| G6 | 精度域复核 | K≤128 实测；8192 长 K fp32 累加 vs golden(fp64) 未真机 | 域边缘（K8192/B64/s28*）用例上真机 | 并入 G3 |

## 4. 里程碑与时间带（诚实估计）

节奏假设：用户每 1–2 天陪跑一轮云端验证；云端队列正常；G1 尽快确认。

- **M-a「判题平台第一次可见提交/出分」（本次问题所指）**：把 H1/H2 定下来的形态跑通、上传现有正确性内容。乐观 **≈1 周（09/13 前后）**；平均 **≈2 周（09/20 前后）**。前提：G1/G5 用户侧确认快 + 仅 false/false 域上传即可先拿判题反馈（WA 详情逐用例 precision/time 极有价值）。
- **M-b「全约束正确性（四布局×全域）稳定 Accepted/最低可跑分」**：G2+G3 主线完成。乐观 **09/27 前后（=父计划 P3 末）**；保守 **10/01 前**。之后进入 P4 性能（有意义的加速比成绩），缓冲到 10/17 封板富余 ~2.5 周。
- 若中途撞 Cube 大 shape/transpose 范式暗坑或云端排队延长，按每翻车 +1–2 天累加。**不会影响 10/17 截止安全线**。

## 5. 需要用户确认（决定下一步走向）

1. 判题平台该题目的**提交入口**：是上传整个工程（zip/目录，含 CMakeLists+main.asc）还是只传 4 个文本（op_host/op_kernel）？（对应 H1/H2）
2. beta「核函数直调」题的**判题数据流**：平台是否按官方直调 main 约定（`./input/x1.bin/x2.bin` → 调 run_kernel → `./output/y.bin`，单组数据跑一次）？还是别的约定（stdin/stdout、固定文件、每用例独立进程）？
3. 当前是否已到「**在线训练**」阶段、能否先低门槛试提交一次拿判题反馈？

## 6. 下一步技术推进（不依赖 G1 即可开工）

- **主线 = G2 前置：transpose 支持方案定稿 + 四布局本地用例/golden 全覆盖（本地 python 资产可先行）**，随后云端验证。
- 关键架构决策（写入 PLAN_BatchMatMulMaxSum.md D 系列，与父计划 R1 呼应）：cube 层是否升级官方高阶 Matmul（含 transA/transB、batch 支持、多核），替代当前「手写 TCubeTiling 直调 + host pad + per-block 串行 launch」。倾向：**升级**——一次解决 G2/G3/G4 三缺口。
- 预算挡板消除与 8192 级用例进入云端矩阵（本地 golden 已支持：assets 有 k8192_fp16/m8192_bf16/b64_fp16/s28* 等 case 设计）。

## 7. 判题入口已由用户截图确认（H1 坐实，2026-09-06 更新）

- 提交方式：页面内嵌编辑器逐个文件粘贴，无上传/不能删文件，仅「下载空工程模板」；每日 50 次。
- 初始目录 = 空工程模板（同本地 `batchmatmulmaxsum_problem_1746_template` 目录）。
- **官方空模板 `kernel.asc` 入口 = `__global__ __cube__ void batch_matmul_max_sum_custom(...)`**；`This file is #included — do NOT add main()/#pragma once/include guards`；给出 `TensorGroupInfo/TensorInfo` + dtype 枚举（0=fp32 1=fp16 2=bf16 3=int8 4=int16 … 11=bool）。
- **G1 已定（H1）+ 新增 G1'：main 契约偏差**——本地 `main.asc` 是自写 manifest runner（读 cases.txt 多 case），判题平台不会生成 cases.txt，直接上传必 RE。必须改用官方 main 契约（读 input/bin 单组 → 调 kernel → 写 output/bin）。
- **待办（前置）**：用户下载空工程模板，提供官方 `main.asc`（关键）/`kernel.asc`/`data_utils.h`/`run.sh` 内容 → 确认 kernel 调用入口（run_kernel vs direct `<<<>>>`）、transposeX1/X2 传入、输出路径。
- **战术**：契约确认后先提交当前 M1（false/false 域）观察判题输出（每用例 precision_ratio+time 反推隐藏用例画像），再按画像补 transpose/域边界。

## 8. 判题契约完全闭合（官方空模板解析，2026-09-06 更新，commit 6ac7b9f）

- 模板副本：`Projects\batchmatmulmaxsum_empty_template(readonly)\`（main.asc/kernel.asc/data_utils.h/CMakeLists/run.sh/scripts 全）。
- **被测接口 = run_kernel(GM_ADDR x1, TensorGroupInfo x1, GM_ADDR x2, ..., GM_ADDR y, ..., availableCoreNum, stream, transposeX1, transposeX2)**；判题用自己的 host driver 遍历隐藏用例调 run_kernel；main.asc 仅 local test（模板自注）。
- **GM_ADDR = `__gm__ uint8_t*`**（kernel_utils_macros.h 已核）；host 侧需显式 cast。
- 语义/golden：`y[b]=Σ_m max_n Σ_k x1[b,m,k]x2[b,k,n]`；输出 y=(B,) fp32；transpose 仅改物理存储布局（golden swapaxes 还原）。
- 容差档：fp32 y rtol/atol/tol≈1e-4。隐藏用例 = 题目 JSON npu_cases 15 个（模板 BatchMatmulMaxSum.py 注释），分布未知。
- **M1 交付 = 单文件 kernel.asc**（run_kernel + device kernels；无 main/guard），其余可官方原样。
- **已做**：本地 kernel.asc/main.asc 签名与调用对齐官方模板并 push。
- **下一步（首跑）**：判题页粘贴 M1 kernel.asc → 读用例表画像 → 补 transpose/域/大 shape/性能。

## 9. 判题入口规则 + 禁打印重构（2026-09-06 用户实测）

- 提交入口只允许改 `kernel.asc`（其余 7 文件平台锁定 readonly）；新建仅 .asc/.h。
- 平台静态扫描禁 printf/cout/cerr/puts/write/syscall/dlopen/dlsym/asm/freopen（同族 fprintf 等一并禁）。
- 已重构：kernel.asc 删全部 19 处 fprintf（guard 改静默 return），GenBlockTiling 去 typeTag，注释清理；禁词全文扫描 = 0 命中；run_kernel 复核到文件尾闭合。
- 判题模型定案：交付单文件 kernel.asc → 平台私有 host driver 调 run_kernel。首跑待用户贴码。
- 后续（不改动本轮已交语义）：transpose 四布局 → K/域 → 大 shape 预算 → 性能。
