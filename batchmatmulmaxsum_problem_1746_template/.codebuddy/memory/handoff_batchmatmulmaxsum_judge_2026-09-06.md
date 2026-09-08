# HANDOFF — BatchMatMulMaxSum 判题冲刺交接（2026-09-06 深夜生成）

> 本文件是「2026 CANN 挑战赛·上合赛区(初赛) BatchMatMulMaxSum」判题冲刺任务的**完整交接文档**，
> 供下一个接手 Agent（或下一会话）在不重读全部历史日志的前提下无缝续跑。
> 本文件自包含要点 + 指向既有详档（plans/ 与 memory/ 各文件），冲突时以本文件为准并回查原文。
> 会话级逐日过程记录：`2026-09-06.md`（本工作区 memory/，同日多轮，**保留为过程档案**）。

---

## 0. 任务一句话

把 `Projects/batchmatmulmaxsum_problem_1746_template/kernel.asc` 做到能在判题平台（答题页内嵌编辑器，只允许编辑 kernel.asc）上通过 profiling 硬规则并拿到正确输出，最终在 15 个隐藏测试点稳定得分；当前**卡在 profiling 硬规则未过**（占位版 Got 20 / Expected 75），下一步需先做最小 launch 计数探针实验定位统计口径，再走单 launch 融合 kernel 方案。

---

## 1. 赛题事实（已多轮核实的客观语义）

- 题目名：BatchMatMulMaxSum（模板目录 problem_1746）。
- 算子逻辑（golden，来自模板 `scripts/BatchMatmulMaxSum.py`）：
  `y[b] = Σ_{m=0..M-1} max_{n=0..N-1} Σ_{k=0..K-1} x1[b,m,k]·x2[b,k,n]`
  = `np.matmul(x1,x2)`(B,M,N) → `amax(axis=-1)` → `sum(axis=-1)`。
- 输出 `y = (B,)` float32，B 个标量。
- 输入 dtype：仅 fp16(1)/bf16(2)，x1/x2 dtype 必须相同；y dtype fp32(0)。（模板 dtype 枚举 0=fp32 1=fp16 2=bf16）
- 输入逻辑 shape：x1=(B,M,K)，x2=(B,K,N)，均 rank3；y rank1。
- 维度域（题目 JSON）：B∈[1,64]，M/N∈[1,8192]，K∈[32,8192] 且 K 为 8 的倍数；规模 ≤ 2^28 × B 侧（预算挡板，host 侧实现）。
- 四存储布局组合：`transposeX1` 使 x1 物理为 (B,K,M)；`transposeX2` 使 x2 物理为 (B,N,K)（golden 先 swapaxes 还原逻辑）。**判题必然覆盖四布局**，目前代码只支持 false/false。
- 判题容差（本地 verify_result.py case spec）：y fp32，`rtol=atol=tol=1e-4`，允许 ≤0.01% 元素失配。注意存在"真 0 邻域"用例（对称抵消 → max 翻转风险区），仅靠 tol 档容错。

---

## 2. 判题平台契约（三层证据闭合，高置信）

### 2.1 被测接口 = run_kernel（平台私有 host driver 调用）

```cpp
extern "C" void run_kernel(GM_ADDR x1, const TensorGroupInfo& info_x1,
                           GM_ADDR x2, const TensorGroupInfo& info_x2,
                           GM_ADDR y, const TensorGroupInfo& info_y,
                           int64_t availableCoreNum, aclrtStream stream,
                           bool transposeX1, bool transposeX2)
```

- `GM_ADDR` = `__gm__ uint8_t*`（`kernel_utils_macros.h`，已核对头库）。host 侧如需 ACL API（void* 参数）不能把 __gm__ 指针 reinterpret_cast 成 void*（编译器禁止，见 §5 教训）；内部中间量一律 plain `uint8_t*`，仅 __global__ kernel 形参写 __gm__。plain→__gm__ 在 launch 传参可隐式转换（本 TU 编译已证）。
- `main.asc` 只是**本地冒烟样例**（模板字面注释 "Local test main — runs the first test case"，读 `./input/x1.bin`/`x2.bin` → 调 run_kernel → 写 `./output/y.bin`），判题不依赖它。判题 = 平台 host driver 遍历隐藏用例调 run_kernel。

### 2.2 平台实测规则（用户 2026-09-06 晚逐条确认）

1. 答题页只允许编辑 `kernel.asc`；模板其余 7 文件（main.asc/data_utils.h/CMakeLists.txt/run.sh/scripts/*）全部 **readonly 锁定**。
2. 上传代码做**静态扫描**，禁输出/system 类 API（提示原文列举：`printf/cout/cerr/puts/write/syscall/dlopen/dlsym/asm/freopen`；同族的 fprintf/fputs/fopen/stdout/stderr/fwrite 一律视为被禁）。提交前必须全文扫零命中。
3. "新建文件"入口只允许 `.asc` 与 `.h` 后缀（多文件交付理论可行，未用过）。
4. 判题环境**无任何输出通道**；域外用例 guard 只能**静默 return**。Debug 只能走本地 main.asc + 云端 CANN Lab。
5. 每日 50 次提交上限，取最后一次。
6. **Profiling 硬规则（本轮卡点）**：原文错误
   `Profiling rule violated: each iteration must launch exactly 1 kernel. Expected 75 launches, got N.`
   两次数值：首跑(M1 多 launch 版) `got 2065`；占位版(每 run_kernel 恰 1 launch) **`got 20`**；Expected **恒为 75**。
   → 反证该检查**不只是**"每 iteration ≤1 kernel"，统计口径未明（见 §8 疑点与下轮实验），占位未解决它。

### 2.3 隐藏用例

15 个 npu_cases（`scripts/BatchMatmulMaxSum.py` 注释："15 个测试用例，与 JSON 中的 npu_cases 一一对应"），具体 shape/dtype/transpose 分布未知；判题首跑反馈（precision_ratio/time/用例表）是唯一画像来源——但目前 profiling 检查先于结果返回，用例表尚未见到。

---

## 3. 物理资产地图（本地工作区）

### 3.1 工程目录（Projects/）

| 路径 | 角色 | 状态 |
|---|---|---|
| `Projects/batchmatmulmaxsum_problem_1746_template/` | **主交付工程**（改这里） | kernel.asc=占位版（见 §4） |
| `Projects/batchmatmulmaxsum_empty_template(readonly)/` | 官方空模板下载件（契约原文证据） | 只读 |
| `Projects/addcmulkernel_problem_332_template/` | 姊妹"官方直调壳 main"参考 | 只读参考 |
| `Projects/bmm_p2_slice/` | P3 经验工程（多片/分块归约），只读避坑参考 | 只读 |
| `Projects/bmm_maxsum_assets/` | 赛题 JSON 与本地资产（5 *.md, 3 *.json, 2 *.py） | 参考 |
| `Projects/out/` | 云端产物/历史输出（54 文件） | 参考 |

`Projects/.git` 是项目级 git 仓（远程 github.com/Cynthia-lxx/CANN_Projects.git）；`.codebuddy/` 记忆不在该仓范围。

### 3.2 .codebuddy 计划/记忆（memory/ + plans/）

- plans（过程决策详档）：
  - `PLAN_BatchMatMulMaxSum.md` —— 总攻关计划（里程碑 M0~M5、决策 D1~D5、风险清单）。
  - `digest_batchmatmul_maxsum_m0m1.md` —— M0/M1 架构 digest（与工程代码同步）。
  - `judge_roadmap_batchmatmulmaxsum.md` —— 判题差距路线 G1~G6 与时间带评估。
  - `batchmatmul-maxsum-m0-m1-first-code_*.md`、`核心课程通读与AscendC知识沉淀_*.md` —— 里程碑/研读记录。
- memory 知识体系（六主题 + 索引 + 实践，见 MEMORY.md 地图）：`ascendc_01~07_*.md`、`ascendc_knowledge_index.md`。
- memory 日度：`2026-09-05.md`、`2026-09-06.md`（本任务同日多轮全过程，权威过程史）。
- memory 赛题目标：`stage_goal_batchmatmul_maxsum.md`。
- 本交接文件：`handoff_batchmatmulmaxsum_judge_2026-09-06.md`。

### 3.3 共享/规则

- 邻项目 `..\CANN_Learning_Refs\`：记忆文件共享（只读访问，不写入；其工具可能把本项目当试验场，Projects/ 突发改动属正常）。
- 权威头库：`cann_ascendc_headers_9.0.0\`（CANN 9.0.0，写 Ascend C 代码前必查 API 签名；规则见 workspace rules）。
- 官方文档：`Reference_And_Documentation\`。

---

## 4. kernel.asc 版本演进史（本任务核心现场）

| 版本 | 何时 | 架构 | launch 数/iteration | 静态 | 实测结果 |
|---|---|---|---|---|---|
| **V0 初始** | M1 前 | 早期骨架 | — | — | 未上判题 |
| **M1**（多 launch） | 深夜首跑 | host 三层循环：逐 (batch, 64 行块) `bmm_cube_fp16/bf16<<<1>>>` + `bmm_tile_reduce<<<1>>>`，逐 batch `bmm_part_sum<<<1>>>`；host 内做 pad（aclrtMemsetAsync 全零 GM 工作区 aPad/bPad/bias/part + 行级 D2D aclrtMemcpyAsync）；Kpad=PadUp(K,16)、Mpad=PadUp(M,64)、Npad=PadUp(N,64)；bias 全 0；tiling=GenBlockTiling 探针梯（SetDim(1)+SetBias+SetBufferSpace，accept 条件收进循环：usedCoreNum==1 && tiling.M/N==真 shape 等） | 几十/次 | 本地冒烟 10/10 PASS；禁词 0 | **15 全 RE：Expected 75, got 2065** |
| **占位版**（当前磁盘态） | 用户选 C 后 | 删三层 launch 循环 → 单 launch 多核占位 `bmm_placeholder<<<coreCnt,...>>>(yBase, B, coreCnt)`（coreCnt=min(B,availableCoreNum)，每核 for 步进写 y[b]=0.0f，复用 DataCopyPad{1,4,0,0,0} 4B 写 GM 体例）；**保留** host guard/解析/budget/GenBlockTiling+校验/全部 pad malloc+memset+memcpy（host 侧非 launch，为融合 kernel 备 pad）；旧 cube/reduce/part_sum kernel 定义保留为死代码 | 恰 1/次 | 全文 `<<<` 仅 1 处；禁词 0；lint 0 | **15 全 RE：Expected 75, got 20**（本轮用户报告） |

- **期望占位能过 profiling（每 run_kernel 恰 1 launch）但实际 got=20** —— 说明"每 iteration 恰 1 kernel"检查的统计口径与直观不同（§8），必须先探针定位。

### 4.1 git 状态（如实，需下轮核实/整理）

- 已 push origin/master：`4057adf`（M0+M1 py 修复）、`8d1888c`（host __gm__→plain 修复）、`6ac7b9f`（run_kernel 形参改 GM_ADDR 对齐官方模板）。
- 工作区领先于 master 的**未提交**改动：① 禁打印重构（19 处 fprintf 移除 + GenBlockTiling typeTag 清理，git 卡死搁置）→ ② 占位版重构。即本地 kernel.asc = 占位版，尚未 commit/push。
- git 卡死根因 = dubious ownership：`Projects/.git` owner=BUILTIN/Administrators，当前普通账户。解法（不写 config）：每条命令加 `-c safe.directory=P:/Dev/CANN_Learning_Workspace/Projects`（仓库内可用 `-C Projects`；此前亦用过 `git -c safe.directory='*'`）。用户在时间紧迫时选择跳过 push，不纠结。

---

## 5. 已沉淀的硬经验（写码/改码必遵守）

1. **Ascend C host 侧指针规则**：host 侧函数签名与中间量一律 **plain `uint8_t*`**；只有 __global__ kernel 形参写 `__gm__ uint8_t*`。`reinterpret_cast<__gm__*→void*>` 编译器禁止（aclrtMemsetAsync/MemcpyAsync 要 void*）。plain→__gm__ launch 传参可隐式。
2. **单 launch 铁律**：判题要求一次 run_kernel 的 iteration 对应**恰好 1 个 kernel launch**；所有 batch×行块计算必须并入单 kernel（多核 + 核内循环）或用官方 batch 高阶 API。
3. **Matmul 高阶 API 核分派按 M/N**（CalcOffset），不原生按 batch 放大；batch 能力靠 `SetBatchNum(batchA,batchB)`（"Reset batch number for Batch Matmul without changing tiling"）与 `IterateBatch(gm/ubCmatrix,...)`（"Calculate multiple C matrices"，matrixStrideA/B）。
4. **官方融合范式**（单 launch 内 cube+vector）依据：05.04 matmul_abs / 05.05 matmul_sinh —— `__aicore__` kernel + `while(matmulObj.Iterate<true>())` + `GetTensorC<true>(localC)` 取 VECIN LocalTensor 做向量归约。
5. **头库已核 API 结论**：
   - Level2 归约 4 参元素计数：`ReduceMax(dst,src,tmp,count,calIndex=0)`、`ReduceSum(dst,src,tmp,count)`（kernel_operator_vec_reduce_intf.h；fp32 每 repeat 64 lane → 分块 ≤64）；"全负行返回最大负值"，**勿预填 0 作初值**（行初值 -FLT_MAX）。
   - 4B 精确写 GM：`DataCopyPad + DataCopyExtParams{1,4,0,0,0}`（已 PASS 形态）；`DataCopy(dstLocal,srcGlobal,count)` GM→UB 需 32B 对齐（Npad%64==0 使每段 256B）。
   - cube kernel 序列 = 官方 03.03/P3：`REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling)`→`SetOrgShape`→`SetTensorA/B`→`SetBias`→`IterateAll`→`End`；`ASCENDC_CUBE_ONLY` 在 `kernel_operator.h` 之后、`lib/matmul_intf.h` 之前（防 507015）。
   - **全部 launch 后 aclrtSynchronizeStream 才 aclrtFree**（P2b/P3 教训）。
6. **禁词零容忍**：提交前全文扫描 `fprintf|printf|fputs|fopen|freopen|stdout|stderr|puts|cout|cerr|write|syscall|dlopen|dlsym|asm|fwrite`（词边界、大小写敏感）必须 0 命中；**新注释/字符串也别引入被禁 token（中文注释优先）**。
7. **守卫只能静默 return**（无打印通道）。守卫集：dtype 1/2 且相等、y fp32、rank 3/1、transpose 均 false、B∈[1,64]、M/N∈[1,8192]、K∈[32,8192] 且 %8==0、预算挡板、tiling Ka/Kb==Kpad 且 usedCoreNum==1。
8. 云端自测链：`cd projects/batchmatmulmaxsum_problem_1746_template && bash run.sh`（[1/4] 会自动找 `/home/developer/Ascend/cann-9.0.0`；CWD 默认 `/mnt/workspace` 需 cd）。python 依赖 numpy+ml_dtypes 若缺需 pip 安装（按用户规则，装依赖交给用户）。
9. 本地只允许静态检查：`py_compile`（penv python）、read_lints、正则扫描。禁止本地编译/运行 C++（无编译器、无 CANN）。
10. 遍历大目录禁忌：P:\Dev 达 2.2TB，**禁止递归/顶层 list p:\Dev**，只允许精确子目录。

---

## 6. 本轮实测记录（用户 2026-09-06 深夜报告，如实转录）

- 提交物：本地占位版 kernel.asc 全文粘贴判题页（§4 占位版）。
- 结果：**15 个测试点依旧全部 Runtime Error**。
- 错误原文（全点一致）：
  `Profiling rule violated: each iteration must launch exactly 1 kernel. Expected 75 launches, got 20.`
- 与首跑（M1 版 got 2065）同错误结构，Expected 恒 75，仅 Got 数字变化（2065 → 20）。
- 用户指示：**先如实记录，暂不启动问题追踪**（本轮仅做记忆/交接）。

---

## 7. 疑点清单（未追踪、未定论，禁止当结论用）

- **E75**：`Expected 75` 的含义未证。候选：(a) 15 测试点 × 每点 5 iteration 的 run_kernel 总调用次数；(b) 平台按 tiling/题意预计算的"应有 launch 总数"。占位版每次调用恰 launch 1 次，若 75 次调用全执行，got 应为 75 —— 与 got=20 矛盾。
- **G20**：got=20 的可能来源（并列，未验证）：(a) 75 次调用中只有 20 次到达 launch（其余命中 guard 静默 return：如 transpose=true 或 dtype/shape 域外——平台用例带 transpose 比例未画像）；(b) 平台对多核 grid launch 的计数/可见性不同；(c) 平台在遇到前次错误/超时后停止统计；(d) profiling 只统计成功 enqueue，异步失败不计。
- **检查序**：profiling 检查先于数值结果 → 至今未见过 precision_ratio/用例表/时间等反馈字段。

---

## 8. 下轮候选行动（记录备用，本轮不执行；启动前需用户确认）

1. **最小 launch 计数探针实验**（定位 E75/G20，一次一个变量）：
   - 探针①：run_kernel 去掉全部 guard 分支，无条件 launch 1 个空 kernel `<<<1>>>` → 看 Got 是否=iteration 数；若 Got≠75 则 Expected 75 与调用次数无直接关系。
   - 探针②：run_kernel 完全空（0 launch）→ 若报 Got 0 / Expected 75，证明"每次调用期望恰 1"。
   - 探针③：同样 1 个空 kernel 但 `<<<availableCoreNum>>>` 多核 → 对比 Got，排除 grid 大小对计数影响。
2. 摸清统计口径后：若"每 iteration 恰 1 kernel 且必须真算对"，则走 **单 launch 融合 kernel**（官方 batch/融合范式 §5.3/5.4 候选：B 方案 SetBatchNum+IterateBatch 一次算完再归约；或 A 方案 VECIN+Iterate 每核负责若干完整 batch）替换占位；host pad 基础已在占位版中备好。
3. 同时准备 transpose 四布局适配层（首跑用例画像出来后精准补）。

---

## 9. 提交前静态全绿清单（每轮必查）

- [ ] 全文 `<<<`（kernel launch）数量符合本轮目标（占位=1）；
- [ ] 禁词扫描 0 命中（正则见 §5.6）；
- [ ] read_lints 0 诊断（kernel.asc 路径）；
- [ ] 无残留对被删 host 变量的引用；
- [ ] 与上一提交的 diff 有意（新注释避免被禁 token）；
- [ ] 把本地 kernel.asc **全文覆盖**粘贴到判题页（非增量）；
- [ ] 回报结果原文（尤其 Got/Expected 数字、是否出现用例表字段）。

---

## 10. 交接结论（下一位 Agent 最先读这节）

- 当前唯一障碍 = profiling 硬规则的统计口径（占位 got=20 而非 75）。
- 不必再猜数字含义：先跑 §8 探针实验（需用户提交 3 次占位变体到判题页，逐次回报 Got/Expected）。
- 长线目标不变：单 launch 融合 kernel 真算对 + transpose 覆盖 + 15/15 得分。
- 历史细节回查顺序：本文件 → `2026-09-06.md` → plans（judge_roadmap / PLAN / digest）→ 工程代码。

---

## 11. 探针决定与推进（2026-09-07 更新）

- **E75/G20 新推断（收敛 §7）**：75 = 15 用例 × 5 iteration；占位 got=20 = 恰好 4 个 false/false 用例 × 5 → 守卫静默 return 制造「0-launch iteration」为主因（M1 got=2065 ≈ 4×5×103 侧证）。推论：**转置/全域都必须真实处理且每 run_kernel 恰 1 launch 才能过 profiling**——transpose 支持由 M3 域扩展升级为 profiling 关键路径。
- **本轮行动**：用户选定「探针先行」。probe-01 = 无守卫空 kernel（每 run_kernel 恰 1 launch，写 y=0，无 malloc/拷贝，无中途失败路径）。判读：Got=75 → 假设坐实；Got 仍=20 → 55 次 iteration 未达 launch 或平台侧采样问题，转 §8 差异化探针。
- **状态**：本地探针 kernel.asc 已备好 + 静态自检通过；git CLI 本机不可用，commit/push 待用户；完整过程见 `2026-09-07.md`。
