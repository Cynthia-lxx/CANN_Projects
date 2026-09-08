# PLAN：BatchMatmulMaxSum 赛题独立攻关计划（本工作区）

> 文档身份：本工作区（CANN_Learning_Workspace）独立攻关 2026 CANN 挑战赛·上合赛区 BatchMatmulMaxSum 的总体计划与技术研读沉淀。
> 目标：以 `Projects/batchmatmulmaxsum_problem_1746_template` 为起点，走「研读→骨架→扩展→性能→真机回归」路线，做出可通过线上 judge、且性能优于"拆分实现基线"的 Ascend C 算子。
> 创建：2026-09-06。赛题窗口：2026/09/05 00:00 — 10/17 18:00；内部 DDL 2026-10-07。
> 补充：隔壁 CANN_Learning_Refs（CANN_OpHelper 工具项目）记忆中有同题 P0–P3 探索记录，仅作**只读参考**（见 §7.4），本计划独立推导、不复刻其工程。

---

## 1. 赛题精读（来源：`memory/stage_goal_batchmatmul_maxsum.md`）

### 1.1 计算语义
逻辑矩阵恒为：x1_logical ∈ R^{B×M×K}，x2_logical ∈ R^{B×K×N}（ND 连续）。

阶段一 BatchMatMul：A[b,m,n] = Σ_k x1[b,m,k]·x2[b,k,n]（K 累加）
阶段二 MaxSim：R[b,m] = max_n A[b,m,n]（沿 N 取最大）
阶段三 Sum：y[b] = Σ_m R[b,m]（沿 M 求和）；y shape=(B,)，dtype=fp32

计算顺序**固定**：先 max(N) 再 sum(M)，不可交换。第 b 组 x1 只与第 b 组 x2 配对。

### 1.2 精度判据
| 输出 dtype | 判据（相对+绝对误差双条件） |
|---|---|
| y=fp32 | < 1e-4 且 < 1e-4 |
| （x1/x2 为 fp16/bf16 输入） | K 点积/Max/Sum 用 **fp32 累加或等效精度** |

- golden 语义：FP64 计算后转 FP32。
- 输入含负值、不含 NaN/±Inf；**MaxSim 初值不得为 0**（须负无穷或行首元素，ReduceMax 逐元素语义天然满足，禁止用 Maxs(0) 预填）。
- 输出全部有限（无 NaN/±Inf）。

### 1.3 输入/输出/属性总览
| 项 | 逻辑 shape | storage shape | dtype |
|---|---|---|---|
| x1 | (B,M,K) | transposeX1=false → (B,M,K)；true → (B,K,M) | fp16 / bf16 |
| x2 | (B,K,N) | transposeX2=false → (B,K,N)；true → (B,N,K) | fp16 / bf16（与 x1 同型） |
| y | (B,) | — | fp32 |

属性仅声明 storage shape，**不表示算子要执行转置计算**，四种 storage 组合都要支持。

### 1.4 维度与规模约束
- 1≤B≤64；1≤M≤8192；1≤N≤8192；32≤K≤8192 且 **K 为 8 的倍数**。
- B×M×K ≤ 2^28；B×N×K ≤ 2^28。
- M、N **建议 16 的整数倍**；非对齐（尾块）场景**必须正确**处理。
- 无 padding/有效长度/mask；输入连续、非空；算子不得修改输入。

### 1.5 四种存储布局（storage, 逻辑同上）
| 组合 | x1 storage | x2 storage |
|---|---|---|
| T,T=false,false | (B,M,K) | (B,K,N) |
| T,T=false,true | (B,M,K) | (B,N,K) |
| T,T=true,false | (B,K,M) | (B,K,N) |
| T,T=true,true | (B,K,M) | (B,N,K) |

### 1.6 官方三示例
- 例1 (B,M,N)=(1,2,2)/(1,2,3) K=2 → y=[2.0]
- 例2 同一逻辑内容、双 transpose=true 物理布局 → y=[2.0]
- 例3 全负相似度 → y=[-1.0]（MaxSim 不可初始为 0）

### 1.7 性能评分口径（规则 4/5）
- 基线 = BatchMatMul + ReduceMax + ReduceSum **拆分实现**总耗时；融合实现按加速比评分。
- 需合理利用 Cube+Vector、双缓冲/流水、沿 B/M 多核分配；msprof 测 Task Duration/Cube/Vector/MTE 利用率。

---

## 2. 题目交付工程模板现状（`Projects/batchmatmulmaxsum_problem_1746_template`）

Ascend C **Direct Invocation** 形态（非 msopgen/aclnn 注册工程），单可执行：

```
batchmatmulmaxsum_problem_1746_template/
├─ CMakeLists.txt         # DI 单文件工程；SOC_ARCH=dav-2201(910B) 默认
├─ data_utils.h           # ReadFile/WriteFile（大小需精确匹配，I/O 封装）
├─ main.asc               # 本地冒烟 main：硬编码 case0(B=1,M=1,N=1,K=32) fp16→fp32 y=(1)
├─ kernel.asc             # 被 main.asc #include；实现 run_kernel()；禁止再写 main
├─ run.sh                 # cmake+make→gen_data→运行→verify_result
└─ scripts/
   ├─ gen_data.py         # 生成 input/*.bin + output/golden_y.bin
   ├─ BatchMatmulMaxSum.py# golden 参考实现（torch 组合）
   └─ verify_result.py    # 逐输出 np.isclose 判定（case0: y fp32 rtol/atol/tol=1e-4）
```

### 2.1 judge 对外接口契约（kernel.asc 必须提供）
```cpp
struct TensorInfo      { const int64_t* shape; int64_t numDims; int32_t dtype; };
struct TensorGroupInfo { const TensorInfo* tensors; int64_t numTensors; };
// dtype: 0=fp32 1=fp16 2=bf16 3=int8 4=int16 5=int32 ...
extern "C" void run_kernel(GM_ADDR x1, const TensorGroupInfo& info_x1,
    GM_ADDR x2, const TensorGroupInfo& info_x2,
    GM_ADDR y, const TensorGroupInfo& info_y,
    int64_t availableCoreNum, aclrtStream stream,
    bool transposeX1, bool transposeX2);
```
- kernel.asc 不得含 main/#pragma once/include guard。
- 核函数需在本文件内 `<<<blockNum,nullptr,stream>>>` 启动；blockNum 自定（≤availableCoreNum，通常再 cap 平台可用核数）。
- 本地跑分核心信息：可从 `info_x1.tensors[0]` 取 B/M/K，`info_x2` 取 N（B、K 校验与 x1 一致）。

### 2.2 本地验证链需扩展（M0）
- gen_data.py：目前仅 case0 超小 shape → 需多 case 覆盖：16 对齐大块 / 非对齐尾块(M 或 N %16≠0) / N%8≠0 / B 中大批 / K 边界 / 全负值构造 / 四 storage 组合。
- main.asc：目前仅固定 case0 64B 读取 → 需要能按 case 参数分配并喂入 run_kernel（读取 shape 列表）。
- verify_result.py：case_output_specs 需随 case 扩充；判据统一 fp16/bf16→1e-3、fp32→1e-4（与赛题一致；模板现 1e-4 仅适合 fp32 冒烟）。

---

## 3. 平台与环境结论（本地「写码前必查」结论）
- 编译/运行全部在云端 CANN Lab；本地禁止安装 CANN/编译 C++。本机仅静态检查。
- 目标 SoC：910B 系 = dav-2201；CANN 9.0.0 头库 = 本地 `cann_ascendc_headers_9.0.0`（权威，只读）。
- 头树关键路径（asc/include 下）：`adv_api/matmul/matmul_intf.h`、`adv_api/matmul/matmul.h`、`adv_api/matmul/bmm_tiling.h`、`adv_api/matmul/matmul_tiling_base.h`、`adv_api/matmul/matmul_tilingdata.h`、`adv_api/tiling/tiling_api.h`、`basic_api/kernel_operator_vec_reduce_intf.h`、`adv_api/reduce/reduce.h`、`kernel_operator.h`。
- 2201 下 `matmul_intf.h` include **kernel_kfc.h**（KFC 客户端），`REGIST_MATMUL_OBJ` = `REGIST_CUBE_OBJ`。

---

## 4. 资料地图（锚点，写作代码时的回查地址）
| 用途 | 路径/锚点 |
|---|---|
| 直启 cube Matmul kernel（范本） | `Reference_And_Documentation/tutorials/ascendc_operator_development_light/03_simple_operator_practice/answer/03.03/matmul_custom.asc`（__global__ __cube__ + REGIST_MATMUL_OBJ + SetOrgShape + SetTensorA/B + IterateAll + End；Host 侧完整 acl 流程） |
| Host tiling 生成链（范本） | 同上 `03.03/matmul_custom_tiling.h`：MultiCoreMatmulTiling：SetDim→SetAType/SetBType/SetCType/SetBiasType(GM,ND,dtype)→SetShape→SetOrgShape→SetFixSplit(128,128,-1)→SetBias(true)→SetBufferSpace(-1,-1,-1)→GetTiling |
| CV 融合取 C 到 UB | `light/.../answer/03.05/matmul_abs.asc`（CType VECIN + Iterate + GetTensorC<true>(ubLocal,false,true) 后 vector 加工；05.04 主课 matmul_abs 同构） |
| Host 调用/workspace 全流程（直启多案例） | `light/.../answer/03.06/matmul_custom.asc`：GetLibApiWorkSpaceSize → aclrtMalloc workspace → launch → aclrtSynchronizeStream → 回拷；03.03/03.05 同构 |
| 完整课程（主课 04/05 章） | `Reference_And_Documentation/tutorials/ascendc_operator_development/04_matmul_basic/` 与 `05_fused_operator_development/`（04.03 matmul_custom 工程、05.04 matmul_abs 融合工程、cube tiling 讲解） |
| 本地知识沉淀 | `.codebuddy/memory/ascendc_01~07_*.md`、`ascendc_knowledge_index.md`（主题①范式、②模板、③tiling、④host-device、⑤API、⑥调试、⑦实践） |
| 头库权威签名 | 见 §5（本地 cann_ascendc_headers_9.0.0，路径标于表内） |
| 调试/避坑 | 主题⑥、cannbot-skills（507015 报错探矿笔记）、隔壁记忆（只读，§7.4） |

---

## 5. 头库核验结论（2026-09-06 已与 cann_ascendc_headers_9.0.0 核对）
> 依据文件绝对路径+行，全部本工作区头库副本实测。

### 5.1 Vector 两级归约 API（`basic_api/kernel_operator_vec_reduce_intf.h`）
- Level2 元素计数重载（无 repeat/mask/stride）：
  - `ReduceMax(dst, src, sharedTmpBuffer, int32_t count, bool calIndex=0)` —— L289
  - `ReduceSum(dst, src, sharedTmpBuffer, int32_t count)` —— L301
  - `ReduceMin` 同形 —— L277；dst 与 src 同 dtype（fp32 用 fp32 buffer 归约）。
- 低层 repeat/mask 版（同文件 L220-265）：需 sharedTmpBuffer + mask/repeat/srcRepStride；**只在我们需要显式控制行切分时才用**。
- BlockReduceMax/PairReduceMax/WholeReduceMax 等另有（L46-204），level2 足够。
- 语义提醒：Level2 ReduceMax 从 src 逐元素取最大（行首即初值）→ 天然满足「全负行返回最大负值」，**禁止预填 0**。
- 平台实现在 `asc/impl/adv_api/detail/reduce/reduce_max|sum/reduce_max|sum_v220_impl.h`（v220 系，910B）。

### 5.2 Matmul 高阶 API（Kernel 侧，2201 → KFC）
- 类：`Matmul<MatmulType<pos,fmt,dtype> aT, bT, cT, biasT> mm;`
- 序列：`REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling)` → `mm.SetOrgShape(tiling.M,tiling.N,tiling.Ka,tiling.Kb)` → `SetTensorA(aGlobal)` → `SetTensorB(bGlobal)` → `SetBias(biasGlobal)`(可选) → `IterateAll(cGlobal)`（整块直写 GM）或 `Iterate()`+`GetTensorC`（取片/融合）→ `mm.End()`。
- `GetTensorC<true>(LocalTensor co2Local, enAtomic=0, enSequentialWrite=false)`：取 C 到 UB 供 vector 继续（CV 融合同步版）。
- host 获取 lib workspace：`platform_ascendc::PlatformAscendCManager::GetInstance()->GetLibApiWorkSpaceSize()`。
- 头库：`adv_api/matmul/matmul.h`（声明 stub L250-320：IterateAll GM/Local、GetTensorC GM/Local）、`adv_api/matmul/matmul_intf.h`（2201 分支 kernel_kfc.h）。

### 5.3 Host Tiling（`adv_api/matmul/bmm_tiling.h` + `matmul_tiling_base.h`）
- `matmul_tiling::MultiCoreMatmulTiling`：SetDim / SetShape(m,n,k) / SetOrgShape(3 或 4 参含 Ka,Kb) / SetSingleShape / GetTiling(optiling::TCubeTiling 或 AscendC::tiling::TCubeTiling) / SetAlignSplit / EnableMultiCoreSplitK(flag)。
- 基类 `MatmulApiTilingBase`（matmul_tiling_base.h）追加：**SetAType/SetBType(pos,fmt,dtype,isTrans=false)**（isTrans 存在！潜在 transpose 支持点）、SetCType、SetBiasType、SetALayout/SetBLayout/SetCLayout(b,s,n,g,d)、**SetBatchInfoForNormal(batchA,batchB,m,n,k)/SetBatchNum(batch)**、SetBias(bool)、**SetFixSplit(baseM,baseN,baseK)**、SetBufferSpace(l1,l0c,ub,bt)、SetSplitRange、SetDoubleBuffer 等。
- **CANN 9.0 自带 `BatchMatmulTiling`**（bmm_tiling.h L192）：GetCoreNum(dim,mDim,nDim,batchCoreM,batchCoreN)、GetTiling（optiling/AscendC 双版）——真批量 tiling 备选方案。
- 临时缓冲查询：`MultiCoreMatmulGetTmpBufSize(V2)`/`BatchMatmulGetTmpBufSize(V2)` 可查 ub/l1/l0c 预算。
- 用法照抄官方范本（03.03 matmul_custom_tiling.h 整段，见 §4）。

### 5.4 其余（写码前按需精读）
- Cast/RoundMode 枚举实际值定义点待精读（`basic_api/reg_compute/kernel_reg_compute_vec_vconv_intf.h` 一带出现 CAST_RINT/CAST_RZ 字样）；默认按官方示例 `Cast(dst, src, RoundMode::CAST_RINT, count)`（fp16→fp32 等）。
- DataCopy/DataCopyPad/DataCopyParams 见主题⑤§11.3 与 05.04 示例（写 GM 带 stride 子矩阵时用）。
- cube 最小 base 块/对齐/真 shape 边界（910B dav-2201）：**写码前必须回主课 04.03 5.x cube tiling 讲解与头库 `matmul_tilingdata`/`kernel_kfc` 确认**；P0 阶段不臆测，先用"全部对齐大 shape"与官方范本一致的保守固定块。

---

## 6. 总体技术路线（里程碑 M0–M4）

### 核心矛盾
1. 输出只有 B 个标量，但中间 A 是 B×M×N 大矩阵 → 必须中间驻留/逐块归约，避免全量 A 写 GM 再二次读（性能评分针对拆分方案，融合就要省中间搬运）。
2. M/N ∈ [1,8192] 任意、建议 16 倍；非对齐尾块必须正确 → cube 的 base 块切割之外要处理残余行列。
3. B 小(1..64) + B×K×N ≤2^28：多核要沿 B 与 M 两个方向分。

### M0：验证链先行（纯本地脚本，无需云）
- 扩 gen_data.py：多 case 表 + 布局组合参数 + 全负样例；golden 保持 BatchMatmulMaxSum.py（fp64→fp32）。
- 扩 main.asc：读 case 配置（shape/transpose/文件），通用化分配；保留 data_utils 精确尺寸。
- 扩 verify_result.py 判据表。
- run.sh 支持多 case 循环。**M0 完成标志：case0 冒烟仍 PASS（空实现 + gen/verify 通）。**

### M1：正确性骨架（transpose=false 域，固定 shape 起）
- kernel.asc 提供 `run_kernel`；内部按 shape/规模走两条 Kernel 路径之一：
  - **Vector 全量兜底**：面向 cube 不可行的退化 shape（例：M=1/N=1/K 小等，B×M×N 很小）；直接 CopyIn→UB 展开：每行 k 点积用 fp32 累加（乘加或 D=K 一次 Add+ReduceSum 不好做就逐元素乘+累加 + 逐行 max 再跨行 sum），保证全 shape 正确性底线。
  - **Cube+Vector 主链（P0 版）**：逐 batch 或逐 (b,m-block)：`Matmul` 一块 C(baseM×N 或真 M×N 一次性) → fp32 UB → 行向 max(N)（ReduceMax count 版每行一次）→ 跨行 sum 累加 → y[b]。tiling 用官方 GenMatmulTiling 链（SetFixSplit 保守 128×128 或按需）。多核先 1 核 per (batch,m-block) 甚至单核跑通。
- workspace：`GetLibApiWorkSpaceSize()` 系统量 + (用户量 0)。kernel 尾 `<<<...>>>` 后由 run_kernel 内保证 **先 `aclrtSynchronizeStream` 再 free 任何 device 资源**。
- **M1 里程碑：云端例1/例2(basic fp16 对齐域) PASS、本地对齐 shape 多 case PASS。**
- 依赖预研（写码第一件事，回文档）：04.03 5.3 cube tiling 字段语义（baseM/baseN/Ka/Kb/depthA1B1/usedCoreNum/singleCore* 含义）；910B 下真 shape 支持（官方 light 04 章 example 建议 SetOrgShape 真 shape + 单核 base）；`ASCENDC_CUBE_ONLY` include 次序。

### M2：全 shape 域 + 尾块正确（仍在 transpose=false 或四布局内的数学对齐）
- cube 块外残余 M/N 行（base 不能整除、非 16 对齐、N%8≠0 场景）显式处理：把 C 区域视为 [M_pad × N_pad] 补齐到 cube 对齐尺寸读取（**P0 阶段保持"padding 不参与数学"正确**，用 DataCopyPad 或矩形搬运补 0，最后归约截断到真 M/N），或用更小 base 块/退化路径 Vector 兜底。具体以 04.03 tiling 与实测为准。
- K 不变量 8 倍但可能需要 Ka/Kb（K=单侧大时 SetSplitK 关闭，默认 K 整段入 depth）。
- 数学负值、全负行天然通过 ReduceMax。

### M3：四种 storage 布局组合
- transposeX1/X2 只是读布局声明 → 数学等价于对逻辑张量按对应索引访问。
- 实现候选（写码前各做一次头库/官方确认）：
  a) 若 tiling `SetAType/SetBType(isTrans=true)` + `SetALayout/SetBLayout` 真支持转置读布局 → 优先；
  b) 否则 kernel 读 GM 时把转置输入当「列主序逻辑」喂 cube 需显式搬运 → 采用「读入 workspace/UB 后按逻辑布局重排」或「整体转置预置 workspace」；
  c) 数学恒等改写：C^T=X2^T·X1^T → 当双转置组合时天然互为 C^T，配合归约方向改换。
- **M3 里程碑：例2（双 transpose）与四组合对齐 shape PASS。**

### M4：多核并行 + 双缓冲 + 性能
- 任务划分：batch 维 B≤64 且各 batch 独立 → 优先 batch×核切分；单 batch 内沿 M 分 m-block。host 生成 TCubeTiling usedCoreNum 对 (batch, m-block) 计数分配；vector 归约与其同核串行（先保证对）。
- Cube/Vector 并行：单核内一次 cube 一块 + 紧接归约（queue 流水）；不同核并行跑不同 (b,m)。
- workspace C 大矩阵方案（如需）：迭代 C 块直接 vector 归约，不落 GM（对齐题目评分=省拆分方案的中间搬运）。
- msprof 采样：Task Duration/Cube/Vector/MTE 利用率分析瓶颈（云端跑）。

### M5：全矩阵真机回归与提交
- case 矩阵覆盖表逐项过：B∈{1,2,64}、M/N∈{1,16,128,8192,8191/非16对齐,非8倍数}、K∈{32, 8 倍数随机 8192}、fp16/bf16 × 四布局 × 多 seed（每次提交前）。
- git 提交规范：`Projects/` 每次改动 commit+push（github.com/Cynthia-lxx/CANN_Projects.git）。

---

## 7. 关键设计决策与依据（随进展就地更新，标注锚点与日期）

- D1 用官方 GenMatmulTiling 链生成 TCubeTiling（03.03 matmul_custom_tiling.h），不手写 tiling 公式。锚点：04 章官方直启范本。日期 2026-09-06。
- D2 MaxSim 用 Level2 `ReduceMax(fp32, count=N)`，不预填初值。锚点：头库 kernel_operator_vec_reduce_intf.h L289。2026-09-06。
- D3 主链 K 点积 = cube fp32 累加（C 类型 fp32），两级归约全在 fp32 UB 完成，最终写 y[b] fp32。锚点：赛题 §3.4/§5。
- D4 中间 A 以「每 cube 块即归约」方式不落 GM（性能项）。锚点：赛题评分口径。待 M1 验证 cube 每块取 C 到 UB 可行性后正式定。
- D5 M0 先扩本地验证链再写 kernel，避免盲改。2026-09-06。

### 7.4 隔壁项目只读参考（CANN_Learning_Refs 记忆，2026-09-05/06 日志摘记）
> 仅避坑参考，不复刻工程。重要事实：
- 2201 上直接含 `lib/matmul_intf.h` 必须 `ASCENDC_CUBE_ONLY` 先行定义（否则 507015 编译错）；include 顺序：kernel_operator.h → 定义宏 → lib/matmul_intf.h。
- Cube matmul 真 shape 可行：SetShape/SetOrgShape 用真 M/N，配 SetFixSplit；锚定 64×64 真块在 910B 全链 PASS（cube 真 shape 大小块皆可）。
- 融合用 `__mix__` 曾三轮未愈（507015）——**不要走 __mix__**；改双指令核内同核串行（cube 出 C→GetTensorC<true>→vector 归约）已通。
- `Iterate<true>`+GetTensorC 的 while 循环是官方融合流水形态；host launch 后必须显式 `aclrtSynchronizeStream` 再 free，否则 c 值错。
- workspace 需 sys workspace（Matmul 内部用）；用户 workspace 0 即可。
- 归约方向经验：先每行 N-max（v1 fixed tiling(128,128)）实测可靠；v0(128,128) 变体曾 FAILED，避开。

---

## 8. 风险与开放问题（写码前必须回查确认）
1. cube 最小 base / 对齐要求（910B dav-2201）与 baseM/baseN=128 何时可退化为更小/真 shape（64×64 可锚定，需查 matmul_tiling_base/matmul_tilingdata 与官方 04.03 5.3）。高优先。
2. transpose 读布局的官方支持点（tiling SetAType isTrans？kernel 端 MatmulType 无 trans → 大概率需显式搬运/恒等改写）。高优先（M3 前定）。
3. N/M 非对齐尾块的 cube 读法（DataCopyPad vs 矩形搬运补 0 对 cube A/B GM 源是否可用——cube 自动 load 对 GM 布局对齐的依赖）。M2 前定。
4. BatchMatmulTiling 真批量方案 vs 自循环 per-batch 方案的开销与实现复杂度取舍。M4 前定。
5. Run 时 blockNum 上限与 GetBlockNum（SPMD）在直启下的实际可用核数核对（availableCoreNum vs PlatformAscendC.GetCoreNum）。
6. local workspace(GetLibApiWorkSpaceSize) 与用户暂存（若需 C 大矩阵）在 direct-invocation aclrtMalloc 一次性分配。

---

## 9. 下一步行动（按序执行）
1. [ ] M0：扩本地 gen_data/main/verify 多 case 链（纯文本+脚本，无需云端）。
2. [ ] 写码前预研：04.03 cube tiling 字段 + 头库真 shape 边界 + Matmul include 顺序铁律；产出「API 结论」增补进主题⑤。
3. [ ] M1 骨架写码：kernel.asc 双路径 + run_kernel 框架 + GenMatmulTiling；云端编译 case0 冒烟 + 小对齐 case。
4. [ ] M1 对齐 shape PASS → commit/push（Projects git）。
5. [ ] 按 M2→M5 推进，每阶段云验 + commit；真机全矩阵回归表存档 `.codebuddy/`。

---
*变更日志：2026-09-06 初稿（研读范围：赛题原文、题目模板全部代码、官方 light 03.03/03.05/03.06 直启源码、官方主课 04/05 章结构、本工作区主题①②③⑤⑦、头库 reduce/matmul/tiling 签名）。*
