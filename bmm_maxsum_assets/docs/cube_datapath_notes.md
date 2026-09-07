# Cube 数据流与 AscendC MatMul/融合范式笔记

> 目的：P0 研读产物——把官方教程（`04_matmul_basic` / `05_fused_operator_development`）中 MatMul 高阶 API 与融合算子的执行骨架、数据流、host/kernel 分工沉淀为可检索笔记，作为 P2 垂直切片与 P4 性能优化的硬件侧依据。
> 来源：官方本地教程 `04_matmul_basic\answer\04.03_answer`、`04.04_answer`（`matmul_custom.cpp` host+kernel）与 `05_fused_operator_development\answer\05.04_answer`（`matmul_abs.cpp` host+kernel），三镜像内容一致，本文档以 `p:\Dev\CANN_Learning_Refs\Documentation_for_Developers\ascendc_operator_development\` 为根引用。
> 状态：**理解记录**。其中标【推断】【待验证】的条目需在 P2 动手前用官方 ipynb 讲解（04.03/05.04 对应 72KB 教学 notebook）与真机验证校正；其余为源码可复述事实。

---

## 1. 昇腾 AI Core 存储/执行层级（概念层，源自教程描述）

```text
GM  (Global Memory, 片外大容量)
 │  MTE 搬运（DataCopy / 高阶库自动搬运）
 ▼
L1 ──► L0A ◄─ 读取 A 分块（Cube 的左边矩阵）
 └──► L0B ◄─ 读取 B 分块（Cube 的右边矩阵）
          │  Cube 单元（矩阵乘核心，处理 baseM×baseN 对齐块）
          ▼
        L0C（乘累加结果缓存区）
          │
          ▼
        UB（Unified Buffer）◄── Vector 单元（逐元素/归约等指令运行于此）
          │
          ▼  MTE DataCopy 写回 GM
```

- **执行单元分工**：Cube 只做矩阵乘（int8/fp16 主算力），Vector 做元素级/归约类运算；二者是异构的独立流水，可重叠执行（协同流水即本题 P4 目标）。
- **内存层级**：GM 为片外；L1/L0A/L0B/L0C 为 Cube 侧缓冲（高阶库内部分配/管理，用户一般不直接触碰 L0）；UB 是用户可见片上缓冲，Vector 指令与部分搬运的落脚点。
- **分离架构注记（源码注释事实）**：Ascend910B 类为 AIC:AIV = 1:2（1 个 Cube 核配 2 个 Vector 核）；MatMul host 侧 `SetDim` 设置的是矢量核数，`context->SetBlockDim` 需设为 Cube 核数。310P 等不同，代码用 `SocVersion` 分支区分（TilingKey 2/1）。

## 2. MatMul 高阶 API 骨架（04.03/04.04 `matmul_custom.cpp` 实录）

### 2.1 host 侧 TilingFunc 关键步骤（`op_host\matmul_custom.cpp`）

1. `MultiCoreMatmulTiling cubeTiling(ascendcPlatform)`：多核切分对象。
2. `cubeTiling.SetDim(n)`：参与并行核数（矢量核视角）。
3. 类型声明——`SetAType/SetBType/SetCType/SetBiasType(TPosition, CubeFormat, DataType)`：
   - A/B 常为 `TPosition::GM, CubeFormat::ND, DT_FLOAT16`；C 可为 `GM`（直接写回全局，04.03）或 **`TPosition::VECIN, ND`**（结果留在 UB 供 Vector 后续处理，05.04 融合关键）。
   - dtype 组合：fp16→fp32（04.03）、int8→int32（04.04）。
4. `SetShape(M,N,K)` 与 `SetOrgShape(M,N,K)`：实际形状与原始形状（后者服务尾块，教程示例两者相同）。
5. `SetFixSplit(baseM, baseN, -1)`：固定基础分块 128×128，K 由库自动选。
6. `SetBias(bool)` / `SetBufferSpace(-1,-1,-1)`（-1 表示库自动分配缓冲）。
7. `GetTiling(tiling.cubeTilingData)` 产出切分参数写入自定 tiling 结构；再 `GetCoreMemSize(UB)` 存 `localMemSize` 供核函数工作空间。
8. 平台差异化：310P → `SetBlockDim(2)+SetTilingKey(2)`；否则 → `SetBlockDim(1)+SetTilingKey(1)`（910B 每 Cube 配 2 Vector）。
9. `tiling.SaveToBuffer(...)` 序列化；`GetWorkspaceSizes` 申请系统工作空间（`GetLibApiWorkSpaceSize`，Matmul 库内部临时内存）。

### 2.2 kernel 侧流程（`op_kernel\matmul_custom.cpp`）

- 类内嵌 `Matmul<MatmulType<TPosition, CubeFormat, type>, ...×4> matmulObj;`——模板参数描述 A/B/C/Bias 位置+格式+dtype。
- `Init`：`GlobalTensor.SetGlobalBuffer(addr, 总元素数)` 绑定 GM → `CalcOffset(GetBlockIdx(), ...)` 算出本核负责的 M 方向行块与 N 方向列块 → `aGlobal = aGlobal[offsetA]` 等**仅记录子区域地址、不搬数据**。
- `CalcOffset` 公式（单核按 M×N 二维网格划分）：
  - `mSingleBlocks = Ceiling(M, singleCoreM)`（向上取整，天然覆盖尾块）；
  - M 索引 = `blockIdx % mSingleBlocks`，N 索引 = `blockIdx / mSingleBlocks`；
  - `offsetA = mIdx·Ka·singleCoreM`（A 行主序 [M,Ka]，行块线性偏移）；
  - `offsetB = nIdx·singleCoreN`；`offsetC = mIdx·N·singleCoreM + nIdx·singleCoreN`（C 行主序 [M,N]）。
- `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, &tilingData.cubeTilingData)`：把 Matmul 对象与流水/切分参数绑定（必须在计算前）。
- 执行路径：
  - `Process<setTmpSpace>`：若平台需显式 UB 空间（310P，TilingKey 2），`pipe->InitBuffer(TBuf, localMemSize)` 后 `matmulObj.SetLocalWorkspace(...)`；910B 不需要。
  - `matmulObj.SetTensorA(aGlobal); SetTensorB(bGlobal); SetBias(biasGlobal);` → 整体版 `IterateAll(cGlobal)` 直接算完整输出写 GM；或逐块版见 §3。
  - 结束 `matmulObj.End()`。
- kernel 入口按 `TILING_KEY_IS(1/2)` 选模板分支执行。

## 3. 融合算子范式（05.04 `matmul_abs.cpp` 实录——与本题结构最相关）

- **C 落到 VECIN**：host 侧 `SetCType(TPosition::VECIN, CubeFormat::ND, DT_FLOAT)`——矩阵乘结果**留在片上 UB**，不先写回 GM，供 Vector 立即处理（这就是"融合省中间搬运"的实现机制）。
- **逐块迭代流水**（kernel 侧 `Process`）：
  1. `matmulObj.SetTensorA/B/Bias`；
  2. `while (matmulObj.Iterate<true>())`：每轮算 `baseM×baseN` 一块（`sync=true` 同步等待本块就绪）；
  3. `MatmulCompute()`：`reluOutLocal = queue.AllocTensor()` + `matmulObj.GetTensorC<true>(reluOutLocal, false, true)` 把本块 C 取到 UB LocalTensor；
  4. `AbsCompute()`：`AscendC::Abs(local, local, baseM*baseN)` **Vector 指令原地处理** → `EnQue`；
  5. `CopyOut(round)`：`DataCopy(cGlobal[offset], local, copyParam)` 带 stride（`baseM` 行、每行 `baseN`、行间跳 `(N-baseN)`）写回 GM 的对应子块 → `FreeTensor`；
  6. `computeRound++` 循环直到 `Iterate` 返回 false；`matmulObj.End()`。
- **要点**：matmul 结果块与 Vector 后处理块在 `TPipe` 队列上接驳；`baseM×baseN` 每次一块，天然支持后续做**分块归约**（每个结果块内部可先做部分归约，再跨块合并）。

## 4. 映射到 BatchMatmulMaxSum（当前理解；【推断】部分待 P2 验证）

- 本题对每个 batch b（`x1[b]:(M,K) × x2[b]:(K,N)`）执行 `C = BMM` 后需沿 **N 归约取 max**（得到 R:(M,)）再沿 **M 求和**（得到标量 y[b]）。输出 `y:(B,)` 恒 FP32。
- **融合点**：仿照 05.04——MatMul 的 C 声明为 `VECIN/UB`，每得到一块 `baseM×baseN` 就在 Vector 侧对 N 方向做部分 Max（`ReduceMax`/手写），把块级结果合并到 `R[b] 的 UB 行向量`；`baseM` 行对应不同 m，注意**同一 m 的 N 可能被切成多个 n 块**（`singleCoreN > baseN` 时逐 `Iterate` 轮切块），需对同 m 跨块结果做二次 max。
- **中间量 A ∈ R^{B×M×N} 不必完整落 GM**：这是相对"拆分三算子基线"的核心加速点（少一次 M×N 写回+读回，且融合 kernel launch）。
- **两级归约的差异**：MaxSim 沿 N 的归约必须在 **M×N 整矩阵**语义上进行（max over 全部 N 列再对 M 求和），与 matmul_abs 的逐元素 Abs 不同——归约跨块语义需要显式设计：
  - N 方向尾块：`SetOrgShape` 语义 + `Iterate` 处理非 baseN 余块时 Vector 归约须跳过越界元素或正确处理不足块；
  - M 方向求和是**顺序敏感的低风险累积**（FP32 累加），跨 m 块合并的累加顺序影响 1e-3 判据安全边际，P3 需在用例矩阵中验证。
- **B 维多核分配**（P4）：官方示例沿 M×N 二维切核（单 batch），本题 B 维还需在此之上叠加 batch 级并行或沿 M 切——与官方 `CalcOffset` 的 m/n 二维模式可类比扩展，待 P4 具体设计。
- 【推断】Vector 归约接口形态（`ReduceMax`/`ReduceSum` 在 UB LocalTensor 上、初值语义是否支持 -∞ 传播）需在 P2 前用官方 vector 归约实例（`03_intermediate_vector_operator_development`）或真机样例确认——特别是**全负行**用例要求"初值不得为 0"，须用首元素/极小负初值策略。
- 【推断】MatMul 高阶库对输入 A/B 的 ND 布局要求（是否要求内部切为分形、B 是否需要 [N,K] 转置语义）将直接影响本题 **transposeX1/X2 四 storage 布局** 的喂入方式——官方 ipynb（04.03/04.04）与 05.03/05.05 的 square_diff/sinh 变体源码可作为对照，P2 前需研读确认。

## 5. 已研读源文件索引（P2 继续研读的清单）

| 内容 | 绝对路径（根 `p:\Dev\CANN_Learning_Refs\Documentation_for_Developers\ascendc_operator_development\`） | 状态 |
|---|---|---|
| 04.03 host | `04_matmul_basic\answer\04.03_answer\op_host\matmul_custom.cpp` | 已读 |
| 04.03 kernel | `04_matmul_basic\answer\04.03_answer\op_kernel\matmul_custom.cpp` | 已读 |
| 04.04 host（int8 变体） | `04_matmul_basic\answer\04.04_answer\op_host\matmul_custom.cpp` | 已读 |
| 04.04 kernel（int8 变体） | `04_matmul_basic\answer\04.04_answer\op_kernel\matmul_custom.cpp` | 已读 |
| 05.04 host | `05_fused_operator_development\answer\05.04_answer\op_host\matmul_abs.cpp` | 已读 |
| 05.04 kernel | `05_fused_operator_development\answer\05.04_answer\op_kernel\matmul_abs.cpp` | 已读 |
| 04.03/04.04 教学 ipynb（72KB 高阶 API 讲解 / chapter_test） | `04_matmul_basic\` 下对应 ipynb | 待读（P2 前） |
| 05.03 square_diff / 05.05 matmul_sinh 变体 | `05_fused_operator_development\answer\` | 待读（P2 前） |
| vector 归约实例（ReduceMax/ReduceSum 范式） | `03_intermediate_vector_operator_development\` | 待读（P2 前） |

> 三镜像等价：`p:\Dev\CANN_Learning_Hub\tutorials\ascendc_operator_development\`、`p:\Dev\CANN_Learning_Workspace\Reference_And_Documentation\tutorials\ascendc_operator_development\` 内容一致；另有轻量版 `ascendc_operator_development_light\`（03 章 .asc 形态 MatMul/CV 融合练习）可作补充示例。
