# BatchMatmulMaxSum M0/M1 API & 架构 Digest

> 2026-09-06 产出，供 kernel.asc/main.asc/scripts 编写直接引用。
> 标注【头库】= 与 `cann_ascendc_headers_9.0.0` 核对；【P3】= 与 `Projects\bmm_p2_slice` 已 PASS 工程核对（只读）；【模板】= 本工程现状。
> 本地无编译器：所有 C++ 仅静态自检，云端 run.sh 验证。

## 0. 最终实现的架构决议（2026-09-06 傍晚代码为准；原稿部分已被实现修订覆盖）

1. **数据布局 = LOGICAL（与 judge 一致）**：gen_data 写 `x1 (B,M,K)`、`x2 (B,K,N)` 连续 row-major；main.asc 按 logical 字节分配。反对 P3 的"磁盘零填充 padded 布局"捷径，不沿用。
2. **pad 内化到 host run_kernel（GM workspace 全零 memset + host 逐行 D2D memcpy）**：实现上用 `aclrtMemsetAsync` 把 aPad/bPad/bias/part 全部精确置 0，再用 `aclrtMemcpyAsync(ACL_MEMCPY_DEVICE_TO_DEVICE)` 逐行拷真实区（每行只拷 K1/N 真实字节）。pad 尺寸：`Mpad=PadUp(M,64)`、`Kpad=PadUp(K,16)`、`Npad=PadUp(N,64)`（Npad%64==0 保证 C tile 每 64-float 段 256B 对齐）。
   - 原稿曾设想"vector pad-copy 内核"，已弃用 → host D2D 行拷贝更简单且同流有序。
3. **M1 支持范围 = 全域单路径 cube**（正确性骨架，性能延后）：
   - 单一 pad-cube 管线处理**所有**题域内 shape：0:0 布局、B∈[1,64]、M/N∈[1,8192]、K∈[32,8192] 且 K%8==0。1×1 单点（case3）不特判，pad 后变普通 64×Npad×Kpad tile；M/N 任意非对齐、K 非 16 对齐同理。
   - 逐 batch、逐 64 行块：cube(64,Npad,Kpad) → C(64×Npad fp32) → `bmm_tile_reduce`(realRows、realCols=N、nPad=Npad，行 max 初值 -FLT_MAX) → 该 batch 块全完后 `bmm_part_sum`(partPad=((blocks+7)/8)*8) 进 y[b]。归约方向：**先 N 行 max 再 M 求和**（顺序反了数值错，P3 教训）。
   - 原稿"vector 兜底路径（单点/非对齐走 dot kernel）"已被 pad-cube 方案取代：不再需要独立 vector 兜底 kernel；m1-fallback todo 的剩余工作仅为守卫核验与注释确认。
   - workspace 预算守卫：rowCopies≤2^18、aPad+bPad+C≤1GiB，超限 stderr 诚实 return（后续里程碑再扩 K 分片等）。
4. **dtype 语义**：输入 dtype=1(fp16)/2(bf16)；`y` fp32。cube 两套实例：`bmm_cube_fp16`（half）与 `bmm_cube_bf16`（bfloat16_t），C=float、bias=float 全 0。
5. **本工程冒烟链路与 judge 一致性优先**：每轮验证即 run.sh（build→gen_data→binary→verify），binary 一次跑 manifest 全 case。
6. **tiling 探针**：`GenBlockTiling` 每次全新 `MultiCoreMatmulTiling`，SetDim(1)+不调 SetFixSplit（官方 (128,128) 探针会 FAILED，v1 no-fix 为 PASS 候选）+SetBias(true)+SetBufferSpace(-1,-1,-1)；**accept 条件在 probe 循环内**：usedCoreNum==1 && M==m && N==n && singleCoreM/N==tiling.M/N && Ka==Kb==k。

## 1. 【头库】vector 归约 / Cast 权威签名（写入 kernel 前已核对）

来源头：`asc\include\basic_api\kernel_operator_vec_reduce_intf.h`
- ReduceMax **Level2 元素计数**（289-291 行）：
  ```cpp
  template <typename T>
  __aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
      const LocalTensor<T>& sharedTmpBuffer, const int32_t count, bool calIndex = 0);
  ```
  即 `ReduceMax(dst, src, sharedTmpBuffer, count)`；`count` 为元素数。
- ReduceSum 同形（同一文件内 ReduceSum 系列）。P3 已 PASS 形态：`ReduceSum<float, true>(part, chunk, tmp, seg)`、`ReduceMax(part, chunk, tmp, seg)` —— 整段归约写 `part[0]`。
- **2201 平台**：fp32 向量每 repeat = 64 lane（P2b 已证），count≤64 整段归约安全（sub-64 自动 mask 到 count lane）。行向 max 按 ≤64 分块循环 + `Max(dst,src0,src1,1)` 合并。
- Cast 元素计数：`Cast(dst, src, roundMode, count)`，half→float 用 `RoundMode::CAST_NONE`（P3 已 PASS 用法）；bf16→float 同路径（bfloat16_t 为头库原生类型，见 vector_compute.h 大量 `asc_bfloat162float`）。带元素计数版本的 dst/src 皆 LocalTensor。
- `Duplicate(dst, scalar, count)`：生成常量向量（初值 `-FLT_MAX`/0）。已 PASS 用法 `Duplicate(outLocal, 0.0f, KCH)`。
- `Max/Add(dst, src0, src1, count)`：元素级 count 模式，offset 0 起（P3 已证 `Max(rowMax,rowMax,part,1)`）。
- `DataCopy(dstLocal, srcGlobal, count)`：GM→UB，count 元素；要求 32B 对齐（P3 中每段 `seg%8==0` fp32 / 64 元素 half 128B）。`DataCopyPad + DataCopyExtParams{blockCount, blockLen, srcStride, dstStride, ...}` 用于 GM 4B 精确写出（P3 写单 fp32 part）。
- GM 视角：kernel 内 `GlobalTensor<T>` + `SetGlobalBuffer(__gm__ T*, count)`；参数直接 `__gm__ uint8_t*`。

## 2. 【头库】Cube Matmul 高阶对象

- include 次序铁律（P3 已 PASS；509 系列错误根因）：
  ```cpp
  #define ASCENDC_CUBE_ONLY
  #include "lib/matmul_intf.h"
  #include "tiling/platform/platform_ascendc.h"
  #include "tiling/tiling_api.h"
  using namespace matmul;
  ```
  【头库】`asc/include/adv_api/matmul/matmul_intf.h:86-113`：`SPLIT_CORE_CUBE + ASCENDC_CUBE_ONLY` → `using Matmul = MatmulServiceAux<...>`（REGIST_CUBE_OBJ 走 InitCurObj 单核，否则 KfcServer 多核循环 507015）。`REGIST_MATMUL_OBJ == REGIST_CUBE_OBJ`（36-37 行）。
- kernel 内序列（P3 已 PASS）：
  ```cpp
  __global__ __cube__ void bmm_cube(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* bias,
                                    __gm__ uint8_t* c, __gm__ uint8_t* workspace,
                                    AscendC::tiling::TCubeTiling tiling) {
    TPipe pipe;
    GlobalTensor<half> aG; aG.SetGlobalBuffer((__gm__ half*)a, M*Ka);   // 形状由 tiling 字段给
    GlobalTensor<float> cG; ...;
    Matmul<MatmulType<GM, ND, half>, MatmulType<GM, ND, half>,
           MatmulType<GM, ND, float>, MatmulType<GM, ND, float>> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
    mm.SetTensorA(aG); mm.SetTensorB(bG); mm.SetBias(biasG);
    mm.IterateAll(cG); mm.End();
  }
  ```
  bf16 版把模板与 buffer 类型换成 `bfloat16_t`。
- TCubeTiling 关键字段（写 host guard 用）：`M N Ka Kb baseM baseN baseK usedCoreNum singleCoreM/N/K isBias`。
- 单 cube core 保证：tiling 探针 `SetDim(1)` 且 **不调用 SetFixSplit**（P3：v0 official SetFixSplit(128,128) 在 N 多变时会 FAILED，v1 no-SetFixSplit 为 PASS 探针）；成功后校验 `usedCoreNum==1 && singleCoreM==tiling.M && singleCoreN==tiling.N && singleCoreK==tiling.Ka && Ka==Kpad`。
- host tiling 构建（P3 已 PASS）：
  ```cpp
  platform_ascendc::PlatformAscendC* plat = platform_ascendc::PlatformAscendCManager::GetInstance();
  matmul_tiling::MultiCoreMatmulTiling ct(*plat);
  ct.SetDim(1);
  ct.SetAType(matmul_tiling::TPosition::GM, CubeFormat::ND, DT_FLOAT16); // bf16 时 DT_BF16?
  ct.SetBType(...同); ct.SetCType(..., DT_FLOAT); ct.SetBiasType(..., DT_FLOAT);
  ct.SetShape(m,n,k); ct.SetOrgShape(m,n,k); ct.SetBias(true); ct.SetBufferSpace(-1,-1,-1);
  ct.GetTiling(tiling)   // -1 => 失败，探针下一候选
  ```
  每次 GetTiling 用全新对象。kMaxProbes 8；候选：v0(128,128 fix)→v1(no fix)→v2(adaptive)→v3(16,16)→… 首个有效者胜。
- workspace：`plat->GetLibApiWorkSpaceSize()` 字节，kernel 经 `GetSysWorkSpacePtr()` 访问；host 传 `aclrtMalloc` 的 GM ws。

## 3. 【P3 只读】已 PASS 管线要点（方向锚，不抄代码）

路径 `Projects\bmm_p2_slice\kernel.asc`（P3 route③，6/6 case PASSED）：
- `bmm_batch_cube`：单 64 行 M 块 cube，fp16x fp16 → fp32 C，bias 全 0。launch `<<<1, nullptr, stream>>>` per (batch, block)。
- `bmm_block_reduce`（vector）：对一块 C tile 逐 realRows 行、行内按 ≤64 fp32 分块 `ReduceMax(part, chunk, tmp, seg)` + `Max(rowMax,rowMax,part,1)` 合并；**行初值 `-FLT_MAX`（非 0，防全负行被归成 0）**；`Add(acc,acc,rowMax,1)` 累计；`DataCopyPad` 精确 4B 写 part。
- `bmm_part_sum`（vector）：把该 batch 的 partPad 个 part 累加写 y[b]。
- host：guard 打印 → 单点 1×1 走 vector dot → 否则 tiling 探针（v1 no-fix 胜出）→ 校验 Ka/Kb==Kpad、usedCoreNum==1 → 分配 bias(全0)、cTile、libws、part → memset → 逐 b/block launch → **`aclrtSynchronizeStream(stream)` 后才 free**。
- 归约方向教训：**先 N 行 max 再 M sum**；顺序反了数值错。
- `main.asc`：case 表 + argv 选 case + `X1_BYTES` 等从形状公式派生 + `aclrtSynchronizeStreamWithTimeout(100s)`。run.sh：build → gen_data → 循环跑 + verify（tol 见 verify spec）。

## 4. 【模板】本工程结构与 M0/M1 落点

- `kernel.asc`：空 run_kernel（host 上下文）。run_kernel 签名：
  `extern "C" void run_kernel(GM_ADDR x1, const TensorGroupInfo& info_x1, GM_ADDR x2, const TensorGroupInfo& info_x2, GM_ADDR y, const TensorGroupInfo& info_y, int64_t availableCoreNum, aclrtStream stream, bool transposeX1, bool transposeX2)`
  被 main.asc include；launch 用 `<<<grid, nullptr, stream>>>`。
- `TensorGroupInfo{tensors,numTensors}`/`TensorInfo{shape,numDims,dtype}`（main.asc 与 kernel.asc 各自 ifndef 定义同一份）；dtype 0=fp32 1=fp16 2=bf16。
- `main.asc`：单 case0 硬编码 logical (1,1,32)；**M0 改 manifest 批量**。
- `run.sh`：单 case 流程；**M0 改全量**。`CMakeLists.txt`：AS CXX14、dav-2201、链接 tiling_api/register/platform/...；一般不动。
- `data_utils.h`：ReadFile(filePath,bufferSize,buffer,bufferLen) / WriteFile(filePath,buffer,size)。

## 5. M0 数据链契约（gen_data / cases.txt / main / verify 必须一致）

- `input/cases.txt`（manifest，C 可解析）：每行 `cid B M N K dtype tx1 tx2`；dtype 1=fp16 2=bf16。
- `input/case{cid}/x1.bin`：logical `(B,M,K)` 2B/元素 row-major；`x2.bin`：(B,K,N)。`output/case{cid}/golden_y.bin`：(B,) fp32（由 BatchMatmulMaxSum.impl 的 FP64 matmul→行 max→sum 给出）。
- main：循环 manifest 逐 case：分配 logical 字节 H/D buffers → H2D → run_kernel → sync → D2H → `output/case{cid}/y.bin`。失败打印不中断，出口码聚合。
- verify_result.py：支持 `verify_result.py`（无参=全 manifest）。判据 `np.isclose(rtol=1e-3, atol=1e-3)` 严格 tol=0。
- M1 case 矩阵（对齐 anchor / 多 batch / bf16 cube / 单点 / M 尾块 / N 非 16 尾 / K 非 16 / 全负 / bf16 小非对齐 / 中 shape K=40）。

## 6. 待写码时核验（不能凭记忆）

- bf16 cube：`DataType::DT_BF16` 枚举确切名（matmul tiling 中 SetAType 用；头库枚举核实）与 `bfloat16_t` 在 kernel 命名空间的确切类型名。
- pad-copy vector kernel：GM 段搬移参数（DataCopyPad 精确 1B/2B 粒度行拷贝可行性，含 blockLen 单位）——M1 kernel 起草时回官方教程 + 头库。
- `aclrtMemsetAsync(devPtr, size, 0, size, stream)`（P3 host 用法）签名已在 main 编译范围可用。

## 7. 实现快照（2026-09-06 傍晚；写码/续接前先读工程文件核对）

### 工程文件清单（Projects\batchmatmulmaxsum_problem_1746_template\）
- `kernel.asc`（M1 已实现，~627 行）：pad helper（kBlockRows=64/kKAlign=16/kNAlign=64/PadUp/Mpadded/Kpadded/Npadded/BlockCount/kMaxPadBytes/kMaxRowCopies）→ `TileProbe` 梯 `GenBlockTiling` → `bmm_cube_fp16`/`bmm_cube_bf16`（`__cube__`，参数 aPad/bPad/bias/cTile/workspace/tiling）→ `bmm_tile_reduce`（`__vector__`）→ `bmm_part_sum`（`__vector__`）→ `run_kernel`（守卫+预算→分配 6 块 GM→memset→逐行 D2D→逐 b/block launch→`aclrtSynchronizeStream`→free）。
- `main.asc`（M0 已实现）：manifest 批量，8 字段 `cid B M N K dtype tx1 tx2`，逐 case 动态 TensorGroupInfo，H2D→run_kernel→`aclrtSynchronizeStreamWithTimeout(120000)`→D2H→写 `./output/case<id>/y.bin`；退出码聚合。argv[1]=cid 可单跑。
- `scripts/gen_data.py`（M0）：CASES 10 条（0-9），`input/case*/x1.bin|x2.bin`（logical 2B/elem）+ `output/case*/golden_y.bin`（impl FP64）+ `input/cases.txt` manifest。seed=20260906+cid。
- `scripts/verify_result.py`：case0 rtol/atol 1e-4（fp16 anchor），其余 1e-3；manifest 驱动，无参全跑/可传 cid。
- `scripts/BatchMatmulMaxSum.py`：`impl` 权威 golden（FP64 matmul→行 max→sum→fp32）；__main__ 自检（注意：case 表与 gen_data 的 CASES 独立，含 10-13 transpose 逻辑条目，仅 golden 语义验证用）。
- `run.sh`：无参=全 case；`bash run.sh cid`=单 case（binary 与 verify 都支持）。
- `CMakeLists.txt`/`data_utils.h`：未动。

### 当前遗留（validate-deliver 要做）
1. gen_data.py case7 改异号区间（见 daily log）使其真测全负行。
2. BatchMatmulMaxSum.py __main__ 删除 `if cid == 2: assert y<0` 误断言。
3. py_compile 全 scripts + read_lints + manifest/路径一致性复查。
4. 输出云端 run.sh 冒烟指令给用户，据反馈修 bug。
5. persist：daily log（已写）+ digest（已更新）+ git commit+push（模板工程全部改动 + .codebuddy 计划/日志按项目习惯提交）。

