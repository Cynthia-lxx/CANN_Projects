---
name: 探针B-UB镜像自检与官方基准对照
overview: 设计"组合探针 B"单次云端往返：①cube_probe 主区保留单 tile 门控写回，同时把 GetTensorC 产物原样连续镜像到 GM 边带，host 比对镜像与 golden，判定错位源在 GetTensorC/L0C 产物还是 strided 写回；②原样复制官方 matmul_abs 教程答案为独立 target 跑通，建立 C220/9.0.0 工具链上该 matmul 范式的通过性基准。
todos:
  - id: mirror-probe-kernel
    content: 修改 cube_probe.asc：GM buffer 扩至 M*N+1024，CopyOut 内对同一 UB 源追加连续镜像 DataCopy 到 cGlobal[M*N]
    status: completed
  - id: mirror-probe-host
    content: 修改 cube_probe.asc host：cFileSize 扩容，新增镜像区整体 mismatch 与逐行 correct 解析打印，保留主区统计
    status: completed
    dependencies:
      - mirror-probe-kernel
  - id: official-baseline
    content: 新增 off_matmul_abs.asc（官方答案逐字副本），CMakeLists 加 target，run_probe.sh 串联运行两可执行
    status: completed
  - id: commit-push
    content: 本地静态核对（无 tiling.K 类残留、镜像偏移不越界）后 git commit 并 push 供云端复跑
    status: completed
    dependencies:
      - mirror-probe-host
      - official-baseline
---

## 需求概述

基于探针 A 云端输出（单 tile 门控后 GM 仅 tile0 区与第 32 行有非零值：row0 全对，row m≥1 的值=63·m·(n+1)，golden=64·(m+1)·(n+1)），设计并实施**组合探针 B**：一次云端往返收集三路证据，将根因从"GetTensorC/L0C→UB 产物错误"与"strided DataCopy 写回错误"之间二分，并用官方 matmul_abs 基准校验该 matmul 范式在当前工具链（910B dav-2201 / CANN 9.0.0）是否整体成立。

## 核心功能点

1. kernel：沿用探针 A 的"只计算并写回首个 tile"门控；在 DeQue 后对同一 UB 源 `cTileLocal` 追加一次**连续无 stride 的镜像 DataCopy**，把整块 1024 个 fp32 原样搬到 GM 新增的边带区 `cGlobal[M*N .. M*N+1024)`，与主 strided 写回同源同运行。
2. host：C 的 GM 尺寸与 `cFileSize` 扩至 `(M*N+1024)` 元素；主区统计/行 dump 保留；新增镜像区解析——镜像整体 mismatch 计数、逐行正确数（行 i 期望 64·(i+1)·(n+1)），用以区分"镜像==golden（主写回错误）" vs "镜像==63m 型（GetTensorC 产物错误）"。
3. 官方基准：新增 `off_matmul_abs.asc`（官方 matmul_abs 答案逐字副本，仅本 target 独立编译），CMake 新增同名可执行 target，run_probe.sh 依次运行两个可执行。
4. 本地编辑 + 静态核对后 git commit/push，云端 `git pull && bash run_probe.sh` 取三路输出。

## 技术方案

### 实现思路

探针 B 不改变探针 A 已建立的"单 tile 门控"实验条件，仅在数据通路上加一个"旁路探针"：把 GetTensorC 之后、Free 之前的 UB tile 数据以连续 DataCopy 镜像到 GM 边带区。主区写回（strided、含 dstStride）与镜像写回（连续、无 stride）读取同一份 UB 数据，host 分别比对 golden，即可判定错误发生在"数据产生（GetTensorC/L0C）"还是"strided 搬移（DataCopy 参数解释）"。

### 关键代码位置与改动

- `cube_probe/cube_probe.asc`
- kernel `Init()`：`cGlobal.SetGlobalBuffer(c, tiling.M * tiling.N)` 扩为 `tiling.M * tiling.N + 1024`（镜像边带，1024 = baseM*baseN = tile 元素数）。
- `Process()`：维持探针 A 的 `if (Iterate<true>) { MatmulCompute(); CopyOut(0); }` 单 tile 门控不变。
- `CopyOut()`：DeQue 之后、FreeTensor 之前追加镜像 DataCopy：`DataCopyParams {blockCount=32, blockLen=baseN*sizeof(CType)/32, srcStride=0, dstStride=0}`，目标 `cGlobal[tiling.M * tiling.N]`（即元素 4096 起）。主区 strided 写回保持与官方逐字一致、不动。
- host `main()`：`cFileSize = (M*N+1024)*sizeof(float)`，memset/memcpy 尺寸同步；保留主区 mismatch/quadrant/dump；新增镜像解析：镜像整体 mismatch 数 + 逐行 correct/32 + 行 0/1 首值打印（期望镜像行 i = 64·(i+1)·(n+1)，i=0..31, n=0..31）。
- `cube_probe/CMakeLists.txt`：新增 `add_executable(off_matmul_abs off_matmul_abs.asc)`，链接库/头目录/编译选项与 cube_probe 相同（tiling_api/register/platform/unified_dlog/dl/m/graph_base，dav-2201）。
- `cube_probe/off_matmul_abs.asc`【NEW】：官方答案 `matmul_abs.asc`（来源：p:\Dev\CANN_Learning_Refs\CANN_Learning_Hub_(for_dev)\tutorials\ascendc_operator_development_light\03_simple_operator_practice\answer\03.05\matmul_abs.asc）逐字复制，kernel 入口名与 host 调用保持原样 `matmul_abs`（不同可执行 target 内符号独立，无冲突）；不改输入、不改 golden 判定（std::equal 对常量 511.5）。
- `cube_probe/run_probe.sh`：构建后依次执行 `./build/cube_probe` 与 `./build/off_matmul_abs`，两段输出用分隔行隔开。

### 判定矩阵（云端输出解读）

| 镜像区 | 主区 | 结论 |
| --- | --- | --- |
| ==golden（1024 全对） | 仍 63m 型 | GetTensorC 产物正确；错误收敛到主 strided DataCopy（srcStride=0/dstStride 语义或 GlobalTensor 索引）→ 下轮改造写回 |
| ==63m 型（仅行 0 对） | 同左 | UB 内容（GetTensorC/L0C→UB）本身错误，DataCopy 无责 → 下轮查 fp32 VECIN 的 GetTensorC 语义/平台行为 |
| off_matmul_abs PASS | — | 本 probe 的 shape/无 bias/参数差异才是触发条件，逐项回退对照 |
| off_matmul_abs FAIL | — | 该 adv_api matmul 范式在此工具链/平台存在系统性限制，需转向保守用法或纯手工方案再议 |


### 风险与规避

- GM 越界：kernel 侧 `cGlobal.SetGlobalBuffer` 长度必须同步扩为 M*N+1024；host 分配/清零/回读尺寸全部用新 `cFileSize`，避免镜像区写入未分配内存。
- DataCopy 参数合法性：blockCount=32、blockLen=4（128B=32 个 fp32）、双 stride=0，合计 128 个 32B block 连续搬移，均在 uint16 上限内。
- 官方文件平台差异：与 cube_probe 使用同一 include 集与编译选项，官方 API（GetTensorC/Abs/REGIST_MATMUL_OBJ）在 9.0.0 头库中均存在，风险低；若个别平台宏分支报错，将错误信息贴回再评估，不做本地猜测性改写。
- 本地不编译不运行；仅文本编辑 + git commit/push。

### 目录结构

```
Projects/batchmatmulmaxsum_problem_1746_template/cube_probe/
├── cube_probe.asc        # [MODIFY] GM 扩容、镜像 DataCopy、host 镜像解析
├── off_matmul_abs.asc    # [NEW] 官方 matmul_abs 答案副本（独立 target 基准）
├── CMakeLists.txt        # [MODIFY] 新增 off_matmul_abs 可执行 target
└── run_probe.sh          # [MODIFY] 构建后依次运行 cube_probe 与 off_matmul_abs
```