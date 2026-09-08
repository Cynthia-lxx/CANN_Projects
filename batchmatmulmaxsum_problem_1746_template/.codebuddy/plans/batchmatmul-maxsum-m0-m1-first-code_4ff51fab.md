---
name: batchmatmul-maxsum-m0-m1-first-code
overview: 在本地空模板工程 Projects/batchmatmulmaxsum_problem_1746_template 上，以"独立编写+对照 P3 已 PASS 设计校验方向"的方式，先完成 M0（本地多 case 自测链扩展），再实现 M1（transpose=false、对齐 shape 域的 cube+vector 主链与 vector 兜底路径），形成可在云端 CANN Lab 冒烟通过的正确性骨架。
todos:
  - id: digest
    content: 用 [subagent:code-explorer] 产出头库 API 事实与官方/P3 只读 digest（签名+锚点），落盘 .codebuddy\plans
    status: completed
  - id: m0-scripts
    content: 扩展 gen_data.py 多 case 矩阵+manifest+批量 golden，改造 verify_result.py 全量判定与 run.sh 流程
    status: completed
    dependencies:
      - digest
  - id: m0-host
    content: 改造 main.asc 为 manifest 驱动的批量 host（动态 shape/dtype 逐 case 调 run_kernel 写 y_{cid}.bin）
    status: completed
    dependencies:
      - m0-scripts
  - id: m1-kernel
    content: kernel.asc 实现 run_kernel 分派与 cube 主链（CUBE_ONLY 位置、64 行 M 块、行向归约、同步后释放），静态自检签名
    status: completed
    dependencies:
      - digest
      - m0-host
  - id: m1-fallback
    content: kernel.asc 补 vector 兜底路径（任意小 shape/单点/K 非 16 对齐，-FLT_MAX 禁 0 初值），完成双路径整合与边界守卫
    status: completed
    dependencies:
      - m1-kernel
  - id: validate-deliver
    content: 本地全套静态自检（范式/宏/头库签名/清单一致性），生成云端 run.sh 冒烟指令交用户执行并据反馈修复
    status: completed
    dependencies:
      - m1-fallback
  - id: persist
    content: 写 2026-09-06 日志与 API 结论沉淀（标注与头库核对），Projects git 提交并 push
    status: completed
    dependencies:
      - validate-deliver
---

## 需求描述

在本地空工程 `Projects\batchmatmulmaxsum_problem_1746_template\` 中，从零独立编写 BatchMatmulMaxSum 赛题（problem_1746）的 Direct Invocation 实现，并为本轮交付做好本地多 case 自测验证链。经澄清，本轮范围与边界如下：

## 本轮范围（用户已确认）

- **编写起点**：从题目模板独立推导编写；`Projects\bmm_p2_slice`（隔壁工具试验产物，含已 PASS 的 P3 设计）仅作**只读对照**校验方向，禁止复制其代码。
- **交付目标（M0 + M1）**：
- M0：把模板"单 case 硬编码"的脚本链（gen_data / main.asc / verify / run.sh）扩展为**多 case 批量自测链**（形状矩阵 × dtype × 边界 shape）。
- M1：实现 `run_kernel`：**transpose=false（0:0 布局）域**内的正确性骨架 —— 对齐 shape 走 cube 主链（每 batch、64 行 M 块 + vector 行向归约），其余（单点、非对齐 M/N、K 非 16 对齐等）走 **vector 兜底路径**，保证域内任意整数 shape 语义正确；输出 y 为 (B,) fp32。
- 交付判据：本地仅做静态自检（头库签名/官方范式比对），云端 `run.sh` 冒烟全 case PASSED。
- **明确不做（后续里程碑）**：transposeX1/X2 四布局组合、N 非 8 对齐的大 shape、多核分片与性能优化（P4）。

## 技术栈

- 语言/形态：Ascend C（kernel.asc + main.asc Direct Invocation，asc 编译器，dav-2201/A2）；host 侧 acl runtime 直启。
- golden/数据：Python 3 + numpy + ml_dtypes（bf16），云端 python3 执行；golden 语义沿用现有 `BatchMatmulMaxSum.py::impl`（FP64 matmul→行 max→sum→fp32）。
- 权威参照：本地头库 `cann_ascendc_headers_9.0.0`（写码前核验 API）；官方直启样板 `Reference_And_Documentation\tutorials\ascendc_operator_development_light\03_simple_operator_practice\answer\03.03\`（纯 cube）与 `03.05\`（CV 融合）；P3 工程 `Projects\bmm_p2_slice\kernel.asc`（只读）。

## 实现思路（关键决策）

### 架构总览

host `run_kernel` 从 TensorGroupInfo 读取 B/M/N/K/dtype 并做**路径分派**（本轮仅 transpose=false）：

```mermaid
flowchart TD
    RK[run_kernel host: 读 shape/dtype/transpose 守卫] --> G{transpose!=false 或超出边界?}
    G -->|是| E[打印边界拒绝, 返回]
    G -->|否| C{可走 cube? N%8==0 且 K%16==0 且 M,N,K 在题域}
    C -->|否| V[vector 兜底: per b,m,n 分块点积+行max+sum]
    C -->|是| H[逐 b: 按 64 行 M 块循环]
    H --> Q1[launch cube 64xN 写 C 到 GM workspace]
    Q1 --> Q2[launch vector 块内归约: 每行 ReduceMax 后求和 → part]
    Q2 --> Q3[launch part_sum: part 累加进 y[b]]
    V --> Y[(y fp32 (B,))]
    Q3 --> Y
```

- **cube 主链（对齐域）**：逐 batch、逐 64 行 M 块 launch `__global__ __cube__`（形状 64×N×K，K%16==0 无需 K 补零）；**末块 r<64 行先经 vector 补零内核把 r 行拷入 64×K 零填充 workspace**（或经 tiling M=r 若核验可行），补零行行 max=0 不影响总和；C 写 bounded GM buffer（64×N×fp32 ≤ 2MB），随后 vector 归约内核按行分块 ReduceMax 并累加得到该块部分和，全部块完成后 part_sum 写入 y[b]。**归约方向：先 N 行 max 再 M 求和**（P3 教训：顺序反了会数值错）。
- **vector 兜底（任意小 shape/非对齐）**：泛化 P3 的单点点积路径——逐 b、m、n 做 K 分块乘加（fp32 累加），维护行 max（初值 -FLT_MAX，**禁 0**）再累加 m 行。只保证正确性，性能非本轮目标。
- **关键铁律（来自已 PASS 工程 + 头库）**：`ASCENDC_CUBE_ONLY` 必须定义在 `#include "lib/matmul_intf.h"` 之前（否则 507015）；cube tiling 用 SetDim(1) + **不调用 SetFixSplit**（v0(128,128) 会 FAILED），并校验 tiling.usedCoreNum==1；Matmul 对象流程 REGIST_MATMUL_OBJ→SetOrgShape/SetTensorA/B→IterateAll(C)→End；host 在**全部 launch 后显式 aclrtSynchronizeStream 再 aclrtFree**。
- **M0 测试链**：gen_data 生成 case 清单（文本 manifest 便于 C++ 解析）与各 case x1/x2 bin + fp64 golden；main.asc 按 manifest 循环构造 TensorGroupInfo 动态 shape/dtype 逐 case 调 run_kernel；verify 按判据（rtol/atol=1e-3）批量比对。case 矩阵覆盖：fp16/bf16、对齐中等 shape、M=1N=1 单点、N 非 8 对齐小 shape、K 非 16 对齐、全负区间（验证 -inf 初值正确）。

## 实现注意（防回归）

- 所有 kernel API 调用前二次核验头库与 digest（ReduceMax Level2 元素计数版签名/2201 fp32 每 repeat 64 lane 需分块 count≤64；Cast 4 参元素计数模式等），结论记录出处。
- 不改 `bmm_p2_slice`、不改头库；本地不做任何 C++ 编译/运行，仅静态文本检查。
- 云端 run.sh 若缺 ml_dtypes 需用户安装：`pip install numpy ml_dtypes`（云端 python3），不自装。
- 边界守卫诚实输出 stderr 并 return，避免越界 UB 污染结果。

## 目录结构

```
Projects\batchmatmulmaxsum_problem_1746_template\
├── kernel.asc                    [MODIFY] M1：run_kernel 分派 + cube 主链(3 个 kernel：补零/cube/块归约/part_sum) + vector 兜底点积路径；ASCENDC_CUBE_ONLY 位置铁律；-FLT_MAX 初值；同步后释放
├── main.asc                      [MODIFY] M0：读 input/cases.txt 批量循环，动态 shape/dtype 构造 TensorGroupInfo，逐 case 调 run_kernel，写 output/y_{cid}.bin
├── run.sh                        [MODIFY] M0：build → gen_data → 单次运行 binary（内部跑全 case）→ verify_result.py 全量判定
├── CMakeLists.txt                [MODIFY] 如需（一般不动；新增 host 侧 manifest 解析仅用标准库）
├── data_utils.h                  [不改]
└── scripts\
    ├── gen_data.py               [MODIFY] M0：多 case 矩阵 + 写 input/cases.txt manifest + 各 case bin + golden_y_{cid}.bin
    ├── BatchMatmulMaxSum.py      [MODIFY] 同步扩展 __main__ 自检 case 表（impl 语义不动，保持 golden 权威）
    └── verify_result.py          [MODIFY] M0：无参/传 all 时遍历 manifest 批量判定，rtol/atol=1e-3 可配
.codebuddy\
├── plans\PLAN_BatchMatMulMaxSum.md  [MODIFY] 追加本轮 M0/M1 执行记录与边界决策
└── memory\2026-09-06.md             [MODIFY] 当日日志 append：API 核验结论（标注"与头库核对"）、云端反馈与坑
```

## 关键结构参考（需按 digest 二次核对后落码，非逐字照抄）

- cube 内核参数形态（P3 同构，只读参照）：`__global__ __cube__ void bmm_cube(gm a, gm b, gm c, gm ws, TCubeTiling tiling)`；vector 归约 `bmm_block_reduce(gm c, gm part, int mRows, int n)` 每行分块 ReduceMax(≤64 lane)+Max 累加；`bmm_part_sum` 累加 partials 入 y。
- run_kernel 守卫骨架：B/M/N/K/dtype 一致性校验 → transposeX1||transposeX2 拒绝 → cube 条件 (N%8==0 && K%16==0 && !(M==1&&N==1)) 分派 cube/兜底。

## Agent 扩展

### SubAgent

- **code-explorer**
- 用途：一次性产出一份"API 事实 digest"——从 2697 个头文件树中核验 ReduceMax/ReduceSum/Max/Cast Level2 签名与 2201 平台语义、matmul_intf.h 的 REGIST_MATMUL_OBJ 与 ASCENDC_CUBE_ONLY 条件编译分支、tiling_api 的 TCubeTiling/MultiCoreMatmulTiling 关键方法；并从官方 03.03/03.05 样板与 bmm_p2_slice（只读）抽取 kernel 结构与 tiling 探针结论，附 file:line 锚点。digest 落盘 `.codebuddy\plans\`，供 M1 写码直接引用，避免主代理重复大范围检索。
- 预期产出：签名/枚举/宏位置精确、与头库一致的实现参照文档（禁止把 P3 代码抄入工程）。