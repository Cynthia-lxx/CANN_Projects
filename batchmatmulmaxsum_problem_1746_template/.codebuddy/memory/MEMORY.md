# MEMORY.md — CANN_Learning_Workspace 长期记忆

> 本文件是本工作区的稳定长期记忆（就地维护）。日度过程记录见 `YYYY-MM-DD.md`；算子知识体系见 6 个主题文件（地图见 `ascendc_knowledge_index.md`）。

## 工作区身份与目标

- 本工作区 `p:\Dev\CANN_Learning_Workspace` 是 CANN 算子工程学习与开发工作区：所有算子工程放 `.\Projects\`（独立子文件夹），官方参考文档副本放 `.\Reference_And_Documentation\`。
- 我的角色：CANN 编程助手，依据本地官方文档编写可在昇腾 NPU 运行的 Ascend C 算子工程，代码由用户上传云端 CANN Lab 编译/测试。

## 环境硬约束（每次任务必守）

- 本地 **无** CANN Toolkit、NPU 驱动、C/C++ 编译器 → **禁止**安装此类软件、**禁止**本地编译/运行任何 C++ 代码、**禁止**动态分析。
- 只允许静态文本检查（头文件是否包含、API 名称/签名、Tiling 字段匹配、代码范式比对官方示例）。
- pip 依赖不得自行安装：需把「包名 + PIP 可执行位置 + 安装命令」告诉用户，等用户装完再做完整性验证。
- 所有会话记忆/计划/任务/决策记录存放在 `.codebuddy\` 下（不放在其他位置）。

## 文档与知识底座（核心资产）

- 官方文档仓库：`Reference_And_Documentation\`（CANN Learning Hub 完整副本）。
- 重点精读对象：`tutorials\ascendc_operator_development\`（完整课程 9 章，工程化/op_host+op_kernel 路线）与 `tutorials\ascendc_operator_development_light\`（精简 4 章，Kernel 直调路线）。
- **文档优先原则**：写任何算子代码前必须查本目录教程/示例；API 用法、编程范式、Tiling 策略以本地官方文档为准，禁止臆造；与文档不符视为错误。
- 知识沉淀：`.codebuddy\memory\` 下 6 个主题文件（范式/模板/Tiling/协作/API/调试），每条要点带来源锚点（章节或源码路径），写作与引用时回到官方原文。
- 知识底座状态：2026-09-05 已完成两大核心课程（主课 9 章 + light 4 章）的系统性精读与落盘；入口文件 `ascendc_knowledge_index.md`（含主题↔章节对照与检索路径）；后续写任何算子先经该索引定位再回到官方原文核对。
- 文档矛盾/模糊时：停下询问用户，不擅自猜测（规则 6.2）。

## 工程规约（写算子工程时）

- 工程放 `.\Projects\{op_name}\`，内含 op_host/、op_kernel/、CMakeLists.txt 等。
- 命名 snake_case（文件/算子），类名 PascalCase。
- 核函数声明须 global + aicore/vector/cube 修饰，参数顺序 输入→输出→workspace→tiling。
- 工程代码风格、头文件顺序与官方 msopgen 示例一致；Tiling 考虑 32B 对齐与多核负载均衡。
- `.\Projects\.git` 是项目级 git 仓库（远程 github.com/Cynthia-lxx/CANN_Projects.git）：每完成一次 Projects 改动后提交并 push。.codebuddy 记忆文件不在该仓库范围。
- CANN 版本基线：教程要求 Atlas A2/A3、CANN 9.0.0+、Python 3.11（云端）。

## 记忆文件地图

| 文件 | 内容 |
|---|---|
| `MEMORY.md` | 本文件：稳定事实、约束、规约 |
| `YYYY-MM-DD.md` | 每日阅读/开发日志（过程追溯） |
| `ascendc_knowledge_index.md` | 六主题文件索引 + 主题↔来源章节对照 |
| `ascendc_01_programming_paradigm.md` | 编程范式（CopyIn→Compute→CopyOut、TPipe/TQue、缓冲、事件） |
| `ascendc_02_operator_templates.md` | 算子实现模板（逐元素/规约/Matmul/融合） |
| `ascendc_03_tiling_strategy.md` | Tiling 策略（多核/核内均分、对齐、Tbuf/Workspace） |
| `ascendc_04_host_device_collaboration.md` | Host(原型/InferShape/Tiling) ↔ Device(Kernel) 协作与工程结构 |
| `ascendc_05_api_reference.md` | 常用 API（DataCopy/Add/Matmul/DumpTensor/printf…）签名与要点 |
| `ascendc_06_debug_common_errors.md` | 典型错误模式与调试方法（CPU 孪生/NPU 上板/msProf） |

## 使用约定

- 语言：简体中文。
- 写新算子前先查 `ascendc_knowledge_index.md` → 主题文件 → 官方原文，按模板展开。
- 更新主题文件用增量追加或就地修订；每章/每轮完成后写当日日志。

## 判题平台提交规则（batchmatmulmaxsum 实测，2026-09-06）

- 提交入口**只允许编辑 `kernel.asc`**；模板其余文件全 readonly；"新建文件"入口仅允许 `.asc`/`.h`。
- 平台静态扫描**禁止输出/system 类 API**：`printf/cout/cerr/puts/write/syscall/dlopen/dlsym/asm/freopen`（"如…"列举式，同族的 fprintf/fputs/fopen/stdout/stderr 一律视为被禁）。提交前必须全文扫零命中。
- 判题用平台私有 host driver 调提交 kernel.asc 里的 `run_kernel`（GM_ADDR 签名）；main.asc 只是 readonly 本地样例。域外用例 guard 只能**静默 return**，无打印通道。
- **编译坑（CE 实测 2026-09-07，probe-01）**：judge host driver 编译 TU 中 `__gm__` 是真实地址空间限定符（Clang），`reinterpret_cast`/`const_cast` 不能去掉它（`error: reinterpret_cast from '__gm__ uint8_t *' to 'uint8_t *' is not allowed`）→ GM 指针一律用 **C 风格强转** `(uint8_t*)y` / `(__gm__ float*)x`（此前占位版能跑 RE 即靠此写法）。
- 注释/字符串也别写被禁 token（保险起见中文注释优先）。
- Debug 走本地/云端自测链（本地工程 main.asc + cloud CANN Lab），判题环境无输出。
- **Profiling 硬规则（实测 2026-09-06）**：`Profiling rule violated: each iteration must launch exactly 1 kernel. Expected 75 launches, got N.` Expected 恒 75；M1 多 launch 版 got 2065，占位版（每 run_kernel 恰 1 launch、但守卫静默 return）got 20 仍未过；profiling 检查先于数值结果返回 → 至今未见 precision/用例表字段。
- **E75/G20 推断（2026-09-07）**：75 = 15 用例 × 5 iteration；got=20 = 恰好 4 个 false/false 用例 × 5 → 守卫 return 制造「0-launch iteration」是主因（M1 got=2065≈4×5×103 侧证）→ 消除守卫（转置/全域都算 + 每 run_kernel 恰 1 launch）是 profiling 通过关键。**探针先行已定**：probe-01 = 无守卫空 kernel 恰 1 launch。详见 `2026-09-07.md`。
- BatchMatMulMaxSum 判题冲刺**完整交接主文档**：`.codebuddy/memory/handoff_batchmatmulmaxsum_judge_2026-09-06.md`（版本演进史/git 状态/硬经验/疑点/探针实验方案/提交前清单）。
- **1746 攻坚当前状态指针（2026-09-08）**：cube 取证收口 = 纯 cube 形态（`ASCENDC_CUBE_ONLY`+`__cube__`+CType GM+`IterateAll(cGlobal)`，全文件无 vector）云端 6/6 有效 shape 0 mismatch；`GetTensorC→vector` 归约通道经 cube_probe mode0..3 复测判死为跨次运行竞态 → v2 定「分相」架构（纯 cube 整写 GM C workspace + 独立 vector 核归约 y[b]）。⚠️ 判题平台曾实测「每 iteration 恰 1 launch」与 v2 多 launch 的合规性冲突**未决**，待赛方口径。F6 cube_mc（纯 cube 多核拼写）/ F7 v2_proto（分相 e2e）已推送 commit f7e3a72 待云跑。完整证据：`.codebuddy/memory/2026-09-08.md`、`.codebuddy/plans/PLAN_1746_probe_F_matmul_multitile_2026-09-08.md`。
