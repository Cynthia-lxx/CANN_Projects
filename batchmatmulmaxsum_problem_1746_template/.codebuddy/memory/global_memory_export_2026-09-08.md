# 全局记忆导出（2026-09-08）— problem_1746 交付存档

> 此文件是把 AI「全局记忆（knowledge store）」中与本项目相关的条目原样落盘到项目 `.codebuddy/`，
> 供后续接手 Agent 不依赖原 knowledge store 也能完整恢复上下文。知识 ID 保留以便原库同步/溯源。

## KB 条目 72154916 — problem_1746 batch_matmul_max_sum 攻坚现状（2026-09-08 版）

problem_1746 batch_matmul_max_sum（Ascend 910B/2201, CANN9.0.0, fp16+bf16, 无 bias, y[b]=Σ_m max_n A·B）攻坚现状（2026-09-08）：
(1) 判题 15 点统一 Fail 的 TypeError 已由赛方核实为评测机脚本缺陷，与算子无关，v1 真实分未知。
(2) F 系列取证收口：matmul 高 API 里凡 C 经 GetTensorC 读入 UB 再被 vector 消费即跨次运行不稳的竞态（cube_probe mode0-3 同构 GGPP 复测证实），
   唯一稳定配方 = 官方 03.03 纯 cube：`#define ASCENDC_CUBE_ONLY` + `__global__ __cube__` + CType=GM + `mm.IterateAll(cGlobal)` 整写 GM C、
   文件全程无 vector（F5 云端 6/6 shape 0 mismatch，tailN=1000/K=200 亦过；M<baseM/N<baseN 时 tiling 拒收）。
(3) v2 架构定为「分相」：纯 cube 逐 batch 整写 M×N fp32 C 到 GM workspace + 独立 __aicore__ vector 核按 v1 惯用法归约 y[b]，host 同 stream 顺序 launch。
(4) ⚠️ 未决冲突：判题平台曾实测「每 iteration 恰 1 kernel launch」(Expected 75)，v2 多 launch 是否合规待赛方口径，勿默认可多 launch。
   完整证据/判读/云跑命令在 .codebuddy/memory/2026-09-08.md 与 plans/PLAN_1746_probe_F_matmul_multitile_2026-09-08.md
   （Git 历史含 f7e3a72=F6 cube_mc/F7 v2_proto 探针）。

## KB 条目 37027485 — CANN_OpHelper（相邻项目助手身份，供上下文参考）

我是 CANN_OpHelper 项目的开发助手。本项目是 Windows 10 本地运行的 Python 工具，辅助生成 CANN Ascend C 算子代码，
但不直接编译/运行任何 C++ 代码。环境约束：Windows 10 版本 19041.208；嵌入式 Python 3.14.5 实际位于
P:\Software\python-3.14.5-embed-amd64，快捷脚本 P:\python.bat(CMD) 与 P:\python.ps1(PowerShell)；项目根目录下用 .penv 作为
虚拟环境(标准方式创建激活)；本地无任何 C/C++ 编译器，不安装 CANN 套件/NPU 驱动，编译运行验证均在云端 CANN Lab 完成。
参考文档与代码：官方教程文档位于 .\Documentation_for_Developers\（Jupyter Notebook），重点子目录
ascendc_operator_development 与 ascendc_operator_development_light，其余作为知识储备；官方示例代码位于
.\CANN_Learning_Hub_(for_dev)\，应优先参考其实现模式。写任何代码之前必须先阅读相关文档，确保符合 CANN 官方编程范式
与最佳实践。项目根目录为 .\CANN_OpHelper。核心功能：帮助快速生成 CANN 算子工程的模板代码，减少重复劳动。
工作策略：不重复造轮子，利用官方 msopgen 工具在云端生成初始框架，本地工具在已有框架上做模板填充与代码修改。
具体：工具根据算子描述(数学表达式/数据类型/形状等，CLI 参数或 YAML)生成完整 msopgen 命令，用户复制到云端执行获得初始工程；
之后工具读取工程，按需求用模板引擎(如 Jinja2)填充 Compute 核心逻辑、优化 Tiling 代码、添加调试语句，最终输出修改后的
完整工程供上传云端验证。技术栈与风格：Python 3.11+(兼容嵌入式版本)；依赖库包括但不限于 Jinja2、PyYAML、rich、typer、
pathlib，具体版本由 AI 按兼容性自选；CLI 必须用 typer，配合 rich 实现富文本输出(表格/面板/进度条等)；所有文件路径操作
必须用 pathlib 确保 Windows 兼容；项目结构由 AI 按职责设计(CLI 入口/模板引擎/策略库/工具函数等)但不预设立文件名，允许
开发中调整。工作流：①CLI 收集算子信息(参数或 YAML) ②生成并输出 msopgen 命令，同时记录算子元信息 ③用户把 msopgen
生成的工程复制到本地指定位置 ④工具读取工程，按元信息与模板修改/替换 Kernel 侧计算逻辑与 Host 侧 Tiling 实现
⑤输出修改后的完整工程到指定目录供上传云端编译验证。功能边界：不实现 C++ 语法/语义分析；不模拟 NPU 执行；不生成从零
开始的完整工程(依赖 msopgen 生成基础框架)；可选简单文本模式检查(如必要头文件是否存在、计算 API 名称是否正确等，仅基于
文本匹配不涉及编译)。开发阶段参考：优先实现逐元素(Element-Wise)算子模板(Add/Mul/Sigmoid 等)；Tiling 先实现核间均分/
核内均分基本模式，后续扩展；模板引擎支持变量替换与简单逻辑，从官方代码提取固定模式作为模板基础。长期要求：始终优先参考
本地文档与示例代码；生成代码必须符合 CANN 官方编程规范；工具保持轻量、无外部不必要依赖，完全在 Windows 本地运行。
