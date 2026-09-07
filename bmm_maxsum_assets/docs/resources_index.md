# 资源引用索引（BatchMatmulMaxSum 项目）

> P0 产出之一：供 P2–P5 检索的**已核实资源清单**（路径逐一核实存在，2026-09-05）。
> 约定：路径用绝对路径；标 `只读` 的区不可写入；本文件仅登记事实，随使用动态增补。

---

## 1. 题目与项目资产（本资产区 / 记忆区）

| 资源 | 路径 | 说明 |
|---|---|---|
| 题目唯一事实源（一字不差存档） | `p:\Dev\CANN_Learning_Refs\.codebuddy\memory\stage_goal_batchmatmul_maxsum.md` | 一切语义裁定以此为准（只读引用） |
| 父计划（P0–P5 路线） | `p:\Dev\CANN_Learning_Refs\.codebuddy\plans\PARENT_PLAN_batchmatmulmaxsum.md` | 战略级；P0 完成后回填状态 |
| 记忆（长期/每日） | `p:\Dev\CANN_Learning_Refs\.codebuddy\memory\MEMORY.md`、`2026-09-05.md` 等 | 只读+按记忆纪律增补 |
| 题目实测工程（P2 载体） | `p:\Dev\CANN_Learning_Workspace\Projects\batchmatmulmaxsum_problem_1746_template\` | 含 `kernel.asc`/`main.asc`/`run.sh`/`scripts\{BatchMatmulMaxSum.py,gen_data.py,verify_result.py}`/`data_utils.h`/`CMakeLists.txt` |
| 姊妹题 Addcmul（P1 载体） | `p:\Dev\CANN_Learning_Workspace\Projects\addcmulkernel_problem_332_template\` | 结构同上，scripts\ 含 `addcmul.py` |
| 相邻区学习计划 | `p:\Dev\CANN_Learning_Workspace\.codebuddy\plans\核心课程通读与AscendC知识沉淀_8339da86.md` | 可衔接的课程学习计划 |
| OpHelper 仓库（工具扩展 P1+ 作用点） | `p:\Dev\CANN_Learning_Refs\CANN_OpHelper\` | 独立仓库；本资产区不改其文件 |

## 2. 官方教程（四份可读副本，内容同构）

教程相对结构（以 `04_matmul_basic` / `05_fused_operator_development` 为主研对象）：
`02_AscendC_basic`、`03_intermediate_vector_operator_development`、`04_matmul_basic`、`05_fused_operator_development`、`08_performance_optimization`；每章含 `*.ipynb` 讲解与 `answer\` 源码。

| # | 根路径 | 备注 |
|---|---|---|
| 1 | `p:\Dev\CANN_Learning_Refs\Documentation_for_Developers\ascendc_operator_development\` | 本工作区文档副本（主引源） |
| 2 | `p:\Dev\CANN_Learning_Refs\CANN_Learning_Hub_(for_dev)\tutorials\ascendc_operator_development\` | 本工作区内副本 |
| 3 | `p:\Dev\CANN_Learning_Hub\tutorials\ascendc_operator_development\` | p:\Dev 并排根（含 skills） |
| 4 | `p:\Dev\CANN_Learning_Workspace\Reference_And_Documentation\tutorials\ascendc_operator_development\` | 实测区旁副本 |

> 注：早期文档以 `CANN_Learning_Hub_(for_dev)` 指代镜像；磁盘上 `p:\Dev\CANN_Learning_Hub`（无后缀）与 `p:\Dev\CANN_Learning_Refs\CANN_Learning_Hub_(for_dev)` 均存在、内容同构。
> 轻量版：`p:\Dev\CANN_Learning_Hub\tutorials\ascendc_operator_development_light\`（01–04 章，03 章为 `.asc` 形态的高阶 Matmul/CV 融合练习）。

### 2.1 与 BMM 最相关的 answer 源码（以根 1 为例，其余镜像同构替换）

| 源码 | 绝对路径（根 1 前缀） | 研读状态 |
|---|---|---|
| 04.03 MatMul host（fp16 高阶 API+bias） | `...\04_matmul_basic\answer\04.03_answer\op_host\matmul_custom.cpp` | 已读（P0） |
| 04.03 MatMul kernel | `...\04_matmul_basic\answer\04.03_answer\op_kernel\matmul_custom.cpp` | 已读（P0） |
| 04.04 MatMul host（int8 变体） | `...\04_matmul_basic\answer\04.04_answer\op_host\matmul_custom.cpp` | 已读（P0） |
| 04.04 MatMul kernel | `...\04_matmul_basic\answer\04.04_answer\op_kernel\matmul_custom.cpp` | 已读（P0） |
| 05.04 CV 融合 MatmulAbs host | `...\05_fused_operator_development\answer\05.04_answer\op_host\matmul_abs.cpp` | 已读（P0） |
| 05.04 CV 融合 MatmulAbs kernel | `...\05_fused_operator_development\answer\05.04_answer\op_kernel\matmul_abs.cpp` | 已读（P0） |
| 05.03 VV 融合 square_diff | `...\05_fused_operator_development\answer\05.03_answer\` | 待读（P2 前） |
| 05.05 matmul_sinh 变体 | `...\05_fused_operator_development\answer\05.05_answer\` | 待读（P2 前） |

章节讲解 ipynb：`04_matmul_basic\04.0X_*.ipynb`、`05_fused_operator_development\05.04_cv_fused_operator_development.ipynb` 等（P2 前精读）。

## 3. skills（`p:\Dev\CANN_Learning_Hub\skills\`，只读）

| skill | 定义文件 | 内容结构 |
|---|---|---|
| ascendc-ops-project | `skills\ascendc-ops-project\SKILL.md` | references（api_best_practices / cmake_guide / precision_standard / tiling_design）、templates（host/kernel/op_host/op_kernel/build 等）、examples（add/clamp） |
| cannjudge-submit | `skills\cannjudge-submit\SKILL.md` | README、cannjudge_cli.py、encrypt_password.py、generate_key.py、example.py |

## 4. AscendC 知识沉淀（记忆区，只读引用）

`p:\Dev\CANN_Learning_Refs\.codebuddy\memory\` 下：
`ascendc_01_programming_paradigm.md`、`ascendc_02_operator_templates.md`、`ascendc_03_tiling_strategy.md`、`ascendc_04_host_device_collaboration.md`、`ascendc_05_api_reference.md`、`ascendc_06_debug_common_errors.md`（含 161001/soc 类问题）、`ascendc_07_custom_op_project_practice.md`、`ascendc_knowledge_index.md`（总索引）。
> 相邻实测工作区 `.codebuddy\memory\` 亦有一份同结构副本（实测经验只读源）。

## 5. 本资产区的产物（自引用）

| 文件 | 说明 |
|---|---|
| `README.md` | 目录说明/使用方式 |
| `docs\semantic_baseline.md` | 语义基线（公式/torch 映射/四布局/判据/三示例推导） |
| `docs\cube_datapath_notes.md` | Cube 数据流与官方范式研读笔记（含待验证清单） |
| `cases\case_matrix.md` / `case_matrix.json` | 用例矩阵（23 用例，人读 + 机器读） |
| `golden\golden_fp64_stdlib.py` | 纯标准库 FP64 golden（本地可跑） |
| `golden\golden_torch_ref.py` | torch 对拍脚本（需 torch 环境） |
| `golden\smoke_outputs\results.json` | 锚点冒烟结果（PASS） |
| `golden\smoke_outputs\results_local_all.json` | 全部 local 用例结果（20 用例 PASS，含 layout 等价断言） |

## 6. 追加说明（P0 执行中发现）

- 题目模板 scripts 提供 `BatchMatmulMaxSum.py`（PyTorch 参考）与 `verify_result.py`，P2 接通工程时应先对照其实现与本资产 golden 的一致性。
- 官方教程三镜像内容一致（均为完整拷贝含 ipynb + answer cpp），差别仅在位置；故研读统一以「根 1」为主，代码引用勿混用不同根造成 diff 噪音。
