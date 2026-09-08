# .codebuddy 项目记忆 — batch_matmul_max_sum problem_1746 交接首屏（2026-09-08）

> 本文档为**工程目录内嵌记忆**，随 git 仓库交付。接手 Agent 请先读本文件，再按「阅读顺序」进入历史细节。
> 源记忆（工作区级 `.codebuddy/`）仍在原工作区，本项目内 `.codebuddy/` 是聚焦 problem_1746 的副本 + 补充。

## 一、目录地图

| 路径 | 内容 |
|---|---|
| `memory/global_memory_export_2026-09-08.md` | AI 全局记忆两条原文落盘（攻坚现状 + OpHelper 相邻项目） |
| `memory/handoff_batchmatmulmaxsum_judge_2026-09-06.md` | **跨日主文档**：版本演进史、判题平台规则、硬经验、提交前清单 |
| `memory/MEMORY.md` | 判题平台提交规则（§判题平台提交规则）：入口=仅 `kernel.asc`、禁 API 清单、**每 iteration 恰 1 launch（Expected 75）**、E75/G20 推断 |
| `memory/stage_goal_batchmatmul_maxsum.md` | 阶段目标/验收基准 |
| `memory/2026-09-08.md` | **最新日志**：F 系列取证收口、v2 分相移植、转置 host 归一化修复、**20 例 case_matrix 云验全绿（§12.8）** |
| `memory/2026-09-07.md` | 判题平台规则实测（E75/G20/profiling）、probe-01、编译坑（`__gm__` cast） |
| `memory/2026-09-06.md` | 前期探针史（A/B/D 探针、GetTensorC 竞态初证） |
| `plans/PLAN_1746_*.md` 等 | 各阶段计划与判读（F 系列/架构/路线图） |

## 二、阅读顺序（接手必读）

1. 本文件（状态快照）
2. `memory/MEMORY.md`（判题规则红线）
3. `memory/handoff_batchmatmulmaxsum_judge_2026-09-06.md`（规则实证史）
4. `memory/2026-09-08.md`（全部最新证据与结论）
5. 有探针细节需求再开 `plans/PLAN_1746_probe_F_matmul_multitile_2026-09-08.md`

## 三、当前状态快照（2026-09-08 收盘）

- **git**：commit `09fde2d` = 转置 host 归一化修复（已 push 云端并 pull 验证）；此前 `788e0db` = v2 分相移植进 kernel.asc。
- **kernel.asc 结构**：`run_kernel`（judge 私有 host driver 直接调用，GM_ADDR 签名）= shape 解析 → 若 transpose 任一为真则 host 归一化
  物理→逻辑（D2H + `p1746_norm_x1/x2` 重排 + H2D，失败静默 return）→ fp16 && M,N≥128 && tiling 通过 && alloc 成功 = cube 分相路径
  （逐 batch：`v2_cube_batch<<<1>>>` 整写 GM C + `v2_reduce_batch<<<1>>>` 归约 y[b]，同 stream）→ 否则 v1 单核 vector 兜底（恰 1 launch）。
- **云验结果（全绿）**：run.sh 10 例回归 ALL PASS；case_matrix local **20/20 PASS**（c200-219，脚本 gen_data_casematrix.py），
  其中 c201 ex2_anchor_tt 双转置 0 diff、c203-210 layout_equiv fp16+bf16 × ff/ft/tf/tt 全 0 diff（正对旧判题点 2-9 错因族）、
  tail/全负/规模例全过；c218 m8192_bf16 绝对 diff 0.0078 但相对 ≈3e-7（正常）。二进制在 `build/`，跑 `./build/batch_matmul_max_sum_custom`。
- **编译 warning 已判读无害**：`cce_global ignored`（GM_ADDR cast，main.asc 亦有同款）、`v2_reduce_batch not marked`（既有）。
- **唯一未提交物**：`.vscode/`（本机 IDE 配置，刻意不入库）。

## 四、遗漏细节补齐（本次转移时核对补入，源记忆个别处无此粒度）

1. **判题编译链风险已被 12.6 实测排除**：v2 判题（commit 788e0db，含 matmul/tiling host 代码 + GM cast + cube 核）在判题环境
   编译通过并产出数值结果 → judge host 具备 kernel.asc 所需全部头/链接/宏环境。后续改 kernel.asc 不必再担心「判题侧缺 matmul 头」。
2. **判题调用语义**：judge driver 每 iteration 调 `run_kernel` 一次。本地 main.asc 只是样例，多 launch 自由**不代表判题侧合规**。
   12.6 判题 2/15 的关键事实：点 2-9（小 shape，µs 级）能跑出 Wrong Answer → **当时多 launch 规则未拦小 shape v1 路径**；
   大 shape/多 launch 是否触发 profiling 规则仍未知（点 11-15 Skipped 成因待查，可能是 profiling/超时/平台侧）。
3. **判题模型假设**（转置修复所依赖）：判题按**逻辑 shape + transposeX1/X2 标志**下发（x1 逻辑 (B,M,K)、x2 逻辑 (B,K,N)；tx 只改物理
   storage 布局为 (B,K,M)/(B,N,K)）。此模型已与 `bmm_maxsum_assets/docs/semantic_baseline.md` 对齐并被本地 20 例强验证。
   若判题实为「物理 shape 下发且无 tx 标志」，仍需改解析——此为残余不确定性。
4. **host 归一化在判题 harness 的依赖面**：run_kernel 转置路径调用 `aclrtMallocHost/aclrtMalloc/aclrtMemcpy(D2H/H2D)/aclrtFree`。
   本地 main.asc 已含 acl init/context，故本地可跑；judge driver 同样持 stream 调 run_kernel，判为低风险但**未经判题实证**。
   失败分支是静默 return（不写 y），不会崩溃。
5. **规模挡板（赛方已核实）**：B∈[1,64]；M/N∈[1,8192]；K∈[32,8192] 且 K%8==0；B×M×K≤2^28 且 B×N×K≤2^28；
   最坏 MACs≈2^41。**v2 cube 门控仅 fp16 且 M,N≥128；bf16/小 shape 永远 v1 单核 vector → 大 bf16 或巨型 fp16 仍可能 TLE**
   （旧点 10 TLE 与此有关）。多核提速拼写（F6：仅 M 向 slab + off1 CalcOffset，tailN 禁多核）已备好但未启用。
6. **未覆盖域**：case_matrix cloud 专属组 s28a/s28b（大 bf16 等）未本地生成；gen_data_casematrix.py 头部注释「21 例」应为「20 例」
   （3 anchors 已含在 20 内），仅注释叙述问题。
7. **云端操作模板**：`cd /mnt/workspace/CANN_Projects/batchmatmulmaxsum_problem_1746_template && git pull && bash run.sh 2>&1 | tee run_xxx.txt`；
   CANN 在 `$HOME/Ascend/cann-9.0.0`；push 走 gh-proxy；本地 PowerShell 用 `git -C` 绝对路径执行。
8. **协作提醒**：相邻 CANN_OpHelper 工具可能把 Projects/ 当试验场地改文件，异动先 diff 再处理。

## 五、下一步动作（按序）

1. 用户重提判题（提交内容 = 现 kernel.asc 全文）。预期两态：
   - 点 2-9 由 Wrong Answer → PASS → transpose 修复确认收口，转攻点 10 TLE / 11-15 Skipped；
   - 仍 Wrong → 判题 shape 可能非「逻辑 + tx」模型，转物理 shape 解析排查。
2. 待赛方口径开放项：① profiling 是否仍强制每 iteration 恰 1 launch；② 隐藏用例规模/时限。
3. 依口径决策：恰 1 launch 强约束 → F8 单 launch 研究；多 launch 合规 → 上 M 向多核提速解决大 shape 耗时。
4. 顺手修 gen_data_casematrix.py 注释（20 vs 21），非阻塞。
