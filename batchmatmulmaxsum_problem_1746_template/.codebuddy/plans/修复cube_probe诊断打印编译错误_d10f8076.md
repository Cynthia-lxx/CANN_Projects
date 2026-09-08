---
name: 修复cube_probe诊断打印编译错误
overview: 修复 cube_probe.asc 中诊断打印引用了 TCubeTiling 不存在的 K 字段导致的编译失败，推送后云端重跑拿到数值错位诊断数据。
todos:
  - id: fix-tiling-print
    content: 在 cube_probe.asc 第 253-257 行诊断 printf 中删除 K=%d 格式符与 tiling.K 实参，并静态核对无 tiling.K 残留
    status: completed
  - id: commit-push
    content: git add/commit/push 至 CANN_Projects 仓库 master（含 cube_probe.asc 修复）
    status: completed
    dependencies:
      - fix-tiling-print
  - id: cloud-rerun-verify
    content: 云端 git pull 后 bash run_probe.sh 复跑，确认编译通过并收集 tiling/象限/指纹诊断输出以定位错位根因
    status: completed
    dependencies:
      - commit-push
---

## 需求说明

cube_probe 诊断版探针在云端编译失败，报错：`no member named 'K' in 'AscendC::tiling::TCubeTiling'`（cube_probe.asc:254）。

根因已通过本地头库核实：`TCubeTiling`（asc\include\adv_api\kernel_tiling.h:305-356）真实字段为 `M/N/Ka/Kb/singleCoreM/singleCoreN/baseM/baseN` 等，**无 `K` 字段**。全文件 14 处 `tiling.` 引用中仅诊断 printf 第 253-254 行非法引用了 `tiling.K`，其余均为合法字段。

目标：修复该编译错误并推送，使云端能重新编译运行诊断版探针，输出 tiling 实际字段值、四象限 correct 统计与 18 格指纹，以定位 v2 Matmul 探针数值错位（当前 4063/4096 mismatch，row0 前 8 格正确）的真实根因。

## 技术方案

### 改动策略

最小单点修复：删除诊断 printf 中的 `K=%d` 格式符与对应实参 `static_cast<int>(tiling.K)`。host main 中 K 本就是局部常量（=64），无打印价值；Ka/Kb 打印足以验证 K 对齐放大情况。保留 M/N/Ka/Kb/baseM/baseN/singleCoreM/singleCoreN 打印（这正是排查 SetFixSplit(32,32,-1) 是否真实生效的关键证据）。

### 精确改动（cube_probe.asc 第 253-257 行）

格式串：`"tiling M=%d N=%d K=%d Ka=%d Kb=%d baseM=%d baseN=%d singleCoreM=%d singleCoreN=%d\n"`
改为：`"tiling M=%d N=%d Ka=%d Kb=%d baseM=%d baseN=%d singleCoreM=%d singleCoreN=%d\n"`
实参行删去第 254 行中的 `, static_cast<int>(tiling.K)`，保持后续参数一一对应。

### 验证链

1. 本地静态检查：确认文件中不再有 `tiling.K` 残留引用。
2. git commit + push 至 GitHub（Cynthia-lxx/CANN_Projects master）。
3. 云端 `git pull && bash run_probe.sh` 复跑，预期编译通过，输出诊断信息。

### 风险控制

- 不改动任何 kernel 计算逻辑、tiling 构造与数据通路代码，仅修 printf，爆炸半径为零。
- 该修复是继续数值排障的前置条件，不引入新行为。