---
name: cube_probe 探针 D：GetTensorC 行序指纹 + DataCopy gap 语义直测 + 官方范式非定值验证
overview: 基于探针 B 输出（UB 连续镜像 band 仅 row0/1 正确、960/1024 错误；官方基线因定值输入完全盲测）设计探针 D：三种手段并行取证，定位 fp32 GetTensorC 产物行序规律、DataCopy dstGap 语义、以及官方 baseN=128 范式在非定值输入下的行映射是否本来正确。
todos:
  - id: copy-sem-probe
    content: 新增 copy_sem_probe.asc：pattern tile 三参数 DataCopy gap 直测 target，接入 CMakeLists 与 run_probe.sh
    status: completed
  - id: off-nc-host
    content: off_matmul_abs.asc host 非定值化：覆盖区交集(512x320)容差比较、行身份列0抽样、numBlocks 按 tiling 核对
    status: completed
  - id: mirror-fingerprint-host
    content: cube_probe.asc host 增镜像行身份 argmax 与抽样行原始值打印，精简旧 dump，kernel 不动
    status: completed
  - id: commit-push
    content: 静态核对（无越界/无 kernel 改动/三 target 齐全）后 git commit 并 push 供云端复跑
    status: completed
    dependencies:
      - copy-sem-probe
      - off-nc-host
      - mirror-fingerprint-host
---

## 需求概述

云端探针 D 已确认：DataCopy 的 dstGap/srcGap 语义正常（R1 行距=blockLen+gap、R2 跳源行），写回通路本身无责；异常集中在 **GetTensorC 的 UB 输出侧**。本轮"取证探针 E"目标是**一锤定音判型 GetTensorC 输出异常的两类独立病因**，不写回、不修复：

1. **跨引擎同步缺失假设**：cube_probe 的 kernel 在 GetTensorC 后直接 DataCopy（同侧引擎 RAW），无官方范式中 Abs 向量算子（V 引擎）充当的跨引擎屏障，导致 MTE3 抢读未写完的 UB（镜像=中间部分和 63/64、main=后续状态，解释"同源不同果"）。验证法：kernel 加 mode 参数，mode1 在 GetTensorC 后补一条官方同款 in-place Abs（正值数据不改变数值）再拷贝，逐行对比 golden。
2. **多 tile 迭代/落址错位假设**（官方 Abs 存在仍出错）：off_matmul_abs 每象限仅约一个 128×128 tile 正确，且 row0 列 128 竟装的是 C[0][512] 的值。验证法：host 侧 8×5 tile 网格正确计数 + row0 各 baseN 边界列的"内容到底属于哪个 golden 列"映射，判定 N-block 排列/迭代错位规律，为后续修复（偏移换算或逐 tile 校正）提供映射表。

预期产出：GetTensorC 后是否需要向量屏障的明确结论 + off 官方范式多 tile 落址规律的完整映射，据此决定下一轮写回修复方案。

## 边界
- 只改 cube_probe.asc（kernel 加 mode 分支 + host 双模式运行打印）与 off_matmul_abs.asc（仅 host 打印新增）；copy_sem_probe 与 run_probe.sh 结构不变。
- 不改变任何计算语义：mode1 屏障用 Abs（官方同款、本探针正值输入下数值不变）。
- 本地仅文本编辑 + lint + git commit/push；云端一次 run_probe.sh 取数。


## 技术方案

### 核心思路

**假设一验证（cube 单 tile 屏障对照）**：同一次运行内先后 launch 两次 kernel——mode0 完全复现当前异常基线（期望 rows0/1 对、rows≥2 = 63(r-1) 型部分和）；mode1 在 `GetTensorC<true>(cTileLocal,false,true)` 之后、EnQue 之前插入官方范式同款 `AscendC::Abs(cTileLocal, cTileLocal, baseM*baseN)`（V 引擎向量操作，正值输入数值不变，仅制造"MTE3→V→MTE3"跨引擎排序），随后 mirror/main 两条 DataCopy 不变。若 mode1 镜像行 0..31 全部命中 golden ⇒ 病因=缺跨引擎同步，生产修复=保留向量步骤即可。

**假设二取证（off 多 tile 网格映射）**：官方 kernel 不动，host 在既有回读数据上新增两类打印：(a) 以 128×128 为粒度统计 8(m-block)×5(n-block) 网格内逐元素命中 golden 计数，输出"哪些 tile 完全正确"；(b) row0 在 baseN 边界列 {0,128,256,384,512} 的 got 值 vs golden 值，并对每个 got 值反查 golden 列号，直接读出 N-block 错位/排列方向（例如已知 col128 处 got=C[0][512]，需确认是否系统性 +384 偏移）。两次输出配合即可判定错误 tile 的落址映射规律。

### 实施要点

1. **执行前先核头库（cann_ascendc_headers_9.0.0）**：
   - `GetTensorC<true>` 模板与 3 个实参（dst, false, true）的语义（尤其末参是否 needWait/等待 L0C），在 `ascendc/include/impl/` 平台实现（dav_c220 / __NPU_ARCH__==2201）里确认 GetTensorC 内部 L0C→UB 的同步点，验证"同引擎 RAW 无自动排序"推断是否成立；
   - `Abs(dst, src, count)` Level2 元素计数签名（对照官方 matmul_abs 行 146 用法即可）。
   - 结论（含出处）追加进 `.codebuddy` 记忆文档，标注"已与 cann_ascendc_headers_9.0.0 核对"。
2. **cube_probe.asc kernel 改动（最小侵入）**：
   - kernel 入口参数表末尾追加 `int32_t mode`（host 同步传 0/1）；
   - `MatmulCompute()` 内 GetTensorC 之后按 mode 条件插入 in-place Abs；Process/CopyOut 逻辑保持逐字不动（仍单 tile 门控 + mirror + main 两拷贝）；
   - 该改动仅影响执行顺序，不影响数据（正值）。文件头注释记录假设与出处。
3. **cube_probe host 双模式运行与打印**：
   - 对 mode∈{0,1}：memset 输出区 → launch → 回读 → 打印带 mode 标签的既有摘要（mismatches/quadrant/mirror row-correct 全 32 行/mirror raw rows 0,1,2,3,15,31 n0-7），新增每 mode 的"mirror 行 r 对 golden 行 r 精确命中计数"汇总（0..31）便于一眼对比；
   - 复用同一次 4096+1024 输出缓冲；注意两次 launch 之间 aclrtMemsetAsync 清零 + stream 同步。
4. **off_matmul_abs.asc host 仅新增打印**（kernel 零改动，数据/比较逻辑不动）：
   - 在既有 coverage/identity 打印后追加 8×5 tile 网格全对计数（每格 tot=16384）与 row0 的 baseN 边界列 got-vs-golden-argmax 映射表；
   - 输出量受控（约 40+5 行）。
5. **性能与风险**：
   - 两次 launch 顺序执行，无并发；唯一开销为第二次 kernel 的 Abs（1024 floats，可忽略）；
   - 缓冲仍为 1024 floats 队列 + GM 5120 floats，无越界；kernel 新增 int32_t 形参不影响 __mix__ 入口与 REGIST_MATMUL_OBJ；
   - 若 mode1 仍异常：证明非同步问题，转入"GetTensorC 结果本身只含部分 k 累加或行偏移"方向（配合头库 GetTensorC 平台实现层继续定位）；本计划已在该分支给出下一步方向。
6. **判定矩阵**：
   | 现象 | 结论 |
   | --- | --- |
   | mode0 复现 rows0/1 对、mode1 镜像 32 行全对 | 缺向量屏障；修复=在 GetTensorC 后保留向量步骤 |
   | mode1 仍仅 rows0/1 对 | GetTensorC 输出/迭代本身异常，转平台实现层研究 |
   | off 网格仅 (0,0) tile 全对、col128 装 C[0][512] | N-block 迭代/落址映射错位，据映射表设计偏移修正 |
   | off 网格多个 tile 对 | 官方范式成立（窄化到小 baseM/baseN 配置问题） |

### 目录结构

```
Projects/batchmatmulmaxsum_problem_1746_template/cube_probe/
├── cube_probe.asc        # [MODIFY] kernel 增 mode 参数 + Abs 屏障分支；host 双模式运行与逐行命中打印
├── off_matmul_abs.asc    # [MODIFY] host 仅新增 tile-grid 全对计数与 baseN 边界列映射打印；kernel 不动
├── copy_sem_probe.asc    # 不动
├── CMakeLists.txt        # 不动
└── run_probe.sh          # 不动
```

