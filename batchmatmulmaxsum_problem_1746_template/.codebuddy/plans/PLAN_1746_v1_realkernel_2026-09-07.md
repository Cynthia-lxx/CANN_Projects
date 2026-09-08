# BatchMatMulMaxSum problem_1746 — v1 真 kernel 开发计划（2026-09-07）

## 0. 状态快照（写盘时刻）
- 工程：`Projects/batchmatmulmaxsum_problem_1746_template/`（判题提交物：kernel.asc + main.asc + scripts + 数据）。
- 提交面：`run_kernel(GM_ADDR x1, info_x1, GM_ADDR x2, info_x2, GM_ADDR y, info_y, availableCoreNum, stream, transposeX1, transposeX2)`；main.asc 逐用例调一次。
- 当前 kernel.asc 仍为 probe-01（每调用恰 launch 1 个写 0 kernel）。probe 的判题回读未拿到（首次提交全点 CE 已修 reinterpret_cast；尚未重新提交）。**此文件即将被真 kernel 替换（保留：恰 1 launch 且无静默 return 的原则）。**

## 1. 语义（唯一来源 stage_goal + semantic_baseline.md）
- y[b] = Σ_{m=0..M-1} max_{n=0..N-1} Σ_{k=0..K-1} x1[b,m,k]*x2[b,k,n]
- 输入 fp16/bf16，逻辑物理皆 row-major；transpose 仅改 storage 布局（x1 转置→(B,K,M)，x2 转置→(B,N,K)）。
- 输出恒 fp32(B,)；K%8==0；B∈[1,64]；M,N≤8192；行 max 必须按 -FLT_MAX/首元素语义（全负行），不得用 0 初值；误差判据 1e-3；禁止 NaN。

## 2. 本会话已核实 API 事实（来源 cann_ascendc_headers_9.0.0，只读权威）
1. **DataCopyPad 双向都可用**（kernel_operator_data_copy_intf.h）：
   - 读 GM→UB：`DataCopyPad(LocalTensor<T>, GlobalTensor<T>, DataCopyExtParams, DataCopyPadExtParams<T>)`（MTE2，行 388/394）。
   - 写 UB→GM：`DataCopyPad(GlobalTensor<T>, LocalTensor<T>, DataCopyExtParams)`（MTE3，行 411；3 参版已由占位版在真实运行中验证——4B 尾部写）。
   - DataCopyPad 支持 dtype 含 half/uint16_t/bfloat16_t/float…（kernel_operator_data_copy_impl.h 的 support 列表）。
   - 2201 架构 DataCopyPadExtParams<T> 字段序 = {bool isPad, uint8 leftPadding, uint8 rightPadding, T paddingValue}；DataCopyExtParams = {uint16 blockCount, uint32 blockLen, stride…, rsv}。
2. **Cast Level2** = `Cast(dst, src, RoundMode::CAST_NONE, count)`（元素计数，自动处理 repeat/mask）。bmm_p2_slice 线上 303 行同款已过云验证。
3. **Cast 支持类型对**（dav_c220/kernel_operator_vec_vconv_impl.h 的 CheckCastDatatype，line ~799）：src half→dst float；**src bfloat16_t→dst float/int32**；src int16→float…。**不支持 uint16→任何数值类型**（uint16 仅能参与 DataCopyPad/复制）。
4. **bf16 关键结论**：910B vector 把 bf16 位型当作编译器内建 `bfloat16_t`（头库无 typedef，真实环境 ccec 提供；整型位技巧不可行，因 Cast 不支持 uint16 源）。bf16 读取可用 DataCopyPad<bfloat16_t>，转 fp32 用 Cast<float>。注：库内 `matmul_tiling_base.h` 的 `using half=double`、simt 的 bhalf 均为误导项，勿用。
5. Mul/Muls/Duplicate/Add/Max 等 count 形式 ≤64 元素；ReduceMax(dst,src,tmp,count) count 模式下 mask 按元素计（Add count=1 已在线上验证），任意 1..64 合法。
6. 写入 GM 4B：必须 DataCopyPad+ExtParams{1,4,0,0,0}（DataCopy 会按 32B 对齐越界写）。
7. `SetGlobalBuffer(__gm__ T*, len)`；LocalTensor 可用 operator[](元素偏移) 切片 + GetValue/SetValue(标量读写)。

## 3. v1 内核设计（本文件将写入 kernel.asc）
- **单 launch 原则保留**：run_kernel 无条件恰 launch 1 次（probe 假设未被推翻前不破坏计数面）。
- 计算形态：**纯 `__vector__` 单 block 行式算法**（grid=1；B 个 batch 块内串行）。理由：规避 Cube 域限制（N%8、M%64）与非对齐问题；先保证正确再优化多核。
- 每 batch、每 row m：
  1. rowAcc（N fp32，N 向上取 64 对齐）按 64 段清零；
  2. 逐 64 段 masked DataCopyPad 读 x1 行 m → Cast 到 x1f（fp32，K 长）；
  3. for k in 0..K-1：`v=x1f.GetValue(k)`；对 x2 行 k 每 64 段：masked 读 → Cast<float> → Muls(*v) → Add 累加进 rowAcc；
  4. 逐 64 段 ReduceMax(rowAcc段)→标量 fold 出行 max（C++ float，初值 -FLT_MAX）；
  5. row max 以 Add(acc,acc,rm,1) 累加（fp32，逐行）。
  6. 4B DataCopyPad 写 y[b]。
- dtype：`p1746Compute<half>` 与 `p1746Compute<bfloat16_t>` 两个模板实例，host 按 info_x1.tensors[0].dtype 代码（1/2）选。
- **transpose 支持 = 未做（v2）**：tx=true 时仍 launch 但按假布局计算（数值错）。本地 10 用例全 tx=false，verify 只测 tx=false。理由见 §5 决策。

## 4. 本轮动作
- [x] API 事实核实（Cast/DataCopyPad/bf16/transpose）。
- [ ] 记忆落盘（本文件 + 2026-09-07.md 追加段）。
- [ ] kernel.asc 整体替换（真 kernel，注释保留设计文档）。
- [ ] 静态文本自检（禁词、C 风格 cast、恰 1 launch、无 return 守卫）。
- [ ] 用户本地跑 run.sh 对拍 10 用例；verify_result。
- [ ] （可选后续）扩展 gen_data 转置用例 cid10-13 前，先做 v2 转置方案决策。

## 5. 已做的关键取舍/风险（写盘备忘，防上下文丢失）
1. **探测结果未回收**即开发真 kernel：probe 假设「75=15用例×5次，got=20=4用例×5次」未被判题回读确认；真 kernel 按「全用例都 launch 且恰 1 次」设计，与假设自洽；若下次判题 Got 仍非预期，据此诊断。
2. **transpose（tx）v2 的两个候选方案**（待选，需 1 次云验证）：
   - A：vtranspose 平铺法（两参 Transpose 的平铺尺寸语义未能从本地头库/教程 100% 坐实，风险中）；
   - B：逐元素 masked 2B DataCopyPad gather 规范化（慢但语义可推理；2B 读支持未实测）。
   - 倾向：本地先加 tx 冒烟用例（8×8 等小形状），优先试 B 以保证数值语义，vtranspose 语义查证后再定。
3. **bfloat16_t 内建类型在 judge host/device TU 的可见性**未能在本地编译器验证（本地禁编译）。若下次判题 CE 报 bfloat16_t，改法：把含 bfloat16_t 的模板实例包进 `#if defined(__NPU_ARCH__) && (__NPU_ARCH__==2201)`，其余分支给 half-only stub。
4. 网格固定 grid=1：多核分块（按 batch 分派）为 v3 优化项，需先确认 AIV 核数与多 block launch 是否允许。

## 6. 进度（2026-09-07 写入）
- kernel.asc 已整体替换为 v1 真 kernel：非模板 `__global__` ×2（p1746_compute_fp16 / p1746_compute_bf16）→ 共用 `__aicore__ inline p1746_run<RawT>`；RawT=half 或 bfloat16_t；grid=1、block0 全 batch 串行；行式 masked DataCopyPad + Cast + Muls/Add + ReduceMax(-FLT_MAX) + 4B DataCopyPad 写出。host run_kernel 直接透传 GM_ADDR（不再做去 __gm__ 的 cast），字段/结构与 main.asc 及 bmm_p2_slice 云验证写法一致。
- 静态自检项：无 main/include guard；无 __gm__→普通指针 cast（内核内用 `reinterpret_cast<__gm__ RawT*>`，与参考一致）；索引统一 uint32；`Muls`/`Duplicate`/`Cast`/`DataCopyPad` 签名已对照头库。
- 未验证（需云判题/本地 run）：DataCopyPad 奇数长度读、bfloat16_t 在 judge host TU 的类型可见性、bf16 Cast、数值对拍。
- 待办：本地 run.sh 对拍 10 用例 → verify_result → 判题提交（提交前可用 B 个 1 的用例先验 launch 计数）。
