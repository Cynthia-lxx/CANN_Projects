# AddcmulCustomFp16 — P1 real-device verification (真机载体操作指引)

Carrier: `Y = A + B * D * 2.0`, FP16, flat length 416 = 26 x 32B blocks
(8 cores -> perCoreBlocks=3, tailBlockNum=2). 请按顺序在**云端 CANN Lab**
与**本机**之间交替执行并回报每一步结果。

## 已就绪的本地文件（本目录）

- `op_spec_addcmul_fp16.yaml`  -- 算子 spec（3 输入 1 输出、float16、shape hint [416]、expr）
- `addcmul_custom_proto.json`  -- msopgen 原型 JSON（gen-msopgen 从 spec 机械导出）
- `CLOUD_STEPS.md`            -- 本文件

## Step 1（云端）：msopgen 生成空壳

```bash
cd /mnt/workspace
mkdir -p projects/addcmul_custom_fp16 && cd projects/addcmul_custom_fp16
# 上传 addcmul_custom_proto.json 到当前目录后执行：
msopgen gen -i addcmul_custom_proto.json -c ai_core-ascend910b4 -lan cpp -out addcmul_custom
ls addcmul_custom   # 应含 op_host/ op_kernel/ framework/ test/ CMakePresets.json build.sh
```

## Step 2（本机）：把空壳放回实测区

把云端的 `addcmul_custom` 工程目录整体复制到本机本目录下，结构为：

```
p:\Dev\CANN_Learning_Workspace\Projects\out\AddcmulCustomFp16\shell\addcmul_custom\
```

（`shell\` 下放 msopgen 空壳的本地副本；之后 fill-op 只改其中三个文件。）

## Step 3（本机）：fill-op 写盘三文件 + verify 资产

```powershell
p:\Dev\CANN_Learning_Refs\penv\Scripts\python.exe -m cann_ophelper fill-op `
  p:\Dev\CANN_Learning_Workspace\Projects\out\AddcmulCustomFp16\op_spec_addcmul_fp16.yaml `
  p:\Dev\CANN_Learning_Workspace\Projects\out\AddcmulCustomFp16\shell\addcmul_custom
```

预期：两阶段表格（三文件覆盖表 + verify 资产表）后提示云端指引。
`shell\addcmul_custom\verify\` 应出现 input_A/B/D.bin（各 832B）、golden.bin、runner 等。

## Step 4（云端）：编译运行比对

把整个 `addcmul_custom`（含新增 verify/）上传回云端 `projects/addcmul_custom_fp16/` 覆盖后：

```bash
cd /mnt/workspace/projects/addcmul_custom_fp16/addcmul_custom
bash verify/run_verify.sh
```

预期结尾：`TEST PASSED!`。若 `build.sh` 需要联网/代理环境变量请沿用 AddCustomTemplate 的既有做法。

## 判读

- 若 TEST PASSED：P1 dtype + 尾块在真机成立（fp16 大/小核 tiling + golden 自洽）。
- 若编译错/数值 diff：把完整输出贴回给开发助手。
