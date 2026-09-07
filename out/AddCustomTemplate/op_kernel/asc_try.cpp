/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


#include "kernel_operator.h"
#include "asc_try_tiling.h"
constexpr int32_t BUFFER_NUM = 1;  // tensor num for each queue
constexpr int32_t QUEUE_DEPTH = 1;

class KernelAscTry {
public:
    __aicore__ inline KernelAscTry() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        aGm.SetGlobalBuffer((__gm__ float *)a + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueA, BUFFER_NUM, this->tileLength * sizeof(float));
        bGm.SetGlobalBuffer((__gm__ float *)b + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueB, BUFFER_NUM, this->tileLength * sizeof(float));
        cGm.SetGlobalBuffer((__gm__ float *)c + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(outQueueC, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmp0, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmp1, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
        AscendC::DataCopy(aLocal, aGm[progress * this->tileLength], this->tileLength);
        inQueueA.EnQue(aLocal);
        AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
        AscendC::DataCopy(bLocal, bGm[progress * this->tileLength], this->tileLength);
        inQueueB.EnQue(bLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> aLocal = inQueueA.DeQue<float>();
        AscendC::LocalTensor<float> bLocal = inQueueB.DeQue<float>();
        AscendC::LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();
        AscendC::LocalTensor<float> s0 = tmp0.Get<float>();
        AscendC::LocalTensor<float> s1 = tmp1.Get<float>();
        AscendC::Duplicate<float>(s0, (float)2, this->tileLength);
        AscendC::Sigmoid(s1, bLocal, this->tileLength);
        AscendC::Div(s0, s0, s1, this->tileLength);
        AscendC::Add(cLocal, aLocal, s0, this->tileLength);
        outQueueC.EnQue(cLocal);
        inQueueA.FreeTensor(aLocal);
        inQueueB.FreeTensor(bLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> cLocal = outQueueC.DeQue<float>();
        AscendC::DataCopy(cGm[progress * this->tileLength], cLocal, this->tileLength);
        outQueueC.FreeTensor(cLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueA;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueB;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueueC;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp1;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};


extern "C" __global__ __aicore__ void asc_try(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AscTryTilingData);
    GET_TILING_DATA_WITH_STRUCT(AscTryTilingData, tiling_data, tiling);
    KernelAscTry op;
    op.Init(a, b, c, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
