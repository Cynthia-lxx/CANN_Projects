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
#include "addcmul_custom_tiling.h"
constexpr int32_t BUFFER_NUM = 1;  // tensor num for each queue
constexpr int32_t QUEUE_DEPTH = 1;

class KernelAddcmulCustom {
public:
    __aicore__ inline KernelAddcmulCustom() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR d, GM_ADDR y, uint32_t bigDataNum, uint32_t smallDataNum, uint32_t tailBlockNum)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->bigDataNum = bigDataNum;
        this->smallDataNum = smallDataNum;
        this->tailBlockNum = tailBlockNum;
        bool isBigCore = blockIdx < this->tailBlockNum;
        this->dataNum = isBigCore ? this->bigDataNum : this->smallDataNum;
        uint32_t gmOffset = isBigCore ?
            blockIdx * this->bigDataNum :
            this->tailBlockNum * this->bigDataNum + (blockIdx - this->tailBlockNum) * this->smallDataNum;
        aGm.SetGlobalBuffer((__gm__ half *)a + gmOffset, this->dataNum);
        pipe.InitBuffer(inQueueA, BUFFER_NUM, this->dataNum * sizeof(half));
        bGm.SetGlobalBuffer((__gm__ half *)b + gmOffset, this->dataNum);
        pipe.InitBuffer(inQueueB, BUFFER_NUM, this->dataNum * sizeof(half));
        dGm.SetGlobalBuffer((__gm__ half *)d + gmOffset, this->dataNum);
        pipe.InitBuffer(inQueueD, BUFFER_NUM, this->dataNum * sizeof(half));
        yGm.SetGlobalBuffer((__gm__ half *)y + gmOffset, this->dataNum);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->dataNum * sizeof(half));
        pipe.InitBuffer(tmp0, this->dataNum * sizeof(half));
        pipe.InitBuffer(tmp1, this->dataNum * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> aLocal = inQueueA.AllocTensor<half>();
        AscendC::DataCopy(aLocal, aGm[0], this->dataNum);
        inQueueA.EnQue(aLocal);
        AscendC::LocalTensor<half> bLocal = inQueueB.AllocTensor<half>();
        AscendC::DataCopy(bLocal, bGm[0], this->dataNum);
        inQueueB.EnQue(bLocal);
        AscendC::LocalTensor<half> dLocal = inQueueD.AllocTensor<half>();
        AscendC::DataCopy(dLocal, dGm[0], this->dataNum);
        inQueueD.EnQue(dLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> aLocal = inQueueA.DeQue<half>();
        AscendC::LocalTensor<half> bLocal = inQueueB.DeQue<half>();
        AscendC::LocalTensor<half> dLocal = inQueueD.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        AscendC::LocalTensor<half> s0 = tmp0.Get<half>();
        AscendC::LocalTensor<half> s1 = tmp1.Get<half>();
        AscendC::Mul(s1, bLocal, dLocal, this->dataNum);
        AscendC::Duplicate<half>(s0, (half)2, this->dataNum);
        AscendC::Mul(s0, s1, s0, this->dataNum);
        AscendC::Add(yLocal, aLocal, s0, this->dataNum);
        outQueueY.EnQue(yLocal);
        inQueueA.FreeTensor(aLocal);
        inQueueB.FreeTensor(bLocal);
        inQueueD.FreeTensor(dLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        AscendC::DataCopy(yGm[0], yLocal, this->dataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueA;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueB;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueD;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueueY;
    AscendC::GlobalTensor<half> aGm;
    AscendC::GlobalTensor<half> bGm;
    AscendC::GlobalTensor<half> dGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp1;
    uint32_t dataNum;
    uint32_t bigDataNum;
    uint32_t smallDataNum;
    uint32_t tailBlockNum;
};


extern "C" __global__ __aicore__ void addcmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR d, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddcmulCustomTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulCustomTilingData, tiling_data, tiling);
    KernelAddcmulCustom op;
    op.Init(a, b, d, y, tiling_data.bigDataNum, tiling_data.smallDataNum, tiling_data.tailBlockNum);
    op.Process();
}
