#include "kernel_operator.h"
#include "asc_try_tiling.h"

constexpr int32_t BUFFER_NUM = 2;   // 每队列缓冲数（双缓冲）
constexpr int32_t QUEUE_DEPTH = 2;

template <typename T>
class KernelAscTry {
public:
    __aicore__ inline KernelAscTry() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum, uint32_t finalSmallTileNum,
                                uint32_t tileDataNum, uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum, uint32_t tailBlockNum)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * blockIdx;
        this->tileDataNum = tileDataNum;
        if (blockIdx < tailBlockNum) {          // 大核：数据量多 32B
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {                                 // 小核：常规核
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            // 前面都是大核，GM 偏移需扣去大小核之差
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (blockIdx - tailBlockNum);
        }
        aGm.SetGlobalBuffer((__gm__ T *)a + globalBufferIndex, this->coreDataNum);
        bGm.SetGlobalBuffer((__gm__ T *)b + globalBufferIndex, this->coreDataNum);
        cGm.SetGlobalBuffer((__gm__ T *)c + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueA, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, this->tileDataNum * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < static_cast<int32_t>(this->tileNum); i++) {
            if (i == static_cast<int32_t>(this->tileNum) - 1) {
                this->processDataNum = this->tailDataNum;   // 最后一批为尾块
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<T> aLocal = inQueueA.AllocTensor<T>();
        AscendC::LocalTensor<T> bLocal = inQueueB.AllocTensor<T>();
        AscendC::DataCopy(aLocal, aGm[progress * this->tileDataNum], this->processDataNum);
        AscendC::DataCopy(bLocal, bGm[progress * this->tileDataNum], this->processDataNum);
        inQueueA.EnQue(aLocal);
        inQueueB.EnQue(bLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<T> aLocal = inQueueA.DeQue<T>();
        AscendC::LocalTensor<T> bLocal = inQueueB.DeQue<T>();
        AscendC::LocalTensor<T> cLocal = outQueueC.AllocTensor<T>();
        AscendC::Add(cLocal, aLocal, bLocal, this->processDataNum);
        outQueueC.EnQue<T>(cLocal);
        inQueueA.FreeTensor(aLocal);
        inQueueB.FreeTensor(bLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<T> cLocal = outQueueC.DeQue<T>();
        AscendC::DataCopy(cGm[progress * this->tileDataNum], cLocal, this->processDataNum);
        outQueueC.FreeTensor(cLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_DEPTH> inQueueA, inQueueB;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUEUE_DEPTH> outQueueC;
    AscendC::GlobalTensor<T> aGm;
    AscendC::GlobalTensor<T> bGm;
    AscendC::GlobalTensor<T> cGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void asc_try(GM_ADDR A, GM_ADDR B, GM_ADDR C,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AscTryTilingData);
    GET_TILING_DATA_WITH_STRUCT(AscTryTilingData, tiling_data, tiling);
    KernelAscTry<float> op;
    op.Init(A, B, C,
            tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum, tiling_data.finalSmallTileNum,
            tiling_data.tileDataNum, tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum, tiling_data.tailBlockNum);
    op.Process();
}
