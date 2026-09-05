
#include "../op_kernel/asc_try_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 平台能力：可用核数、UB 容量
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNum();

    // 输入总元素数与类型长度 -> 总字节数
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint32_t inputLength = inputNum * typeLength;
    const uint32_t BLOCK_SIZE = 32;

    // 总字节数向上 32B 对齐后，按 32B 块在多核间均分；余数块分给前 tailBlockNum 个核（大核）
    uint32_t inputLengthAlign32 = ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    coreNum = std::min(coreNum, inputLengthAlign32 / BLOCK_SIZE);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    uint32_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;
    context->SetBlockDim(coreNum);

    // 核内切批：A/B/C 三路数据同驻 UB（ubDataNumber=3），按双缓冲(BUFFER_NUM=2)核算单批元素数
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t BUFFER_NUM = 2;
    const uint32_t ubDataNumber = 3;
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;

    // 小核（常规核）切分参数
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - tileDataNum * smallTileNum;
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    // 大核数据量比小核多 32B，切分参数同理
    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    AscTryTilingData *tiling = context->GetTilingData<AscTryTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class AscTry : public OpDef {
public:
    explicit AscTry(const char* name) : OpDef(name)
    {
        this->Input("A")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("B")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("C")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(AscTry);
}
