/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


#include "register/op_def_registry.h"
#include "../op_kernel/addcmul_custom_tiling.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(8);
    AddcmulCustomTilingData *tiling = context->GetTilingData<AddcmulCustomTilingData>();
    constexpr uint32_t ELEMS_PER_BLOCK = 16;  // 32B / sizeof(half)
    uint32_t totalBlocks = totalLength / ELEMS_PER_BLOCK;
    uint32_t perCoreBlocks = totalBlocks / 8;
    uint32_t tailBlockNum = totalBlocks % 8;
    tiling->totalLength = totalLength;
    tiling->bigDataNum = (perCoreBlocks + (tailBlockNum > 0 ? 1 : 0)) * ELEMS_PER_BLOCK;
    tiling->smallDataNum = perCoreBlocks * ELEMS_PER_BLOCK;
    tiling->tailBlockNum = tailBlockNum;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AddcmulCustom : public OpDef {
public:
    explicit AddcmulCustom(const char *name) : OpDef(name)
    {
        this->Input("A")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("B")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("D")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->Output("Y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(AddcmulCustom);
}  // namespace ops
