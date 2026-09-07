/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


#ifndef ADDCMUL_CUSTOM_TILING_H
#define ADDCMUL_CUSTOM_TILING_H
#include <cstdint>

struct AddcmulCustomTilingData {
    uint32_t totalLength;
    uint32_t bigDataNum;
    uint32_t smallDataNum;
    uint32_t tailBlockNum;
};
#endif // ADDCMUL_CUSTOM_TILING_H
