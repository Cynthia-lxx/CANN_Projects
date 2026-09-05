/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */


#ifndef ASC_TRY_TILING_H
#define ASC_TRY_TILING_H
#include <cstdint>

struct AscTryTilingData {
    uint32_t smallCoreDataNum;   // 小核(常规核)总元素数
    uint32_t bigCoreDataNum;     // 大核(多32B)总元素数
    uint32_t finalBigTileNum;    // 大核循环批次数
    uint32_t finalSmallTileNum;  // 小核循环批次数
    uint32_t tileDataNum;        // 单批标准元素数(UB预算决定)
    uint32_t smallTailDataNum;   // 小核尾批元素数
    uint32_t bigTailDataNum;     // 大核尾批元素数
    uint32_t tailBlockNum;       // 大核个数(=32B块总数对核数取余)
};

#endif // ASC_TRY_TILING_H
