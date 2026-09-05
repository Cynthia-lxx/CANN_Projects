#include "kernel_operator.h"
#include "asc_try_tiling.h"

extern "C" __global__ __aicore__ void asc_try(GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AscTryTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // TODO: user kernel impl
}