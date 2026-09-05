#pragma once

#include <cstdint>
#include <vector>

#include "acl/acl.h"

using HostDataType = float;
// 建议用非 32B 对齐尺寸以覆盖"大核/小核 + 尾块"路径；
// 例如 (7,97)=679 元素；常规大 shape 可用 {8,16,64}
const std::vector<int64_t> CASE_SHAPE = {7, 97};
constexpr aclDataType CASE_ACL_DTYPE = aclDataType::ACL_FLOAT;
