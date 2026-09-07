cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(cube_probe LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 14)

if(DEFINED NPU_ARCH)
    set(SOC_ARCH ${NPU_ARCH})
else()
    set(SOC_ARCH "dav-2201")
endif()

add_executable(cube_probe cube_probe.asc)

target_link_libraries(cube_probe PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(cube_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(cube_probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SOC_ARCH}>
)
