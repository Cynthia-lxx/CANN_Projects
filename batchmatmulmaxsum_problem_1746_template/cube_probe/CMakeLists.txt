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
add_executable(off_matmul_abs off_matmul_abs.asc)
add_executable(copy_sem_probe copy_sem_probe.asc)
add_executable(shape_sweep shape_sweep.asc)
add_executable(cube_iterall cube_iterall.asc)
add_executable(cube_mc cube_mc.asc)
add_executable(v2_proto v2_proto.asc)

foreach(_tgt cube_probe off_matmul_abs copy_sem_probe shape_sweep cube_iterall cube_mc v2_proto)
    target_link_libraries(${_tgt} PRIVATE
        tiling_api
        register
        platform
        unified_dlog
        dl
        m
        graph_base
    )

    target_include_directories(${_tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_compile_options(${_tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SOC_ARCH}>
    )
endforeach()
