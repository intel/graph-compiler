include_guard()

get_property(UMF_INCLUDES GLOBAL PROPERTY UMF_INCLUDES)

if (NOT DEFINED UMF_INCLUDES)
        include(functions)
        set(HW_PATH "/home/zhangyan/graph_compiler_v2/hwloc/install/")
        set(CMAKE_PREFIX_PATH "/home/zhangyan/graph_compiler_v2/hwloc/install/")
        set(UMF_BUILD_LEVEL_ZERO_PROVIDER OFF)
        set(UMF_DISABLE_HWLOC OFF)
        set(UMF_BUILD_SHARED_LIBRARY ON)
        set(UMF_BUILD_TESTS OFF)
        set(UMF_BUILD_EXAMPLES OFF)
        gc_fetch_content(umf main https://github.com/oneapi-src/unified-memory-framework
            CMAKE_ARGS "-DCMAKE_PREFIX_PATH=${HW_PATH};-DUMF_BUILD_LEVEL_ZERO_PROVIDER=OFF;-DUMF_DISABLE_HWLOC=OFF;-DUMF_BUILD_SHARED_LIBRARY=ON;-DUMF_BUILD_TESTS=OFF;-DUMF_BUILD_EXAMPLES=OFF"
        )
        message(STATUS "including UMF")
        set(UMF_INCLUDES
                ${umf_BINARY_DIR}/include
                ${umf_SOURCE_DIR}/include
                ${umf_SOURCE_DIR}/src

        )

        set_property(GLOBAL PROPERTY UMF_INCLUDES ${UMF_INCLUDES})
endif ()