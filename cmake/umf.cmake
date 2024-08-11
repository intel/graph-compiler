include_guard()

get_property(UMF_INCLUDES GLOBAL PROPERTY UMF_INCLUDES)
if (NOT DEFINED UMF_INCLUDES)
        gc_fetch_content(umf main https://github.com/oneapi-src/unified-memory-framework
                SET UMF_BUILD_LEVEL_ZERO_PROVIDER=OFF UMF_BUILD_SHARED_LIBRARY=ON UMF_BUILD_TESTS=OFF UMF_BUILD_EXAMPLES=OFF CMAKE_PREFIX_PATH=/home/zhangyan/graph_compiler_v2/hwloc/install
        )

        message(STATUS "including UMF")

        set(UMF_INCLUDES
                ${umf_BINARY_DIR}/include
                ${umf_SOURCE_DIR}/include
                ${umf_SOURCE_DIR}/src
        )
        set_property(GLOBAL PROPERTY UMF_INCLUDES ${UMF_INCLUDES})
endif ()