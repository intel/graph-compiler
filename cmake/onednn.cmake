include_guard()
include(${CMAKE_CURRENT_LIST_DIR}/common.cmake)

set(DNNL_CMAKE_ARGS
        -DDNNL_IS_MAIN_PROJECT=FALSE
        -DDNNL_BUILD_TESTS=FALSE
        -DDNNL_BUILD_EXAMPLES=FALSE
)
DeclareContent(dnnl main https://github.com/intel-ai/oneDNN.git)
FetchContent_Populate(dnnl)
FetchContent_GetProperties(dnnl)
message(STATUS "dnnl source dir: ${dnnl_SOURCE_DIR}")
execute_process(COMMAND ${CMAKE_COMMAND}
        -Wno-dev
        -S ${dnnl_SOURCE_DIR}
        -B ${dnnl_BINARY_DIR}
        ${DNNL_CMAKE_ARGS}
)
