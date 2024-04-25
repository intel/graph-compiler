cmake_minimum_required(VERSION 3.11)
project(graph_compiler)

include(FetchContent)

option(GC_TEST "run tests" ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

# oneDNN
set(ONEDNN_CMAKE_ARGS
        -DDNNL_IS_MAIN_PROJECT=FALSE
        -DDNNL_BUILD_TESTS=FALSE
        -DDNNL_BUILD_EXAMPLES=FALSE
)
if (DEFINED ONEDNN_SRC_DIR)
    FetchContent_Declare(
            oneDNN
            SOURCE_DIR ${ONEDNN_SRC_DIR}
            CMAKE_ARGS ${ONEDNN_CMAKE_ARGS}
    )
else ()
    set(ONEDNN_VERSION "main" CACHE STRING "oneDNN version")
    if (${ONEDNN_VERSION} STREQUAL "main")
        set(ONEDNN_TAG ${ONEDNN_VERSION})
    else ()
        set(ONEDNN_TAG v${ONEDNN_VERSION})
    endif ()
    message(STATUS "oneDNN version: ${ONEDNN_VERSION}")
    FetchContent_Declare(
            oneDNN
            # GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
            GIT_REPOSITORY https://github.com/intel-ai/oneDNN.git
            GIT_TAG ${ONEDNN_TAG}
            CMAKE_ARGS ${ONEDNN_CMAKE_ARGS}
    )
endif ()
FetchContent_Populate(oneDNN)
FetchContent_GetProperties(oneDNN)
message(STATUS "oneDNN source dir: ${onednn_SOURCE_DIR}")
execute_process(COMMAND ${CMAKE_COMMAND}
        -S ${onednn_SOURCE_DIR}
        -B ${onednn_BINARY_DIR}
        ${ONEDNN_CMAKE_ARGS}
)
# end of oneDNN

include_directories(
        include
        ${onednn_BINARY_DIR}/include
        ${onednn_SOURCE_DIR}/include
        ${onednn_SOURCE_DIR}/src/graph/backend/elyzor/include
)

add_library(graph_compiler SHARED
        src/dnnl_graph_compiler.cpp
)

#target_link_libraries(graph_compiler PRIVATE dnnl)

add_subdirectory(test)
