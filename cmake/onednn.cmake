include_guard()

set(ONEDNN_CMAKE_ARGS
        -DDNNL_IS_MAIN_PROJECT=FALSE
        -DDNNL_BUILD_TESTS=FALSE
        -DDNNL_BUILD_EXAMPLES=FALSE
)

if (${GC_ONEDNN_VERSION} STREQUAL "main")
    set(ONEDNN_TAG ${GC_ONEDNN_VERSION})
else ()
    set(ONEDNN_TAG v${GC_ONEDNN_VERSION})
endif ()
message(STATUS "oneDNN version: ${GC_ONEDNN_VERSION}")
set(FETCHCONTENT_QUIET FALSE)
FetchContent_Declare(
        oneDNN
        GIT_REPOSITORY https://github.com/intel-ai/oneDNN.git
        GIT_TAG ${ONEDNN_TAG}
        CMAKE_ARGS ${ONEDNN_CMAKE_ARGS}
        GIT_PROGRESS TRUE
)

FetchContent_Populate(oneDNN)
FetchContent_GetProperties(oneDNN)
message(STATUS "oneDNN source dir: ${onednn_SOURCE_DIR}")
FetchContent_MakeAvailable(oneDNN)