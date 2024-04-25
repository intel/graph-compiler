include_guard()
include(${CMAKE_CURRENT_LIST_DIR}/common.cmake)

set(SIMDJSON_CMAKE_ARGS
        -DSIMDJSON_BUILD_STATIC_LIB=ON
)
DeclareContent(simdjson v3.9.1 https://github.com/simdjson/simdjson.git)
FetchContent_MakeAvailable(simdjson)
