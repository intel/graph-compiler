cmake_minimum_required(VERSION 3.20)
project(GraphCompiler LANGUAGES C CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(GC_LEGACY_ENABLE ON)

set(GC_ONEDNN_VERSION "main" CACHE STRING "oneDNN version")
include(onednn)
add_subdirectory(${onednn_SOURCE_DIR} ${onednn_BINARY_DIR})

if(GC_LEGACY_ENABLE)
    add_subdirectory(legacy/core)
endif()
