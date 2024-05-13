cmake_minimum_required(VERSION 3.20)
project(GraphCompiler VERSION "0.1.0" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(GC_LEGACY_ENABLE ON)
option(GC_TEST_ENABLE "Build the tests" ON)

if(GC_LEGACY_ENABLE)
    add_subdirectory(legacy/core)
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
        ${PROJECT_BINARY_DIR}/include
        ${PROJECT_SOURCE_DIR}/include
)

# The paths are added in the subfolders using the gc_add_path() function.
# These lists are also used by tests.
set(GC_LIB_SOURCES CACHE INTERNAL "The graph_compiler library source paths")
set(GC_LIB_INCLUDES CACHE INTERNAL "The graph_compiler library include paths")

add_definitions(${LLVM_DEFINITIONS})

set(GC_MLIR_CXX_FLAGS "")
# Silence a false positive GCC -Wunused-but-set-parameter warning in constexpr
# cases, by marking SelectedCase as used. See
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85827 for details. The issue is
# fixed in GCC 10.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
  check_cxx_compiler_flag("-Wno-unused-but-set-parameter" CXX_SUPPORTS_WNO_UNUSED_BUT_SET_PARAMETER)
  append_if(CXX_SUPPORTS_WNO_UNUSED_BUT_SET_PARAMETER "-Wno-unused-but-set-parameter" GC_MLIR_CXX_FLAGS)
endif()
include("cmake/version.cmake")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(src)

add_library(graph_compiler SHARED ${GC_LIB_SOURCES})
target_include_directories(graph_compiler PUBLIC ${GC_LIB_INCLUDES})
set(GC_LIB_LINKED_LIBS
        MLIRLinalgx
        MLIRMicrokernel
        MLIROnednnGraph
)
target_link_libraries(graph_compiler PRIVATE ${GC_LIB_LINKED_LIBS})

add_subdirectory(unittests)
add_subdirectory(test)
