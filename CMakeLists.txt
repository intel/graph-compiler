################################################################################
# Copyright (C) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# SPDX-License-Identifier: Apache-2.0
################################################################################

cmake_minimum_required(VERSION 3.20)
project(GraphCompiler VERSION "0.1.0" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(GC_LEGACY_ENABLE ON)
option(GC_TEST_ENABLE "Build the tests" ON)
option(GC_USE_GPU "Enable GPU backend" OFF)
option(GC_ENABLE_BINDINGS_PYTHON "Enable Graph Complier Python Binding" ON)
option(GC_DEV_LINK_LLVM_DYLIB "Link dynamic libraries of LLVM and MLIR. For developers only. Do not use it in packing the library." OFF)

if(GC_LEGACY_ENABLE)
    add_subdirectory(legacy/core)
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if(GC_USE_GPU)
  add_definitions(-DGC_USE_GPU=1)
  include(imex)
  if(GC_DEV_LINK_LLVM_DYLIB)
    message(WARN "GPU backend may not be compatible with dynamic linking to LLVM")
  endif()
endif()

if(GC_ENABLE_BINDINGS_PYTHON AND NOT MLIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Failed to enable Python API due to the 'MLIR_ENABLE_BINDINGS_PYTHON' for LLVM is not ON.")
  set(GC_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "" FORCE)
endif()

if(GC_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

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
if(GC_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()

set(GC_LIB_LINKED_LIBS
        GCPasses
        MLIROneDNNGraph
)
add_library(graph_compiler SHARED ${GC_LIB_SOURCES})
target_include_directories(graph_compiler PUBLIC ${GC_LIB_INCLUDES})
target_compile_options(graph_compiler PRIVATE -fvisibility=hidden)
target_link_options(graph_compiler PRIVATE -Wl,--gc-sections)
target_link_libraries(graph_compiler PRIVATE ${GC_LIB_LINKED_LIBS})

add_subdirectory(test)
