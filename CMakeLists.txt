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

############################# Cmake options ####################################
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Silence a false positive GCC -Wunused-but-set-parameter warning in constexpr
# cases, by marking SelectedCase as used. See
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85827 for details. The issue is
# fixed in GCC 10.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
  check_cxx_compiler_flag("-Wno-unused-but-set-parameter" CXX_SUPPORTS_WNO_UNUSED_BUT_SET_PARAMETER)
  append_if(CXX_SUPPORTS_WNO_UNUSED_BUT_SET_PARAMETER "-Wno-unused-but-set-parameter" CMAKE_CXX_FLAGS)
endif()
################################################################################

############################ Build options #####################################
option(GC_ENABLE_LEGACY ON)
option(GC_ENABLE_DNNL_API "Enable the oneDNN library API integration" ON)
option(GC_ENABLE_TEST "Build the tests" ON)
option(GC_ENABLE_TEST_DNNL_API "Build the dnnl tests" ${GC_ENABLE_DNNL_API})
option(GC_ENABLE_TEST_MLIR "Build the mlir tests" ON)
option(GC_ENABLE_TOOLS "Build the tools" ON)
option(GC_ENABLE_OPT "Build gc-opt" ${GC_ENABLE_TOOLS})
option(GC_ENABLE_IMEX "Enable Intel® Extension for MLIR" OFF)
option(GC_ENABLE_BINDINGS_PYTHON "Enable Graph Complier Python Binding" ON)
option(GC_DEV_LINK_LLVM_DYLIB "Link dynamic libraries of LLVM and MLIR. For developers only. Do not use it in packing the library." OFF)
option(GC_ENABLE_RUNTIME_NAIVE_BRGEMM "Use naive BRGEMM as runtime backend for debug purpose." OFF)

if(GC_ENABLE_LEGACY)
  add_subdirectory(legacy/core)
endif()

if(GC_ENABLE_DNNL_API)
  set(GC_ONEDNN_DIALECT_LIB_NAME MLIROneDNNGraph)
endif()
################################################################################

############################## Targets #########################################
# All common options, includes etc. are added to this interface target.
add_library(GcInterface INTERFACE)
target_compile_features(GcInterface INTERFACE cxx_std_17)
target_include_directories(GcInterface INTERFACE
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

include(functions)
include(version)
include(mlir)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(python)
add_subdirectory(test)
################################################################################

############################### Install ########################################
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ TYPE INCLUDE)
install(DIRECTORY ${PROJECT_BINARY_DIR}/include/ TYPE INCLUDE
  REGEX "CMake.*|.*cmake" EXCLUDE)

# Export the targets
get_property(GC_TOOLS GLOBAL PROPERTY GC_TOOLS)
get_property(GC_MLIR_LIBS GLOBAL PROPERTY GC_MLIR_LIBS)
get_property(GC_PASS_LIBS GLOBAL PROPERTY GC_PASS_LIBS)
get_property(GC_DIALECT_LIBS GLOBAL PROPERTY GC_DIALECT_LIBS)
install(TARGETS
  GcInterface
  ${GC_TOOLS}
  ${GC_MLIR_LIBS}
  ${GC_PASS_LIBS}
  ${GC_DIALECT_LIBS}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
export(EXPORT ${PROJECT_NAME}Targets
  FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
)
install(EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION lib/cmake/${PROJECT_NAME}
)

# Generate the config files
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "lib/cmake/${PROJECT_NAME}"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  COMPATIBILITY AnyNewerVersion
)
install(FILES
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION "lib/cmake/${PROJECT_NAME}"
)
################################################################################
