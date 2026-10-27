# ##############################################################################
# Copyright (C) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License. SPDX-License-Identifier: Apache-2.0
# ##############################################################################

cmake_minimum_required(VERSION 3.20)
project(
  GraphCompiler
  VERSION "1"
  LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)

# ########################### Build options ####################################
option(GC_ENABLE_TEST "Build the tests" ON)
option(GC_ENABLE_TOOLS "Build the tools" ON)
option(GC_ENABLE_OPT "Build gc-opt" ${GC_ENABLE_TOOLS})
option(GC_ENABLE_ZE "Build Level Zero GPU runtime" ON)
cmake_dependent_option(GC_ENABLE_PYTHON_BINDINGS "Build Python bindings" ON
                       "GC_ENABLE_ZE" OFF)
option(GC_DYLINK "Link dynamic libraries of LLVM and MLIR" OFF)
set(GC_EXTERNALS_DIR
    "${PROJECT_SOURCE_DIR}/externals"
    CACHE PATH "Directory for external dependencies")
set(MLIR_DIR
    "${GC_EXTERNALS_DIR}/llvm-project/build/lib/cmake/mlir"
    CACHE PATH "MLIR cmake directory")
set(LLVM_DIR
    "${MLIR_DIR}/../llvm"
    CACHE PATH "LLVM cmake directory")
# ##############################################################################

# ############################# Cmake options ##################################
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffunction-sections -fdata-sections")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "-O2 -g -DNDEBUG -ffunction-sections -fdata-sections")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Werror -Werror=missing-declarations)
endif()

# Recent MLIR marks some of the default op builders that mlir-tblgen generates
# as deprecated, and the generated *.inc code itself calls them from other
# deprecated builders. Our sources never call these builders, so there is
# nothing to fix on our side. Clang ignores calls made from inside deprecated
# functions, but GCC warns about them and -Werror turns the warning into a build
# failure. Keep them as plain warnings for GCC.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_compile_options(-Wno-error=deprecated-declarations)
endif()

# Silence a false positive GCC -Wunused-but-set-parameter warning
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION
                                           VERSION_LESS "10.0")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-Wno-unused-but-set-parameter"
                          CXX_SUPPORTS_WNO_UNUSED_BUT_SET_PARAMETER)
  if(CXX_SUPPORTS_WNO_UNUSED_BUT_SET_PARAMETER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-parameter")
  endif()
endif()
# ##############################################################################

# ############################# Targets ########################################
# All common options, includes etc. are added to this interface target.
add_library(GcInterface INTERFACE)
target_compile_features(GcInterface INTERFACE cxx_std_17)
target_include_directories(
  GcInterface
  INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
if(NOT GC_DYLINK AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(GcInterface INTERFACE -ffunction-sections
                                               -fdata-sections)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(functions)
include(mlir)

find_package(Git QUIET)
add_custom_target(
  GcVersion ALL
  COMMAND
    ${CMAKE_COMMAND} -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    -DBUILD_DIR="${PROJECT_BINARY_DIR}" -DGIT_EXECUTABLE="${GIT_EXECUTABLE}"
    -DVERSION_CODE="${PROJECT_VERSION_MAJOR}"
    -DLLVM_VERSION_MAJOR="${LLVM_VERSION_MAJOR}"
    -DLLVM_VERSION_SUFFIX="${LLVM_VERSION_SUFFIX}" -P
    "${PROJECT_SOURCE_DIR}/cmake/version.cmake"
  BYPRODUCTS "${PROJECT_BINARY_DIR}/include/gc/Version.h"
  COMMENT "Generating Version.h")
add_dependencies(GcInterface GcVersion)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(test)
# ##############################################################################

# ############################## Install #######################################
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ TYPE INCLUDE)
install(
  DIRECTORY ${PROJECT_BINARY_DIR}/include/
  TYPE INCLUDE
  REGEX "CMake.*|.*cmake" EXCLUDE)

# Export the targets
get_property(GC_TOOLS GLOBAL PROPERTY GC_TOOLS)

if(GC_DYLINK)
  set(GC_LIBS ${PROJECT_NAME})
else()
  get_property(GC_MLIR_LIBS GLOBAL PROPERTY GC_MLIR_LIBS)
  get_property(GC_PASS_LIBS GLOBAL PROPERTY GC_PASS_LIBS)
  get_property(GC_DIALECT_LIBS GLOBAL PROPERTY GC_DIALECT_LIBS)
  set(GC_LIBS ${PROJECT_NAME} ${GC_MLIR_LIBS} ${GC_PASS_LIBS}
              ${GC_DIALECT_LIBS})
endif()

if(GC_ENABLE_ZE)
  list(APPEND GC_LIBS GcLevelZero)
endif()

install(
  TARGETS GcInterface ${GC_TOOLS} ${GC_LIBS}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

export(EXPORT ${PROJECT_NAME}Targets
       FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
# ##############################################################################
