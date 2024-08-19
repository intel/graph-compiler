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

if(GC_ENABLE_BINDINGS_PYTHON)
  if(NOT MLIR_ENABLE_BINDINGS_PYTHON)
    message(STATUS "Failed to enable Python API due to the 'MLIR_ENABLE_BINDINGS_PYTHON' for LLVM is not ON.")
    set(GC_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "" FORCE)
    return()
  endif()
  message(STATUS "Enabling Python API")
else()
  return()
endif()

include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()
include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `gc_mlir`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=gc_mlir.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(GcPythonSources)

declare_mlir_python_sources(GcPythonSources.Common
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gc_mlir"
  ADD_TO_PARENT GcPythonSources
  SOURCES
    __init__.py
    graph_compiler.py
    dialects/__init__.py
    # init hooks
    _mlir_libs/_site_initialize_0.py
)

################################################################################
# Dialect bindings
################################################################################

if(GC_ENABLE_DNNL)
  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT GcPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gc_mlir"
    TD_FILE dialects/OneDNNGraphOps.td
    SOURCES 
      dialects/onednn_graph.py
    DIALECT_NAME onednn_graph
  )
endif()

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT GcPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gc_mlir"
  TD_FILE dialects/CPURuntimeOps.td
  SOURCES 
    dialects/cpuruntime.py
  DIALECT_NAME cpuruntime
)

declare_mlir_python_extension(GcPythonSources.Extension
  MODULE_NAME _gc_mlir
  ADD_TO_PARENT GcPythonSources
  SOURCES
    MainModule.cpp
  EMBED_CAPI_LINK_LIBS
    GcCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(GcPythonCAPI
  INSTALL_COMPONENT GcPythonModules
  INSTALL_DESTINATION python_packages/gc_mlir_core/gc_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/gc_mlir_core/gc_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    GcPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.linalg
    MLIRPythonSources.ExecutionEngine
)
target_link_libraries(GcPythonCAPI PUBLIC GcInterface)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(GcPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/gc_mlir_core/gc_mlir"
  INSTALL_PREFIX "python_packages/gc_mlir_core/gc_mlir"
  DECLARED_SOURCES
    GcPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
    MLIRPythonSources.ExecutionEngine
  COMMON_CAPI_LINK_LIBS
    GcPythonCAPI
  )

add_dependencies(GcPythonModules GcCpuRuntime)
configure_file(config.py.in ${MLIR_BINARY_DIR}/python_packages/gc_mlir_core/gc_mlir/config.py @ONLY)
