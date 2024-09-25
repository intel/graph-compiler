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

if(NOT GC_ENABLE_TOOLS)
  message(STATUS "Cpu runner is not enabled.")
  return()
endif()

if(GC_DEV_LINK_LLVM_DYLIB)
  set(LLVM_LINK_COMPONENTS
    LLVM
    )
  set(MLIR_LINK_COMPONENTS
    MLIR
    MLIRExecutionEngineShared
    MLIRJitRunner
  )
endif()

gc_add_mlir_tool(gc-cpu-runner gc-cpu-runner.cpp)
target_link_libraries(gc-cpu-runner PRIVATE GcCpuRuntime)
mlir_check_all_link_libraries(gc-cpu-runner)
