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

if(GC_DEV_LINK_LLVM_DYLIB)
  set(LLVM_LINK_COMPONENTS
    LLVM
    )
  set(MLIR_LINK_COMPONENTS
    MLIR
    MLIRExecutionEngineShared
    MLIRJitRunner
  )
else()
  # the dependency list copied from mlir/tools/mlir-cpu-runner/CMakeLists.txt of upstream
  set(LLVM_LINK_COMPONENTS
    Core
    Support
    nativecodegen
    native
  )
  set(MLIR_LINK_COMPONENTS
    MLIRAnalysis
    MLIRBuiltinToLLVMIRTranslation
    MLIRExecutionEngine
    MLIRIR
    MLIRJitRunner
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRToLLVMIRTranslationRegistration
    MLIRParser
    MLIRTargetLLVMIRExport
    MLIRSupport
  )
endif()

if(GC_MLIR_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GC_MLIR_CXX_FLAGS}")
endif()

#LLVM_LINK_COMPONENTS is processed by LLVM cmake in add_llvm_executable
set(gc_cpu_runner_libs
        ${MLIR_LINK_COMPONENTS}
        GCCpuRuntime)
if(GC_USE_GPU)
        list(APPEND gc_cpu_runner_libs
        MLIRGENTarget)
endif()
add_mlir_tool(gc-cpu-runner
        gc-cpu-runner.cpp
)
llvm_update_compile_flags(gc-cpu-runner)

target_link_libraries(gc-cpu-runner PRIVATE ${gc_cpu_runner_libs})
mlir_check_all_link_libraries(gc-cpu-runner)
