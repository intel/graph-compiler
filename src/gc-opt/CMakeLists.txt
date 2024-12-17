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

if(NOT GC_ENABLE_OPT)
  message(STATUS "gc-opt is not enabled.")
  return()
endif()

if(GC_DEV_LINK_LLVM_DYLIB)
  set(MLIR_LINK_COMPONENTS
    MLIR
  )
  get_property(dialect_libs GLOBAL PROPERTY GC_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY GC_PASS_LIBS)
else()
  set(MLIR_LINK_COMPONENTS
    MLIROptLib
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRToLLVMIRTranslationRegistration
  )
  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
endif()

add_llvm_executable(gc-opt gc-opt.cpp)
llvm_update_compile_flags(gc-opt)
mlir_check_all_link_libraries(gc-opt)

get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(gc-opt PUBLIC GcInterface)
target_link_libraries(gc-opt PRIVATE
  ${dialect_libs}
  ${extension_libs}
  ${conversion_libs}
  ${MLIR_LINK_COMPONENTS}
  GcPasses
  GcAnalysis
)

if(GC_ENABLE_IMEX)
  include(imex)
  get_property(IMEX_INCLUDES GLOBAL PROPERTY IMEX_INCLUDES)
  target_include_directories(gc-opt PRIVATE ${IMEX_INCLUDES})
  target_link_libraries(gc-opt PRIVATE
    GcGpuPasses
    IMEXGPUXDialect
    IMEXXeTileDialect
    IMEXRegionDialect
    IMEXRegionTransforms
    IMEXTransforms
    IMEXGPUToGPUX
    IMEXGPUToSPIRV
    IMEXGPUXToLLVM
    IMEXXeGPUToVC
    IMEXXeTileToXeGPU
    IMEXUtil
  )
endif()

set_property(GLOBAL APPEND PROPERTY GC_TOOLS gc-opt)
