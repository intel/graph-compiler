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
  set(MLIR_LINK_COMPONENTS
    MLIR
  )
  get_property(dialect_libs GLOBAL PROPERTY GC_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY GC_PASS_LIBS)
else()
  set(MLIR_LINK_COMPONENTS
    MLIROptLib
  )
  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
endif()

get_property(gc_passes_libs GLOBAL PROPERTY GC_PASS_LIBS)

set(gc_opt_libs 
        ${dialect_libs}
        ${conversion_libs} 
        ${MLIR_LINK_COMPONENTS}
        ${gc_passes_libs})

if(GC_USE_IMEX)
  add_definitions(-DGC_USE_IMEX=1)
  get_property(IMEX_INCLUDES GLOBAL PROPERTY IMEX_INCLUDES)
  include_directories(${IMEX_INCLUDES})
  list(APPEND gc_opt_libs IMEXGPUXDialect IMEXXeTileDialect IMEXRegionDialect IMEXRegionTransforms
    IMEXTransforms IMEXGPUToGPUX IMEXGPUToSPIRV IMEXGPUXToLLVM IMEXXeGPUToVC IMEXXeTileToXeGPU IMEXUtil)
endif()
if(GC_MLIR_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GC_MLIR_CXX_FLAGS}")
endif()

add_llvm_executable(gc-opt gc-opt.cpp)

target_link_libraries(gc-opt PRIVATE ${gc_opt_libs})
llvm_update_compile_flags(gc-opt)
mlir_check_all_link_libraries(gc-opt)

