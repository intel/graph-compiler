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

if(NOT GC_ENABLE_DNNL)
  message(STATUS "oneDNN library integration is not enabled.")
  return()
endif()

include(onednn)

set(GC_DNNL_SOURCES
  JsonParser.cpp
  dnnl_graph_compiler.cpp
)
set(GC_DNNL_LINKED_LIBS
  GcJitWrapper
  GcCpuRuntime
)
gc_add_mlir_library(GcDnnl SHARED
  ${GC_DNNL_SOURCES}
  LINK_LIBS PRIVATE ${GC_DNNL_LINKED_LIBS}
)
target_link_libraries(GcDnnl PUBLIC GcInterface)
target_include_directories(GcDnnl PUBLIC ${GC_DNNL_INCLUDES})
target_compile_options(GcDnnl PRIVATE -fvisibility=hidden -fexceptions)
target_link_options(GcDnnl PRIVATE -Wl,--gc-sections)

if(GC_ENABLE_TEST_DNNL)
  # Static graph compiler library to be used in tests
  gc_add_mlir_library(GcDnnlStatic STATIC
    ${GC_DNNL_SOURCES}
    LINK_LIBS PUBLIC GcInterface ${GC_DNNL_LINKED_LIBS}
  )
  target_compile_options(obj.GcDnnlStatic PUBLIC -fexceptions)
  target_include_directories(GcDnnlStatic PUBLIC ${GC_DNNL_INCLUDES})
endif()
