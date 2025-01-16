################################################################################
# Copyright (C) 2025 Intel Corporation
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

if(NOT GC_ENABLE_TOOLS OR NOT GC_ENABLE_GPU)
  message(STATUS "Gpu runner is not enabled.")
  return()
endif()

gc_add_mlir_tool(gc-gpu-runner GpuRunner.cpp)
target_link_libraries(gc-gpu-runner PRIVATE
  GcJitWrapper
  GcGpuOclRuntime
)
mlir_check_all_link_libraries(gc-gpu-runner)
