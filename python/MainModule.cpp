
/*
 * Copyright (C) 2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gc-c/Dialects.h"
#include "gc-c/Passes.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"


#include <stdio.h>
#include <stdint.h>
 
void get_l1_data_cache_size() {
    uint32_t eax, ebx, ecx, edx;
 
    // Query the cache information using CPUID with EAX=4 and ECX=1 (L1 data cache)
    eax = 4; // Cache information
    ecx = 0; // Cache level (0 for L1 data cache)
 
    __asm__ __volatile__(
        "cpuid"
        : "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx)
        : "a" (eax), "c" (ecx)
    );
 
    // Extract cache size information
    uint32_t cache_type = eax & 0x1F;
    if (cache_type != 1) { // 1 indicates data cache
        printf("No L1 data cache\n");
        return;
    }
 
    uint32_t cache_level = (eax >> 5) & 0x7;
    uint32_t cache_sets = ecx + 1;
    uint32_t cache_coherency_line_size = (ebx & 0xFFF) + 1;
    uint32_t cache_partitions = ((ebx >> 12) & 0x3FF) + 1;
    uint32_t cache_ways_of_associativity = ((ebx >> 22) & 0x3FF) + 1;
 
    uint32_t cache_size = cache_ways_of_associativity * cache_partitions * cache_coherency_line_size * cache_sets;
 
    printf("L%d Data Cache Size: %u KB\n", cache_level, cache_size / 1024);
}

PYBIND11_MODULE(_gc_mlir, m) {
  m.doc() = "Graph-compiler MLIR Python binding";

  mlirRegisterAllGCPassesAndPipelines();

  //===----------------------------------------------------------------------===//
  // OneDNNGraph
  //===----------------------------------------------------------------------===//
#ifdef GC_HAS_ONEDNN_DIALECT
  auto onednn_graphM = m.def_submodule("onednn_graph");
  onednn_graphM.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__onednn_graph__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
#endif

  //===----------------------------------------------------------------------===//
  // CPURuntime
  //===----------------------------------------------------------------------===//
  auto cpuruntimeM = m.def_submodule("cpuruntime");
  cpuruntimeM.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__cpuruntime__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);


  cpuruntimeM.def("get_l1_data_cache_size", &get_l1_data_cache_size, "---");
}