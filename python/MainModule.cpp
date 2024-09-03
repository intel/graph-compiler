
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

#include <stdint.h>
#include <stdio.h>

#include <iostream>

// 使用GCC内联汇编的CPUID函数
void cpuid(int info[4], int InfoType, int ECXValue) {
  __asm__ __volatile__("cpuid"
                       : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]),
                         "=d"(info[3])
                       : "a"(InfoType), "c"(ECXValue));
}

void get_cpu_info() {
  int info[4];
  cpuid(info, 0, 0); // 获取最大的CPUID功能号
  int nIds = info[0];

  for (int i = 0; i <= nIds; ++i) {
    cpuid(info, 4, i); // 查询缓存参数
    int cacheType = info[0] & 0x1F;
    if (cacheType == 0) {
      break; // 没有更多的缓存级别
    }
    int cacheLevel = (info[0] >> 5) & 0x7;
    int cacheLinesPerTag = ((info[1] >> 0) & 0xFFF) + 1;
    int cacheAssociativity = ((info[1] >> 12) & 0x3FF) + 1;
    int cachePartitions = ((info[1] >> 22) & 0x3FF) + 1;
    int cacheSets = info[2] + 1;
    int cacheSize =
        cacheLinesPerTag * cacheAssociativity * cachePartitions * cacheSets;

    std::cout << "L" << cacheLevel << " ";
    if (cacheType == 1) {
      std::cout << "Data Cache: ";
    } else if (cacheType == 2) {
      std::cout << "Instruction Cache: ";
    } else if (cacheType == 3) {
      std::cout << "Unified Cache: ";
    }
    std::cout << cacheSize << " bytes" << std::endl;
  }
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

  cpuruntimeM.def("get_cpu_info", &get_cpu_info, "---");
}