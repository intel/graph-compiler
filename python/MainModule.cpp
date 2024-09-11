
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
}