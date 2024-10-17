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

#include "gc/Analysis/MatmulConfigAnalysis.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

PYBIND11_MODULE(_tools, m) {

  m.def(
      "validate_matmul_config",
      [](const std::vector<uint32_t> &cfg_list, std::vector<uint32_t> &shape,
         bool allow_indivisible_innerblock, bool is_vnni_mm2d) {
        if (cfg_list.size() != 9) {
          throw std::invalid_argument("cfg_list must have exactly 9 elements");
        }
        mlir::gc::MatmulConfig cfg{cfg_list[0], cfg_list[1], cfg_list[2],
                                   cfg_list[3], cfg_list[4], cfg_list[5],
                                   cfg_list[6], cfg_list[7], cfg_list[8]};
        return mlir::gc::validateConfig(
            cfg, shape, allow_indivisible_innerblock, is_vnni_mm2d);
      },
      py::arg("cfg_list"), py::arg("shape"),
      py::arg("allow_indivisible_innerblock"), py::arg("is_vnni_mm2d"),
      "Validate the matmul configuration");
}