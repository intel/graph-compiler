
/*
 * Copyright (C) 2025 Intel Corporation
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

#include "gc/Transforms/Passes.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace mlir::gc {
void registerGPUPipeline();
} // namespace mlir::gc

int main(int argc, char *argv[]) {
  mlir::DialectRegistry &registry = mlir::gc::getDialectRegistry();
  mlir::gc::registerGPUPipeline();
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Graph Compiler modular optimizer driver\n", registry));
}
