
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

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char *argv[]) {
    mlir::registerAllPasses();
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    return mlir::asMainReturnCode(mlir::MlirOptMain(
            argc, argv, "Graph Compiler modular optimizer driver\n", registry));
}
