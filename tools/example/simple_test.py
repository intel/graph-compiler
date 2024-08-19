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

import os
import sys

import numpy as np
from gc_mlir import ir
from gc_mlir.graph_compiler import GraphCompiler
from numpy.testing import assert_allclose

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import ml_dtypes
import torch
from utils import get_mlir_args

# an example of simple validation
if __name__ == "__main__":
    with ir.Context() as ctx:
        ctx.enable_multithreading(False)
        module = ir.Module.parse(
            """
                module {
                  func.func @main_entry(%arg0: tensor<10x10xbf16>, %arg1: tensor<10x10xbf16>) -> tensor<10x10xbf16> attributes {llvm.emit_c_interface} {
                    %cst = arith.constant 0.000000e+00 : bf16
                    %0 = tensor.empty() : tensor<10x10xbf16>
                    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<10x10xbf16>) -> tensor<10x10xbf16>
                    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<10x10xbf16>, tensor<10x10xbf16>) outs(%1 : tensor<10x10xbf16>) -> tensor<10x10xbf16>
                    return %2 : tensor<10x10xbf16>
                  }
                }    
            """
        )
        torch_arg0 = torch.full((10, 10), 1.0, dtype=torch.bfloat16)
        torch_arg1 = torch.full((10, 10), 1.0, dtype=torch.bfloat16)
        ref_res = torch.matmul(torch_arg0, torch_arg1)
               
        np_arg0 = torch_arg0.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
        np_arg1 = torch_arg1.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)      
        gc_res = np.zeros((10, 10), dtype=ml_dtypes.bfloat16)

        entry = "main_entry"
        mlir_args = get_mlir_args(module, entry, [np_arg0, np_arg1, gc_res])
        passes = "any(gc-cpu-pipeline)"
        
        # just run
        compiler = GraphCompiler(passes)
        engine = compiler.compile_and_jit(module, ir_printing=True)
        engine.invoke(entry, *mlir_args)
            
        print(gc_res)
        assert_allclose(
            gc_res.astype(np.float32),
            ref_res.to(torch.float32).numpy(),
            rtol=1e-5,
            atol=0,
        )
