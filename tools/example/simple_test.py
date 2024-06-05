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
from bench import py_timeit_bench
from enhanced_np_to_memref import ranked_memref_to_numpy
from utils import get_mlir_args

if __name__ == "__main__":
    with ir.Context() as ctx:
        module = ir.Module.parse(
            """
            module {
                func.func @main_entry(%arg0:tensor<10x10xbf16>, %arg1:tensor<10x10xbf16>) -> tensor<10x10xbf16> attributes {llvm.emit_c_interface} {
                    %0 = onednn_graph.matmul %arg0, %arg1: (tensor<10x10xbf16>, tensor<10x10xbf16>) -> tensor<10x10xbf16>
                    return %0:tensor<10x10xbf16>
                    }
                }      
            """
        )
        torch_arg0 = torch.full((10, 10), 1.0, dtype=torch.bfloat16)
        torch_arg1 = torch.full((10, 10), 1.0, dtype=torch.bfloat16)
        # torch_arg0 = torch.randn((10, 10), dtype=torch.bfloat16)
        # torch_arg1 = torch.randn((10, 10), dtype=torch.bfloat16)
        ref_res = torch.matmul(torch_arg0, torch_arg1)
        
        # https://github.com/llvm/torch-mlir/blob/50f7103098ee41799a1180210f0e94400fac47cb/python/torch_mlir/extras/fx_importer.py#L1795-L1800
        np_arg0 = np.array(torch_arg0.tolist(), dtype=ml_dtypes.bfloat16)
        np_arg1 = np.array(torch_arg1.tolist(), dtype=ml_dtypes.bfloat16)

        entry = "main_entry"
        mlir_args = get_mlir_args(module, entry, [np_arg0, np_arg1])
        passes = "any(gc-cpu-pipeline)"
        shared_libs = [os.environ["MLIR_C_RUNNER_UTILS"], os.environ["MLIR_RUNNER_UTILS"]]
        # bench
        # cost = py_timeit_bench(
        #     module,
        #     "main_entry",
        #     passes,
        #     mlir_args,
        #     shared_libs,
        # )
        # print("cost=", cost)
        
        # just run
        compiler = GraphCompiler(passes, shared_libs)
        engine = compiler.compile_and_jit(module)
        engine.invoke(entry, *mlir_args)
            
        gc_res = ranked_memref_to_numpy(mlir_args[0][0])
        print(gc_res)
        assert_allclose(
            gc_res.astype(np.float32),
            ref_res.to(torch.float32).numpy(),
            rtol=1e-5,
            atol=0,
        )
