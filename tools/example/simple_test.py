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
from numpy.testing import assert_allclose

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import ml_dtypes
from bench import py_timeit_bench
from enhanced_np_to_memref import ranked_memref_to_numpy
from utils import get_mlir_args

if __name__ == "__main__":
    with ir.Context() as ctx:
        module = ir.Module.parse(
            """
            module {
                func.func @main_entry(%arg0:tensor<10x10xbf16>, %arg1:tensor<10x10xbf16>, %arg2:tensor<10xbf16>) -> tensor<10x10xbf16> attributes {llvm.emit_c_interface} {
                    %0 = onednn_graph.matmul %arg0, %arg1, %arg2 : (tensor<10x10xbf16>, tensor<10x10xbf16>, tensor<10xbf16>) -> tensor<10x10xbf16>
                    return %0:tensor<10x10xbf16>
                    }
                }      
            """
        )
        arg0 = np.ones((10, 10), dtype=ml_dtypes.bfloat16)
        arg1 = np.ones((10, 10), dtype=ml_dtypes.bfloat16)
        arg2 = np.zeros((10), dtype=ml_dtypes.bfloat16)
        
    
        entry = "main_entry"
        mlir_args = get_mlir_args(module, entry, [arg0, arg1, arg2])
        passes = "any(gc-cpu-pipeline)"
        cost = py_timeit_bench(
            ctx,
            module,
            "main_entry",
            passes,
            mlir_args,
            [os.environ["MLIR_C_RUNNER_UTILS"], os.environ["MLIR_RUNNER_UTILS"]],
        )

        # get result
        print("cost=", cost)
        gc_res = ranked_memref_to_numpy(mlir_args[0][0])
        print(gc_res)
        assert_allclose(gc_res, 10.0, rtol=1e-5, atol=0)
