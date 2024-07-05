################################################################################
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

from benchgc.mlir import MLIRCache
import gc_mlir.ir
import gc_mlir._mlir_libs._mlir.ir
import torch
import benchgc.util

from typing import Dict

def ref_constant(cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    value = op.attributes["value"]
    if isinstance(value, gc_mlir._mlir_libs._mlir.ir.FloatAttr):
        var[cache.res[0]] = torch.full(size=tuple(), 
            fill_value= value.__float__(), 
            dtype = torch.float)
    elif isinstance(value, gc_mlir._mlir_libs._mlir.ir.DenseFPElementsAttr):
        if value.is_splat:
            var[cache.res[0]] = torch.full(size=tuple(value.type.shape), 
                fill_value= value.get_splat_value().value,
                dtype = benchgc.util.get_dtype(str(value.get_splat_value().type)))
        else:
            raise Exception("only support splat value now")
    else:
        raise Exception("Not support constant type %s", type(value))

def ref_mulf(cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    var[cache.res[0]] = var[cache.opr[0]] * var[cache.opr[1]]

def ref_addf(cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    var[cache.res[0]] = var[cache.opr[0]] + var[cache.opr[1]]