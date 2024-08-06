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

import torch
import argparse
import gc_mlir.ir

from benchgc.mlir.util import MLIRCache
from benchgc.mlir.module import init_i1o1_module

from gc_mlir.dialects import linalg

from benchgc.arg import Arg
from typing import Dict, List

def ref_softmax(cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    dimension: gc_mlir.ir.IntegerAttr = op.attributes["dimension"]
    var[cache.res[0]] = torch.softmax(var[cache.opr[0]], dimension.value)


def mlir_softmax(flags: argparse.Namespace, args: List[Arg]) -> gc_mlir.ir.Module:
    return init_i1o1_module(
        args[0],
        args[1],
        lambda ctx, arg0: linalg.softmax(result=[args[1].get_ranked_tensor_type(ctx)], input=arg0, output=args[1].get_empty_op(ctx), dimension=flags.dimension),
    )
