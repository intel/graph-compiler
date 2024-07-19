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
from gc_mlir.dialects.linalg.opdsl.lang.comprehension import TypeFnType
import gc_mlir.ir

from benchgc.mlir.util import MLIRCache, init_i2o1_module

from gc_mlir.dialects import linalg

from benchgc.arg import Arg
from typing import Dict, List


def ref_matmul_transpose_b(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
):
    var[cache.res[0]] = torch.matmul(
        var[cache.opr[0]], var[cache.opr[1]].transpose(-1, -2)
    )


def mlir_matmul_transpose_b(
    flags: argparse.Namespace, ins: List[Arg], outs: List[Arg]
) -> gc_mlir.ir.Module:
    return init_i2o1_module(
        ins[0],
        ins[1],
        outs[0],
        lambda ctx, arg0, arg1: linalg.matmul_transpose_b(
            arg0, arg1, outs=[outs[0].get_empty_op(ctx)], cast=TypeFnType(flags.cast)
        ),
    )
