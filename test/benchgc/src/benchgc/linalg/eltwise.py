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

from benchgc.mlir import MLIRCache, init_i1o1_module

from gc_mlir.dialects import linalg

from benchgc.arg import Arg
from typing import Dict, List


def ref_negf(cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    var[cache.res[0]] = torch.neg(var[cache.opr[0]])


def mlir_negf(
    flags: argparse.Namespace, ins: List[Arg], outs: List[Arg]
) -> gc_mlir.ir.Module:
    return init_i1o1_module(
        ins[0],
        outs[0],
        lambda ctx, arg0: linalg.negf(arg0, outs=[outs[0].get_empty_op(ctx)]),
    )


def ref_exp(cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    var[cache.res[0]] = torch.exp(var[cache.opr[0]])


def mlir_exp(
    flags: argparse.Namespace, ins: List[Arg], outs: List[Arg]
) -> gc_mlir.ir.Module:
    return init_i1o1_module(
        ins[0],
        outs[0],
        lambda ctx, arg0: linalg.negf(arg0, outs=[outs[0].get_empty_op(ctx)]),
    )
