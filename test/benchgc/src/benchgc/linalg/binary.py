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

from benchgc.arg import Arg
from benchgc.linalg.mlir import init_i2o1_module, escape_var
from typing import Dict, Tuple

from gc_mlir.dialects import linalg
import gc_mlir.ir


def __ref_init(
    op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, str]:

    src0 = var[escape_var(op.operands[0].get_name())]
    src1 = var[escape_var(op.operands[1].get_name())]
    dst_var: str = escape_var(op.results[0].get_name())
    return (src0, src1, dst_var)


def map_binary_args(args: Dict[str, Arg]):
    for k, v in {"arg0": "src0", "arg1": "src1", "1": "dst"}.items():
        args[k] = args[v]
        del args[v]


def ref_add(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src0, src1, dst_var = __ref_init(op, var)
    var[dst_var] = torch.add(src0, src1)


def mlir_add(flags: argparse.Namespace, args: Dict[str, Arg]) -> gc_mlir.ir.Module:
    map_binary_args(args)
    return init_i2o1_module(flags, args, linalg.add)
