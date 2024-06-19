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

from benchgc.linalg.mlir import init_binary_module

from gc_mlir.dialects import linalg

from benchgc.arg import Arg
from typing import Dict, Tuple

def __ref_init(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, str]:

    src0 = var[op.operands[0].get_name().removeprefix("%").removeprefix("@")] 
    src1 = var[op.operands[1].get_name().removeprefix("%").removeprefix("@")]
    dst_var: str = op.results[0].get_name().removeprefix("%").removeprefix("@")
    return (src0, src1, dst_var)

def map_matmul_args(args: Dict[str, Arg]):
    for k, v in {"arg0": "src", "arg1": "wei", "1": "dst"}.items():
        args[k] = args[v]
        del args[v]

def ref_batch_matmul(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    src, wei, dst_var = __ref_init(op, var) 
    var[dst_var] = torch.matmul(src, wei)

def mlir_batch_matmul(flags: argparse.Namespace, args: Dict[str, Arg]) -> gc_mlir.ir.Module:
    map_matmul_args(args)
    return init_binary_module(flags, args, linalg.batch_matmul)