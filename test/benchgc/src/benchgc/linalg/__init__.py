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

import argparse
import importlib
from typing import Callable, Dict, List, Tuple

import torch
from benchgc.arg import Arg
from benchgc.mlir.util import MLIRCache
from gc_mlir import ir

ref_op: Dict[
    str,
    Callable[
        [MLIRCache, ir.OpView, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, ...],
    ],
] = {}
mlir_op: Dict[str, Callable[[argparse.Namespace, List[Arg]], ir.Module]] = {}

for dri in [
    "binary",
    "matmul",
    "eltwise",
    "misc",
    "generic",
    "softmax",
    "conv",
    "pool",
    "reduce",
]:
    mod = importlib.import_module(f"benchgc.linalg.{dri}")
    for key in mod.__dict__:
        if key.startswith("ref_"):
            op: str = key.removeprefix("ref_")
            ref_op[op] = mod.__dict__[key]
        if key.startswith("mlir_"):
            op: str = key.removeprefix("mlir_")
            mlir_op[op] = mod.__dict__[key]
