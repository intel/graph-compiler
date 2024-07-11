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
import importlib
import gc_mlir.ir
from benchgc.mlir import MLIRCache
from benchgc.arg import Arg
from typing import Dict, Callable, List

ref_op: Dict[str, Callable[[MLIRCache, gc_mlir.ir.OpView, Dict[str, torch.Tensor]], None]] = {}
mlir_op: Dict[
    str, Callable[[argparse.Namespace, List[Arg], List[Arg]], gc_mlir.ir.Module]
] = {}

for dri in ["basic"]:
    mod = importlib.import_module("benchgc.arith.%s" % dri)
    for key in mod.__dict__:
        if key.startswith("ref_"):
            op: str = key.removeprefix("ref_")
            ref_op[op] = mod.__dict__[key]
        if key.startswith("mlir_"):
            op: str = key.removeprefix("mlir_")
            mlir_op[op] = mod.__dict__[key]

