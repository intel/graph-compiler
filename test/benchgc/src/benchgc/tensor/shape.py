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

from typing import Dict, List, Tuple

import gc_mlir.ir
import torch
from benchgc.mlir.util import MLIRCache


def ref_collapse_shape(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    # permute axis and do reshape
    reassociation: gc_mlir.ir.ArrayAttr = op.attributes["reassociation"]
    permutation: List[int] = []
    shape: List[int] = []
    for outdim in reassociation:
        d: int = 1
        for indim in outdim:
            permutation.append(int(indim))
            d = d * int(op.operands[0].type.shape[int(indim)])
        shape.append(d)
    return (
        torch.permute(var[cache.opr[0]], tuple(permutation))
        .contiguous()
        .reshape(shape)
        .contiguous(),
    )


def ref_expand_shape(
    cache: MLIRCache, op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    # permute axis and do reshape
    reassociation: gc_mlir.ir.ArrayAttr = op.attributes["reassociation"]
    permutation: List[int] = [0] * len(op.result.type.shape)
    shape: List[int] = []

    d: int = 0
    for indim in reassociation:
        for outdim in indim:
            shape.append(int(op.result.type.shape[int(outdim)]))
            permutation[int(outdim)] = d
            d = d + 1
    return (torch.reshape(var[cache.opr[0]], shape).permute(permutation).contiguous(),)
