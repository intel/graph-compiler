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

import gc_mlir.ir
import torch
from benchgc.mlir import escape_var
import benchgc.util

from typing import Dict

def ref_empty(op: gc_mlir.ir.OpView, var: Dict[str, torch.Tensor]):
    var[escape_var(op.results[0].get_name())] = torch.zeros(size = op.results[0].type.shape, dtype = benchgc.util.get_dtype(str(op.results[0].type.element_type)))