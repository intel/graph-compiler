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

from . import util, graph, gapi
import torch
from typing import Dict, List


class RefRunner:
    # tensor id -> torch tensor
    tensors: Dict[int, torch.Tensor] = {}

    def __init__(self, graph: graph.Graph):
        self.graph = graph
        for tensor_id in graph.tensors:
            self.tensors[tensor_id] = graph.tensors[tensor_id].clone()

    def execute_ref_op(
        self, op: gapi.Op, ins: List[torch.Tensor], outs: List[torch.Tensor]
    ):
        exec_op = self.graph.exec_ops[op.id]
        exec_op.ref(ins, outs)

        # check output shape with the definition in json
        for i in range(len(op.outputs)):
            if list(outs[i].shape) != op.outputs[i].shape:
                raise Exception(
                    "tensor %d shape mismatch! json: %s / runtime: %s"
                    % (
                        op.outputs[i].id,
                        str(op.outputs[i].shape),
                        str(list(outs[i].shape)),
                    )
                )
            # for u8/s8 result, clamp the tensor
            if op.outputs[i].dtype == "u8":
                outs[i] = torch.clamp(outs[i], 0, 255)
            elif op.outputs[i].dtype == "s8":
                outs[i] = torch.clamp(outs[i], -128, 127)

            # cast if the data type mismatch
            dt = util.get_dtype(op.outputs[i].dtype)
            if dt != outs[i].dtype:
                outs[i] = outs[i].to(dt)

    def execute(self):
        for op in self.graph.topo_ops:
            ins: List[torch.Tensor] = []
            outs: List[torch.Tensor] = []

            # prepare the input tensor for this op
            for i in op.inputs:
                ins.append(self.tensors[i.id])

            self.execute_ref_op(op, ins, outs)

            # save the result tensor into self.tensors
            for idx in range(len(op.outputs)):
                self.tensors[op.outputs[idx].id] = outs[idx]
