################################################################################
# Copyright (C) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# SPDX-License-Identifier: Apache-2.0
################################################################################


from typing import List

from benchgc.tuner.op_config import OP_TO_CONFIG, Config
from gc_mlir import ir


def get_all_tunable_ops(op: ir.Operation):
    """Get tunable ops from the children op"""
    tunable_ops = []
    for region in op.regions:
        for block in region:
            for child_op in block:
                if (
                    "skipTuner" in child_op.attributes
                    and child_op.attributes["skipTuner"]
                ):
                    continue
                if child_op.name in OP_TO_CONFIG:
                    tunable_ops.append(child_op)
                tunable_ops = tunable_ops + get_all_tunable_ops(child_op)
    return tunable_ops


def gen_configs_from_ir(ir_module: ir.Module):
    """Genrate configs from ir module"""
    tunable_ops = get_all_tunable_ops(ir_module.operation)
    configs = []
    for op in tunable_ops:
        if op.name in OP_TO_CONFIG:
            configs.append(OP_TO_CONFIG[op.name](op))
    return configs


def attach_configs_to_ir(ir_module: ir.Module, configs: List[Config]):
    """Add configs to ir module"""
    tunable_ops = get_all_tunable_ops(ir_module.operation)
    assert len(tunable_ops) == len(
        configs
    ), "tunable ops and configs should have the same length"
    for i, op in enumerate(tunable_ops):
        if op.name in OP_TO_CONFIG:
            with ir_module.context:
                configs[i].attach_to_ir(op)
