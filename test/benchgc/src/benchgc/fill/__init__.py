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
import torch
from benchgc.arg import Arg
from typing import Dict
import benchgc.util
import importlib


def set_default_fill_param(flags: argparse.Namespace, args: Dict[str, Arg], arg: Arg):

    if arg.shape == [] or arg.dtype == "" or arg.type == "":
        raise Exception("arg %s filling: shape/dtype/fill_type is not set" % arg.name)
    if arg.type == "D" and len(arg.param) == 0:
        # need to generate a default param for driver filling here
        if flags.driver not in ["linalg"]:
            raise Exception("unsupported driver %s for default filling" % flags.driver)

        if flags.case in ["add"]:
            arg.param = [
                "binary",
                arg.name,
                args["src0"].dtype,
                args["src1"].dtype,
                args["dst"].dtype,
            ]
        elif flags.case in [
            "batch_matmul",
            "batch_matmul_transpose_a",
            "batch_matmul_transpose_b",
            "batch_matvec",
            "batch_mmt4d",
            "batch_reduce_matmul",
            "batch_vecmat",
        ]:
            arg.param = [
                "matmul",
                arg.name,
                args["src"].dtype,
                args["wei"].dtype,
                args["dst"].dtype,
            ]

            if (
                flags.case == "batch_matmul_transpose_a"
                and arg.name == "src"
                or flags.case == "batch_matmul_transpose_b"
                and arg.name == "wei"
                or flags.case == "batch_matmul"
                and arg.name == "src"
                or flags.case == "batch_matvec"
                or flags.case == "batch_vecmat" and arg.name == "src"
            ):
                arg.param.append(str(arg.shape[-1]))

            elif flags.case == "batch_reduce_matmul" and arg.name == "src":
                arg.param.append(str(arg.shape[0] * arg.shape[-1]))
            elif flags.case == "batch_reduce_matmul" and arg.name == "wei":
                arg.param.append(str(arg.shape[0] * arg.shape[-2]))
            elif flags.case == "batch_mmt4d":
                arg.param.append(str(arg.shape[-1] * arg.shape[-3]))
            else:
                arg.param.append(str(arg.shape[-2]))
        elif flags.case in ["abs"]:
            arg.param = ["eltwise", flags.case, "", ""]


def fill_tensor(flags: argparse.Namespace, arg: Arg) -> torch.Tensor:
    if arg.shape == [] or arg.dtype == "" or arg.type == "":
        raise Exception("arg %s filling: shape/dtype/fill_type is not set" % arg.name)

    if arg.type == "N" and len(arg.param) == 2:
        # Normal distribution
        mean = float(arg.param[0])
        std = float(arg.param[1])
        tensor = torch.normal(mean=mean, std=std, size=arg.shape)

    elif arg.type == "P" and len(arg.param) == 1:
        # Poisson distribution
        _lambda = float(arg.param[0])
        lambda_tensor = torch.full(arg.shape, _lambda)
        tensor = torch.poisson(lambda_tensor)
    elif arg.type == "B" and len(arg.param) == 2:
        # Binomial distribution
        n = int(arg.param[0])
        p = float(arg.param[1])
        bdist = torch.distributions.binomial.Binomial(total_count=n, probs=p)
        tensor = bdist.sample(torch.Size(arg.shape))
    elif arg.type == "U" and len(arg.param) == 2:
        # Uniform distribution
        a = float(arg.param[0])
        b = float(arg.param[1])
        tensor = torch.distributions.uniform.Uniform(a, b).sample(torch.Size(arg.shape))
    elif arg.type == "F" and len(arg.param) == 1:
        # read from pytorch tensor dump file
        filename = arg.param[0]
        tensor = torch.load(f=filename)
        if not isinstance(tensor, torch.Tensor):
            raise Exception(
                "torch object from file %s is not a tensor object" % filename
            )
        if tensor.shape != torch.Size(arg.shape):
            raise Exception(
                "tensor object from file %s does not match shape" % filename
            )
        if tensor.dtype != benchgc.util.get_dtype(arg.dtype):
            raise Exception(
                "tensor object from file %s does not match dtype" % filename
            )
    elif arg.type == "D" and len(arg.param) > 0:
        # Driver fill
        driver: str = arg.param[0]
        driver_module = importlib.import_module("benchgc.fill.%s" % driver)
        tensor = driver_module.fill(
            arg.shape, benchgc.util.get_dtype(arg.dtype), arg.param[1:]
        )
    else:
        raise Exception("invalid fill type or fill parameter")

    tensor = tensor.to(benchgc.util.get_dtype(arg.dtype))
    if flags.verbose >= benchgc.util.INPUT_VERBOSE:
        print("fill arg: " + arg.name)
        print(tensor)
    return tensor
