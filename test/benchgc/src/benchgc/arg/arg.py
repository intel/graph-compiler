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

from typing import List

import benchgc.mlir.arg
import benchgc.util


class Arg(benchgc.mlir.arg.MLIRArg):
    fill_type: str
    fill_param: List[str]

    cmp_type: str
    cmp_param: List[str]

    index: int

    def __init__(self, index: int):
        self.dtype = ""
        self.fill_type = "-"
        self.fill_param = []
        self.cmp_type = "-"
        self.cmp_param = []
        self.index = index

    def print_verbose(self, verbose: int):
        if verbose >= benchgc.util.ARG_VERBOSE:
            print(
                f"arg{self.index} shape: {self.shape} dtype: {self.dtype} fill_type: {self.fill_type} fill_param: {self.fill_param} cmp_type: {self.cmp_type} cmp_param: {self.cmp_param}"
            )

    def set_fill(self, fill: str):
        splited: List[str] = fill.split(":")
        self.fill_type = splited[0]
        self.fill_param = splited[1:]

    def set_cmp(self, cmp: str):
        splited: List[str] = cmp.split(":")
        self.cmp_type = splited[0]
        self.cmp_param = splited[1:]
