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

import argparse
from abc import ABC, abstractmethod

from gc_mlir import ir


class Pattern(ABC):
    """Abstract class for pattern."""

    @staticmethod
    @abstractmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add arguments to parser"""

    @abstractmethod
    def handle_args(self, args: argparse.Namespace):
        """Get and handle the args"""

    def __init__(self, ctx: ir.Context, args: argparse.Namespace):
        self.main_entry = "entry"
        self.handle_args(args)
        self.ir_module = self.init_module(ctx)

    @abstractmethod
    def init_module(self, ctx: ir.Context) -> ir.Module:
        """Create MLIR moudule by args"""
