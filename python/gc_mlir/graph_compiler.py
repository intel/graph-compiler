# ===-- graph_compiler.py - DESC ------------------------------*- Python -*-===#
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#

from gc_mlir import execution_engine, ir, passmanager
from gc_mlir.config import GC_CPU_RUNTIME, MLIR_C_RUNNER_UTILS, MLIR_RUNNER_UTILS

__all__ = [
    "GraphCompiler",
]


class GraphCompiler:
    def __init__(
        self,
        pipeline: str = "any(gc-cpu-pipeline)",
        opt_level: int = 3,
    ):
        self.shared_libs = [MLIR_C_RUNNER_UTILS, MLIR_RUNNER_UTILS, GC_CPU_RUNTIME]
        self.pipeline = pipeline
        self.opt_level = opt_level

    def __call__(self, module: ir.Module, ir_printing: bool = False):
        self.compile(module, ir_printing)

    def compile(self, module: ir.Module, ir_printing: bool = False):
        pm = passmanager.PassManager.parse(self.pipeline)
        if ir_printing:
            pm.enable_ir_printing()
        pm.run(module.operation)

    def jit(self, module: ir.Module) -> execution_engine.ExecutionEngine:
        return execution_engine.ExecutionEngine(
            module, opt_level=self.opt_level, shared_libs=self.shared_libs
        )

    def compile_and_jit(
        self, module: ir.Module, ir_printing: bool = False
    ) -> execution_engine.ExecutionEngine:
        self.compile(module, ir_printing)
        return self.jit(module)
