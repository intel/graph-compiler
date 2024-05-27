from gc_mlir import execution_engine
from gc_mlir import ir
from gc_mlir import passmanager
from typing import Sequence


class GraphComplier:
    def __init__(self, shared_libs: Sequence[str], pipeline: str, opt_level: int = 3):
        self.shared_libs = shared_libs
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
