# ===-- _site_initialize_0.py - For site init -----------------*- Python -*-===#
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#


_dialect_registry = None
def get_dialect_registry():
    global _dialect_registry
    if _dialect_registry is None:
        from ._mlir import ir
        _dialect_registry = ir.DialectRegistry()
    return _dialect_registry
    
def context_init_hook(context):
    
    from ._gc_mlir.onednn_graph import register_dialect as register_onednn_graph_dialect
    from ._gc_mlir.cpuruntime import register_dialect as register_cpuruntime_dialect
    from ._gc_mlir.linalgx import register_dialect as register_linalgx_dialect
    from ._gc_mlir.linalgx import register_interface_imp as register_linalgx_interface
    
    register_onednn_graph_dialect(context)
    register_cpuruntime_dialect(context)
    register_linalgx_dialect(context)
  
    
    d = get_dialect_registry()
    register_linalgx_interface(d)
    context.append_dialect_registry(d)
