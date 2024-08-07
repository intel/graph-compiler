# ===-- _site_initialize_0.py - For site init -----------------*- Python -*-===#
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#


def context_init_hook(context):

    from ._gc_mlir.cpuruntime import register_dialect as register_cpuruntime_dialect

    register_cpuruntime_dialect(context)
    
    try:
        from ._gc_mlir.onednn_graph import (
            register_dialect as register_onednn_graph_dialect,
        )

        register_onednn_graph_dialect(context)
    except ModuleNotFoundError:
        print("onednn_graph dialect not found")
