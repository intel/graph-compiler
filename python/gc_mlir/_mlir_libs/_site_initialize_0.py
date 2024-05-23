# ===-- _site_initialize_0.py - DESC ---------------------------*- Python -*-===#
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===------------------------------------------------------------------------===#


def context_init_hook(context):
    from ._gc_mlir.onednn_graph import register_dialect

    register_dialect(context)
