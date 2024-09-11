# ===-- cpuinfo.py - Getting the CPU info ---------------------*- Python -*-===#
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#

from .._mlir_libs import _cpuinfo

_cache_sizes = []
_max_vector_width = None


def get_cache_sizes():
    global _cache_sizes
    if not _cache_sizes:
        _cache_sizes = _cpuinfo.get_cache_sizes()
    return _cache_sizes


def get_max_vector_width():
    global _max_vector_width
    if _max_vector_width is None:
        _max_vector_width = _cpuinfo.get_max_vector_width()
    return _max_vector_width
