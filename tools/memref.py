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

#  This file contains functions to convert between Memrefs and NumPy arrays and vice-versa.

import torch
import ctypes


def dtype_to_ctype(dtype: torch.dtype):
    if dtype == torch.float32:
        return ctypes.c_float
    elif dtype == torch.float64:
        return ctypes.c_double
    elif dtype == torch.int32:
        return ctypes.c_int
    elif dtype == torch.int64:
        return ctypes.c_longlong
    elif dtype == torch.uint8:
        return ctypes.c_ubyte
    elif dtype == torch.int8:
        return ctypes.c_byte
    elif dtype == torch.int16 or dtype == torch.bfloat16 or torch.float16:
        return ctypes.c_short
    elif dtype == torch.bool:
        return ctypes.c_bool
    else:
        raise ValueError(f"Unsupported torch dtype: {dtype}")


def make_nd_memref_descriptor(ndim: int, dtype: torch.dtype):
    class MemRefDescriptor(ctypes.Structure):
        """Builds an empty descriptor for the given rank/dtype, where rank>0."""

        _fields_ = [
            ("allocated", ctypes.c_longlong),
            ("aligned", ctypes.POINTER(dtype_to_ctype(dtype))),
            ("offset", ctypes.c_longlong),
            ("shape", ctypes.c_longlong * ndim),
            ("strides", ctypes.c_longlong * ndim),
        ]

    return MemRefDescriptor


def make_zero_d_memref_descriptor(dtype: torch.dtype):
    class MemRefDescriptor(ctypes.Structure):
        """Builds an empty descriptor for the given dtype, where rank=0."""

        _fields_ = [
            ("allocated", ctypes.c_longlong),
            ("aligned", ctypes.POINTER(dtype_to_ctype(dtype))),
            ("offset", ctypes.c_longlong),
        ]

    return MemRefDescriptor


class UnrankedMemRefDescriptor(ctypes.Structure):
    """Creates a ctype struct for memref descriptor"""

    _fields_ = [("rank", ctypes.c_longlong), ("descriptor", ctypes.c_void_p)]


def get_ranked_memref_descriptor(tensor: torch.Tensor):
    """Returns a ranked memref descriptor for the given numpy array."""
    if tensor.ndim == 0:
        x = make_zero_d_memref_descriptor(tensor.dtype)()
        x.allocated = tensor.data_ptr()
        x.aligned = ctypes.cast(
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.POINTER(dtype_to_ctype(tensor.dtype)),
        )
        x.offset = ctypes.c_longlong(0)
        return x

    x = make_nd_memref_descriptor(tensor.ndim, tensor.dtype)()
    x.allocated = tensor.data_ptr()
    x.aligned = ctypes.cast(
        ctypes.c_void_p(tensor.data_ptr()), ctypes.POINTER(dtype_to_ctype(tensor.dtype))
    )
    x.offset = ctypes.c_longlong(0)

    shape_ctype_t = ctypes.c_longlong * tensor.ndim
    x.shape = shape_ctype_t(*tensor.shape)

    # Numpy uses byte quantities to express strides, MLIR OTOH uses the
    # torch abstraction which specifies strides in terms of elements.
    strides_ctype_t = ctypes.c_longlong * tensor.ndim
    x.strides = strides_ctype_t(*tensor.stride())
    return x
