//===-- Passes.cpp - DESC ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"
#include "mlir-c/Pass.h"
#include "mlir/CAPI/Pass.h"

#include "gc/Transforms/Passes.capi.h.inc"
using namespace mlir::gc;

#ifdef __cplusplus
extern "C" {
#endif

#include "gc/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif