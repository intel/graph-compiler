//===-- Dialects.cpp - DESC -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc-c/Dialects.h"
#include "gc/Dialect/CPURuntime/IR/CPURuntimeDialect.h"
#include "mlir/CAPI/Registration.h"

#ifdef GC_HAS_ONEDNN_DIALECT
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(OneDNNGraph, onednn_graph,
                                      mlir::onednn_graph::OneDNNGraphDialect)
#endif

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(CPURuntime, cpuruntime,
                                      mlir::cpuruntime::CPURuntimeDialect)