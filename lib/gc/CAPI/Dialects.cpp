//===-- Dialects.cpp - DESC -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc-c/Dialects.h"
#include "gc/Dialect/CPURuntime/IR/CPURuntimeDialect.h"
#include "gc/Dialect/OneDNNGraph/OneDNNGraphDialect.h"
#include "mlir/CAPI/Registration.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(OneDNNGraph, onednn_graph,
                                      mlir::onednn_graph::OneDNNGraphDialect)

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(CPURuntime, cpuruntime,
                                      mlir::cpuruntime::CPURuntimeDialect)

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Linalgx, linalgx,
                                      mlir::linalgx::LinalgxDialect)