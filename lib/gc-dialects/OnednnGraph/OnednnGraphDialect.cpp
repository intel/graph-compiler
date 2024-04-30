//===- OnednnGraphDialect.h - OneDNN input dialect --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc-dialects/OnednnGraph/OnednnGraphDialect.h"
#include "gc-dialects/OnednnGraph/OnednnGraphOps.h"

using namespace mlir;
using namespace mlir::onednn_graph;

void OnednnGraphDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "gc-dialects/OnednnGraph/OnednnGraphOps.cpp.inc"
      >();
}
