//===-- BufferizableOpInterfaceImpl.cpp - linalgx bufferize -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Linalgx/Transforms/BufferizableOpInterfaceImpl.h"
#include "gc/Dialect/Linalgx/IR/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/IR/LinalgxOps.h"

//===----------------------------------------------------------------------===//
// Builder helper from Linalg/Transforms/BufferizableOpInterfaceImpl.cpp
//===----------------------------------------------------------------------===//

#include "BufferizableOpInterfaceImpl.cpp.inc"

using namespace mlir;
using namespace mlir::linalgx;

void mlir::linalgx::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, linalgx::LinalgxDialect *dialect) {
        // Register all Linalg structured ops. `LinalgOp` is an interface and it
        // is not possible to attach an external interface to an existing
        // interface. Therefore, attach the `BufferizableOpInterface` to all ops
        // one-by-one.
        LinalgOpInterfaceHelper<
#define GET_OP_LIST
#include "gc/Dialect/Linalgx/IR/LinalgxStructuredOps.cpp.inc"
            >::registerOpInterface(ctx);
      });
}
