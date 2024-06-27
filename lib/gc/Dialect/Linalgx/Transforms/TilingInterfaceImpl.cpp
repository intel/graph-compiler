//===-- TilingInterfaceImpl.cpp - linalgx TilingInterface -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Linalgx/Transforms/TilingInterfaceImpl.h"
#include "gc/Dialect/Linalgx/IR/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/IR/LinalgxOps.h"

//===----------------------------------------------------------------------===//
// Builder helper from Linalg/Transforms/TilingInterfaceImpl.cpp
//===----------------------------------------------------------------------===//

#include "TilingInterfaceImpl.cpp.inc"

using namespace mlir;
using namespace mlir::linalgx;

template <typename OpType> static void registerOne(MLIRContext *ctx) {
  OpType::template attachInterface<LinalgOpTilingInterface<OpType>>(*ctx);
  OpType::template attachInterface<LinalgOpPartialReductionInterface<OpType>>(
      *ctx);
}

/// Variadic helper function.
template <typename... OpTypes> static void registerAll(MLIRContext *ctx) {
  (registerOne<OpTypes>(ctx), ...);
}

#define GET_OP_LIST

void mlir::linalgx::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, linalgx::LinalgxDialect *dialect) {
        registerAll<
#include "gc/Dialect/Linalgx/IR/LinalgxStructuredOps.cpp.inc"
            >(ctx);
      });
}
