//===-- TileUsingInterfaceX.h -  upstream eXtension -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TILE_USING_INTERFACE_X_H
#define TILE_USING_INTERFACE_X_H

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

namespace mlir {
namespace scfX {

SmallVector<LoopLikeOpInterface> getOuterNestLoopsWhile(
    LoopLikeOpInterface loop,
    const std::function<LogicalResult(LoopLikeOpInterface)> &pred);

FailureOr<OpResult> getRealProducerFromExtractSliceOp(
    Operation *candidateSliceOp,
    SmallVector<tensor::ExtractSliceOp> &backwardSlice, unsigned curDepth = 0,
    unsigned maxDepth = 5);

FailureOr<SmallVector<OpOperand *>> getRealConsumersFromInsertSliceOp(
    Operation *candidateSliceOp,
    SmallVector<OffsetSizeAndStrideOpInterface> &forwardSlice,
    unsigned curDepth = 0, unsigned maxDepth = 5);

// Extension for upstream `tileAndFuseProducerOfSlice`
std::optional<scf::SCFFuseProducerOfSliceResult>
tileAndFuseProducerOfSlice(RewriterBase &rewriter, Operation *candidateSliceOp);

// Extension for upcoming upstream `tileAndFuseConsumerOfSlice`
FailureOr<scf::SCFFuseConsumerOfSliceResult>
tileAndFuseConsumerOfSlice(RewriterBase &rewriter, Operation *candidateSliceOp);
} // namespace scfX
} // namespace mlir

#endif
