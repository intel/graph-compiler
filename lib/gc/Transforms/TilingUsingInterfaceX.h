//===- TileUsingInterface.h - Tiling ops using TilingInterface --*- C++ -*-===//
//
// This file is only temporarily used to extend upstream or upcoming utility in
// TilingInterface, which finally aims for upstream.
//
//===----------------------------------------------------------------------===//

#ifndef TEMPORARY_TILEUSINGINTERFACE_X_H
#define TEMPORARY_TILEUSINGINTERFACE_X_H

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

namespace mlir {
namespace scfX {
// Extension for upstream `tileAndFuseProducerOfSlice`
std::optional<scf::SCFFuseProducerOfSliceResult>
tileAndFuseProducerOfSlice(RewriterBase &rewriter,
                           tensor::ExtractSliceOp candidateSliceOp,
                           MutableArrayRef<LoopLikeOpInterface> loops);

// Extension for upcoming upstream `tileAndFuseConsumerOfSlice`
FailureOr<scf::SCFFuseConsumerOfSliceResult>
tileAndFuseConsumerOfSlice(RewriterBase &rewriter, Operation *candidateSliceOp);

SmallVector<LoopLikeOpInterface> getOuterLoopsOfSliceOp(
    OffsetSizeAndStrideOpInterface sliceOp,
    std::optional<LoopLikeOpInterface> untilLoop = std::nullopt);

/** Get the Result of top-level Loop which yield the target InsertSliceOp
 *
 * %1 = scf.for
 *  %2 = scf.for
 *   %3 = scf.for
 *      ...
 *      %4 = insert
 *      yield %4
 *   %5 = insert %3
 *   yield %5
 *  yield %2
 *
 * @param targetSliceOp: %4 = insert
 * @return Result Value: %1
 *         Collected insertSliceOp List during walk including targetSliceOp:
 *                %4 = insert and %5 = insert %3
 */
FailureOr<std::pair<Value, SmallVector<OffsetSizeAndStrideOpInterface>>>
getResultOfTopLevelLoopYieldInsertSliceOp(
    OffsetSizeAndStrideOpInterface targetSliceOp, int curDepth = 0,
    int maxDepth = 5);
} // namespace scfX
} // namespace mlir

#endif
