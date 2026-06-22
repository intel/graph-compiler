//===-- ApplyPaddingLevel.cpp ------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"
#include "gc/Transforms/TensorMaskingOpInterface.h"
#include "gc/Utils/Transform.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_APPLYPADDINGLEVEL
#define GEN_PASS_DEF_APPLYPADDINGLEVEL
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

static bool isSafeToPadWithZeros(TilingInterface op) {
  if (isa<linalg::ContractionOpInterface>(op.getOperation())) return true;
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation()))
    if (linalgOp.getNumReductionLoops() == 0) return true;
  return false;
}

static FailureOr<SmallVector<OpFoldResult>>
getPadMultiples(OpBuilder &b, TilingInterface tilingOp) {
  auto wgAttr = tilingOp->getAttrOfType<DenseI64ArrayAttr>(
      mlir::gc::GC_ATTR_WG_TILE_SIZES);
  if (!wgAttr) return failure();

  ArrayRef<int64_t> tileSizes = wgAttr.asArrayRef();
  SmallVector<int64_t> padSizes(tileSizes.begin(), tileSizes.end());

  return getAsIndexOpFoldResult(b.getContext(), padSizes);
}

static LogicalResult applyPadding(IRRewriter &rewriter,
                                  TilingInterface tilingOp) {
  FailureOr<SmallVector<OpFoldResult>> padMultiples =
      getPadMultiples(rewriter, tilingOp);
  if (failed(padMultiples)) return failure();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(tilingOp);

  if (auto maskingOp =
          dyn_cast<gc::TensorMaskingOpInterface>(tilingOp.getOperation())) {
    FailureOr<SmallVector<Value>> result =
        maskingOp.getMaskedImplementation(rewriter, *padMultiples);
    if (failed(result)) return failure();
    rewriter.replaceOp(tilingOp, result.value());
    return success();
  }

  if (isSafeToPadWithZeros(tilingOp)) {
    linalg::PadTilingInterfaceOptions options =
        linalg::PadTilingInterfaceOptions()
            .setPaddingSizes(*padMultiples)
            .setPadToMultipleOf(false);

    FailureOr<linalg::PadTilingInterfaceResult> result =
        linalg::rewriteAsPaddedOp(rewriter, tilingOp, options);
    if (failed(result)) return failure();
    rewriter.replaceOp(tilingOp, result->replacements);
    return success();
  }

  return failure();
}

struct ApplyPaddingLevel final
    : gc::impl::ApplyPaddingLevelBase<ApplyPaddingLevel> {

  void runOnOperation() override {
    auto funcOp = getOperation();

    SmallVector<TilingInterface> targets;
    funcOp->walk([&](TilingInterface op) {
      if (op->hasAttrOfType<DenseI64ArrayAttr>(gc::GC_ATTR_WG_TILE_SIZES))
        targets.push_back(op);
    });

    IRRewriter rewriter(funcOp);

    for (TilingInterface op : targets) {
      if (op->getBlock() == nullptr) continue;
      (void)applyPadding(rewriter, op);
    }
  }
};

} // namespace
