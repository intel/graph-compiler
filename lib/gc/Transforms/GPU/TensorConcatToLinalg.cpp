//===- TensorConcatToLinalg.cpp - Eliminate tensor.concat -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gc {
#define GEN_PASS_DECL_TENSORCONCATTOLINALG
#define GEN_PASS_DEF_TENSORCONCATTOLINALG
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

using namespace mlir;

namespace {

// Eliminate concat by replacing producers outputs with strided subviews of the
// concat's output.
//
// Matches:
//   %i1 = linalg.* outs(%empty_i = tensor.empty()) ...
//   %i2 = linalg.* outs(%empty_i = tensor.empty()) ...
//   %c   = tensor.concat dim(d) %i1, %i2, ...
//
// Replaces with:
//   %sub_i = memref.subview %concat_out[offset_i...][size_i...][1...]
//   %s_i   = bufferization.to_tensor %sub_i restrict writable
//   %r_i   = linalg.* outs(%s_i) ...
//   bufferization.materialize_in_destination %r_i in %sub_i
struct RewriteLinalgOut : public OpRewritePattern<tensor::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ConcatOp concat,
                                PatternRewriter &rw) const override {
    auto inputs = concat.getInputs();
    for (auto input : inputs) {
      auto linalgOp = input.getDefiningOp<linalg::LinalgOp>();
      if (!linalgOp) return failure();
      auto outs = linalgOp.getDpsInits();
      if (outs.size() != 1 || !outs[0].getDefiningOp<tensor::EmptyOp>())
        return failure();
      if (llvm::any_of(cast<RankedTensorType>(input.getType()).getShape(),
                       ShapedType::isDynamic))
        return failure();
    }

    auto concatDim = concat.getDim();
    auto resultType = cast<RankedTensorType>(concat.getResult().getType());
    auto rank = resultType.getRank();
    auto loc = concat.getLoc();
    auto linalgOps = llvm::map_to_vector(
        inputs, [](Value v) { return v.getDefiningOp<linalg::LinalgOp>(); });

    // If concat feeds a materialize_in_destination, reuse its memref.
    bufferization::MaterializeInDestinationOp mat;
    for (auto *user : concat->getUsers())
      if (auto m = dyn_cast<bufferization::MaterializeInDestinationOp>(user))
        if (dyn_cast<TypedValue<MemRefType>>(m.getDest())) {
          mat = m;
          break;
        }

    Value mem;
    if (mat) {
      mem = mat.getDest();
    } else {
      Operation *first = inputs.front().getDefiningOp();
      for (auto input : inputs.drop_front()) {
        if (input.getDefiningOp()->isBeforeInBlock(first))
          first = input.getDefiningOp();
      }
      rw.setInsertionPoint(first);
      auto memType =
          MemRefType::get(resultType.getShape(), resultType.getElementType());
      mem = memref::AllocOp::create(rw, loc, memType);
    }

    int64_t offset = 0;
    for (auto input : inputs) {
      linalg::LinalgOp op = input.getDefiningOp<linalg::LinalgOp>();
      auto sliceType = cast<RankedTensorType>(input.getType());
      auto sliceShape = sliceType.getShape();
      SmallVector<OpFoldResult> offsets(rank, rw.getIndexAttr(0));
      SmallVector<OpFoldResult> strides(rank, rw.getIndexAttr(1));
      SmallVector<OpFoldResult> sizes = llvm::to_vector(
          llvm::map_range(sliceShape, [&](int64_t s) -> OpFoldResult {
            return rw.getIndexAttr(s);
          }));
      offsets[concatDim] = rw.getIndexAttr(offset);
      offset += sliceShape[concatDim];

      rw.setInsertionPoint(op);
      auto subview =
          memref::SubViewOp::create(rw, loc, mem, offsets, sizes, strides);
      auto subTensor = bufferization::ToTensorOp::create(
          rw, loc, sliceType, subview, /*restrict=*/true, /*writable=*/true);
      rw.modifyOpInPlace(op, [&]() {
        op->replaceUsesOfWith(op.getDpsInits()[0].getDefiningOp()->getResult(0),
                              subTensor.getResult());
      });

      rw.setInsertionPointAfter(op);
      bufferization::MaterializeInDestinationOp::create(rw, loc, TypeRange{},
                                                        input, subview)
          .setWritable(true);
    }

    if (mat) {
      rw.eraseOp(mat);
      rw.eraseOp(concat);
    } else {
      auto result = bufferization::ToTensorOp::create(rw, loc, resultType, mem,
                                                      /*restrict=*/true,
                                                      /*writable=*/true);
      rw.replaceOp(concat, result);
    }
    return success();
  }
};

// Convert tensor.concat to linalg.copy into strided subviews of the
// concat's output.
struct ConcatToLinalgCopy : public OpRewritePattern<tensor::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ConcatOp concat,
                                PatternRewriter &rw) const override {
    auto inputs = concat.getInputs();
    auto concatDim = concat.getDim();
    auto resultType = cast<RankedTensorType>(concat.getResult().getType());
    auto rank = resultType.getRank();
    auto loc = concat.getLoc();

    bufferization::MaterializeInDestinationOp mat;
    if (concat->hasOneUse())
      if (auto m = dyn_cast<bufferization::MaterializeInDestinationOp>(
              concat->use_begin()->getOwner()))
        if (dyn_cast<TypedValue<MemRefType>>(m.getDest())) mat = m;

    Value mem;
    if (mat) { // Reuse the memref from the materialize_in_destination.
      mem = mat.getDest();
    } else {
      rw.setInsertionPoint(concat);
      auto memType =
          MemRefType::get(resultType.getShape(), resultType.getElementType());
      mem = memref::AllocOp::create(rw, loc, memType);
    }

    int64_t offset = 0;
    for (auto input : inputs) {
      auto type = cast<RankedTensorType>(input.getType());
      auto shape = type.getShape();
      if (llvm::any_of(shape, ShapedType::isDynamic)) return failure();
      SmallVector<OpFoldResult> offsets(rank, rw.getIndexAttr(0));
      SmallVector<OpFoldResult> strides(rank, rw.getIndexAttr(1));
      SmallVector<OpFoldResult> sizes = llvm::to_vector(
          llvm::map_range(shape, [&](int64_t s) -> OpFoldResult {
            return rw.getIndexAttr(s);
          }));
      offsets[concatDim] = rw.getIndexAttr(offset);
      offset += shape[concatDim];

      auto subview =
          memref::SubViewOp::create(rw, loc, mem, offsets, sizes, strides);
      auto outTensor = bufferization::ToTensorOp::create(
          rw, loc, type, subview, /*restrict=*/true, /*writable=*/true);
      auto copy = linalg::CopyOp::create(rw, loc, input, outTensor.getResult());
      bufferization::MaterializeInDestinationOp::create(
          rw, loc, TypeRange{}, copy.getResult(0), subview)
          .setWritable(true);
    }

    if (mat) {
      rw.eraseOp(mat);
      rw.eraseOp(concat);
    } else {
      auto result = bufferization::ToTensorOp::create(rw, loc, resultType, mem,
                                                      /*restrict=*/true,
                                                      /*writable=*/true);
      rw.replaceOp(concat, result);
    }
    return success();
  }
};

struct TensorConcatToLinalg final
    : gc::impl::TensorConcatToLinalgBase<TensorConcatToLinalg> {
  void runOnOperation() override {
    auto fn = getOperation();
    if (fn.isExternal()) {
      return;
    }
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<RewriteLinalgOut>(ctx); // Try to eliminate concat first.
    patterns.add<ConcatToLinalgCopy>(ctx, /*benefit=*/0);
    if (failed(applyPatternsGreedily(fn, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
