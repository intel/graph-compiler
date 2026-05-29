//===---- MemrefCopyToGpuMemcpy.cpp - Convert memref.copy to gpu.memcpy ---===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <algorithm>
#include <cstdint>

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_MEMREFCOPYTOGPU
#define GEN_PASS_DEF_MEMREFCOPYTOGPU
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct MemrefCopyToGpuMemcpy : OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rw) const override {
    auto src = op.getSource();
    auto dst = op.getTarget();
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType = cast<MemRefType>(dst.getType());
    auto rank = srcType.getRank();

    SmallVector<int64_t> srcStrides, dstStrides;
    int64_t srcOffset, dstOffset;
    if (failed(srcType.getStridesAndOffset(srcStrides, srcOffset)) ||
        failed(dstType.getStridesAndOffset(dstStrides, dstOffset)))
      return failure();
    if (ShapedType::isDynamic(srcOffset) || ShapedType::isDynamic(dstOffset))
      return failure();

    auto shape = srcType.getShape();
    auto elemType = srcType.getElementType();

    // Find how many trailing dims are contiguous.
    // Contiguous means stride[i] == stride[i+1] * size[i+1].
    auto isContiguous = [&](ArrayRef<int64_t> strides) -> int64_t {
      int64_t stride = 1;
      for (int64_t i = rank - 1; i >= 0; --i) {
        if (ShapedType::isDynamic(strides[i]) ||
            ShapedType::isDynamic(shape[i]) || strides[i] != stride)
          return i + 1; // first non-contiguous dim from the right
        stride *= shape[i];
      }
      return 0; // fully contiguous
    };

    int splitDim = std::max(isContiguous(srcStrides), isContiguous(dstStrides));
    if (splitDim == 0) { // Fully contiguous — direct gpu.memcpy.
      rw.replaceOpWithNewOp<gpu::MemcpyOp>(op, TypeRange{}, ValueRange{}, dst,
                                           src);
      return success();
    }

    // Partially strided — loop over outer dims, gpu.memcpy each contiguous
    // inner slice.
    auto loc = op.getLoc();
    auto zero = arith::ConstantIndexOp::create(rw, loc, 0);
    auto one = arith::ConstantIndexOp::create(rw, loc, 1);
    SmallVector<Value> lbs(splitDim, zero);
    SmallVector<Value> ubs, steps(splitDim, one);
    for (int i = 0; i < splitDim; ++i) {
      if (ShapedType::isDynamic(shape[i])) return failure();
      ubs.push_back(arith::ConstantIndexOp::create(rw, loc, shape[i]));
    }

    // Inner slice type: trailing (rank - splitDim) dims, dynamic offset.
    auto innerShape = shape.drop_front(splitDim);
    SmallVector<int64_t> innerStrides(innerShape.size());
    int64_t s = 1;
    for (int i = (int)innerShape.size() - 1; i >= 0; --i) {
      innerStrides[i] = s;
      s *= innerShape[i];
    }
    auto innerType = MemRefType::get(
        innerShape, elemType,
        StridedLayoutAttr::get(op.getContext(), ShapedType::kDynamic,
                               innerStrides));

    scf::buildLoopNest(
        rw, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange ivs) {
          // Compute the flat element offset: sum(iv[i] * stride[i]).
          auto slice = [&](Value ref, ArrayRef<int64_t> strides,
                           int64_t baseOffset) -> Value {
            Value offset = arith::ConstantIndexOp::create(b, loc, baseOffset);
            for (int i = 0; i < splitDim; ++i) {
              auto stride = arith::ConstantIndexOp::create(b, loc, strides[i]);
              auto term = arith::MulIOp::create(b, loc, ivs[i], stride);
              offset = arith::AddIOp::create(b, loc, offset, term);
            }
            SmallVector<OpFoldResult> innerSizeFolds, innerStrideFolds;
            for (int64_t sz : innerShape)
              innerSizeFolds.push_back(b.getIndexAttr(sz));
            for (int64_t st : innerStrides)
              innerStrideFolds.push_back(b.getIndexAttr(st));
            return memref::ReinterpretCastOp::create(
                       b, loc, innerType, ref, OpFoldResult(offset),
                       innerSizeFolds, innerStrideFolds)
                .getResult();
          };
          auto srcSlice = slice(src, srcStrides, srcOffset);
          auto dstSlice = slice(dst, dstStrides, dstOffset);
          gpu::MemcpyOp::create(b, loc, TypeRange{}, ValueRange{}, dstSlice,
                                srcSlice);
        });

    rw.eraseOp(op);
    return success();
  }
};

struct MemrefCopyToGpuPass final
    : gc::impl::MemrefCopyToGpuBase<MemrefCopyToGpuPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MemrefCopyToGpuMemcpy>(&getContext());
    if (failed(applyPatternsGreedily(
            getOperation(), FrozenRewritePatternSet(std::move(patterns)))))
      signalPassFailure();
  }
};
} // namespace