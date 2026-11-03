//===- NormalizeMemrefLayouts.cpp - Row-major views of memrefs --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/VectorInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_NORMALIZEMEMREFLAYOUTS
#define GEN_PASS_DEF_NORMALIZEMEMREFLAYOUTS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

/// Return the permutation of the dims of `type` that gives it a row-major
/// layout, or std::nullopt if the type already has a unit innermost stride or
/// cannot be made row-major by reordering its dims. `perm[i]` is the dim of
/// `type` that becomes dim `i` of the permuted type.
///
/// The dim with unit stride becomes the innermost one; that is all the 2D
/// block accesses need. The remaining dims are sorted by decreasing stride when
/// every stride is static, which yields the unique row-major order. When some
/// stride is dynamic they keep their order.
static std::optional<SmallVector<int64_t>>
getRowMajorPermutation(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset))) return std::nullopt;
  int64_t rank = strides.size();
  if (rank < 2 || strides.back() == 1) return std::nullopt;

  // The last dim with a unit stride; several only occur with unit-size dims.
  int64_t unitDim = -1;
  for (int64_t d = rank - 1; d >= 0; --d) {
    if (strides[d] == 1) {
      unitDim = d;
      break;
    }
  }
  if (unitDim < 0) return std::nullopt;

  SmallVector<int64_t> perm;
  for (int64_t d = 0; d < rank; ++d)
    if (d != unitDim) perm.push_back(d);
  if (llvm::none_of(strides, ShapedType::isDynamic))
    llvm::stable_sort(
        perm, [&](int64_t a, int64_t b) { return strides[a] > strides[b]; });
  perm.push_back(unitDim);
  return perm;
}

/// Create a row-major view of `memref` by transposing its dims with `perm`,
/// e.g. for perm = [1, 0]:
///
///   memref.transpose %m (d0, d1) -> (d1, d0)
///     : memref<512x2048xf16, strided<[1, 512]>>
///     to memref<2048x512xf16, strided<[512, 1]>>
///
/// The view is inserted right after the definition of `memref`, so that it is
/// shared by all its users and, for a function argument, stays outside of the
/// kernels that read the memref.
static Value createRowMajorView(OpBuilder &builder, Value memref,
                                ArrayRef<int64_t> perm) {
  OpBuilder::InsertionGuard guard(builder);
  if (auto arg = dyn_cast<BlockArgument>(memref))
    builder.setInsertionPointToStart(arg.getOwner());
  else builder.setInsertionPointAfter(memref.getDefiningOp());

  AffineMap permMap = AffineMap::getPermutationMap(perm, builder.getContext());
  return memref::TransposeOp::create(builder, memref.getLoc(), memref,
                                     AffineMapAttr::get(permMap));
}

/// Redirect the transfer to `view`, the row-major view of its source obtained
/// with `perm`. The access stays the same: the indices are reordered along with
/// the dims, and the permutation map is renamed to the dims of the view, e.g.
///
///   %v = vector.transfer_read %m[%n, %k]
///          : memref<512x2048xf16, strided<[1, 512]>>, vector<64x64xf16>
/// becomes
///   %v = vector.transfer_read %view[%k, %n]
///          {permutation_map = affine_map<(d0, d1) -> (d1, d0)>}
///          : memref<2048x512xf16, strided<[512, 1]>>, vector<64x64xf16>
///
/// A transposing read of a row-major memref is what the XeGPU lowering already
/// handles. `in_bounds` is indexed by vector dim and does not change. The mask
/// is shaped in memref-dim order (over the dims the map reads), so explicit
/// transfer masks and enclosing vector.mask masks are transposed with the dims.
static void redirectTransfer(OpBuilder &builder,
                             VectorTransferOpInterface xferOp, Value view,
                             ArrayRef<int64_t> perm) {
  MLIRContext *ctx = xferOp.getContext();
  // Dim `perm[i]` of the source is dim `i` of the view.
  SmallVector<int64_t> viewDim(perm.size());
  for (auto [i, sourceDim] : llvm::enumerate(perm)) viewDim[sourceDim] = i;

  ValueRange indices = xferOp.getIndices();
  SmallVector<Value> viewIndices = llvm::map_to_vector(
      perm, [&](int64_t sourceDim) { return indices[sourceDim]; });
  // The map's inputs are the dims of the memref. Express each source dim by
  // the dim of the view it became.
  AffineMap sourceMap = xferOp.getPermutationMap();
  AffineMap viewMap =
      sourceMap.compose(AffineMap::getPermutationMap(viewDim, ctx));

  Value mask = xferOp.getMask();
  auto maskOp = dyn_cast<vector::MaskOp>(xferOp->getParentOp());
  if (mask || maskOp) {
    // Mask dim `j` is the j-th (in increasing order) source dim the map reads;
    // the new mask dim `i` is the i-th view dim it reads.
    SmallVector<int64_t> readSourceDims, readViewDims;
    for (int64_t sourceDim = 0; sourceDim < (int64_t)perm.size(); ++sourceDim) {
      if (!sourceMap.isFunctionOfDim(sourceDim)) continue;
      readSourceDims.push_back(sourceDim);
      readViewDims.push_back(viewDim[sourceDim]);
    }
    llvm::sort(readViewDims);
    SmallVector<int64_t> transposition =
        llvm::map_to_vector(readViewDims, [&](int64_t dim) -> int64_t {
          return llvm::find(readSourceDims, perm[dim]) - readSourceDims.begin();
        });
    if (!llvm::is_sorted(transposition)) {
      // Keep mask transposes outside the single-operation vector.mask region.
      builder.setInsertionPoint(maskOp ? maskOp.getOperation()
                                       : xferOp.getOperation());
      auto transposeMask = [&](Value value) -> Value {
        return vector::TransposeOp::create(builder, xferOp.getLoc(), value,
                                           transposition);
      };
      if (mask) mask = transposeMask(mask);
      if (maskOp)
        maskOp.getMaskMutable().assign(transposeMask(maskOp.getMask()));
    }
  }

  llvm::TypeSwitch<Operation *>(xferOp.getOperation())
      .Case([&](vector::TransferReadOp readOp) {
        readOp.getBaseMutable().assign(view);
        readOp.getIndicesMutable().assign(viewIndices);
        readOp.setPermutationMap(viewMap);
        if (mask) readOp.getMaskMutable().assign(mask);
      })
      .Case([&](vector::TransferWriteOp writeOp) {
        writeOp.getBaseMutable().assign(view);
        writeOp.getIndicesMutable().assign(viewIndices);
        writeOp.setPermutationMap(viewMap);
        if (mask) writeOp.getMaskMutable().assign(mask);
      });
}

struct NormalizeMemrefLayouts final
    : gc::impl::NormalizeMemrefLayoutsBase<NormalizeMemrefLayouts> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<std::pair<VectorTransferOpInterface, SmallVector<int64_t>>>
        transfers;
    funcOp.walk([&](Operation *op) {
      if (!isa<vector::TransferReadOp, vector::TransferWriteOp>(op)) return;
      auto xferOp = cast<VectorTransferOpInterface>(op);
      auto memrefType = dyn_cast<MemRefType>(xferOp.getShapedType());
      if (!memrefType) return;
      if (auto perm = getRowMajorPermutation(memrefType))
        transfers.emplace_back(xferOp, std::move(*perm));
    });

    // One view per source memref, shared by all the transfers of that memref.
    OpBuilder builder(funcOp.getContext());
    llvm::DenseMap<Value, Value> views;
    for (auto &[xferOp, perm] : transfers) {
      Value source = xferOp.getBase();
      Value &view = views[source];
      if (!view) view = createRowMajorView(builder, source, perm);
      redirectTransfer(builder, xferOp, view, perm);
    }
  }
};

} // namespace
