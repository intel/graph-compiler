//===-- HoistStaticForInit.cpp -----------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/IndexingMapOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hoist-static-for-init"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_HOISTSTATICFORINIT
#define GEN_PASS_DEF_HOISTSTATICFORINIT
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

//===----------------------------------------------------------------------===//
// Shared utilities
//===----------------------------------------------------------------------===//

static bool allZero(ArrayRef<OpFoldResult> ofrs) {
  return llvm::all_of(ofrs, [](OpFoldResult ofr) {
    return getConstantIntValue(ofr) == static_cast<int64_t>(0);
  });
}

static bool allOne(ArrayRef<OpFoldResult> ofrs) {
  return llvm::all_of(ofrs, [](OpFoldResult ofr) {
    return getConstantIntValue(ofr) == static_cast<int64_t>(1);
  });
}

static tensor::ExtractSliceOp matchStaticSliceYield(Value yieldOperand) {
  auto sliceOp = yieldOperand.getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp) return nullptr;
  auto srcTy = cast<RankedTensorType>(sliceOp.getSource().getType());
  if (!srcTy.hasStaticShape()) return nullptr;
  if (!allZero(sliceOp.getMixedOffsets()) || !allOne(sliceOp.getMixedStrides()))
    return nullptr;
  return sliceOp;
}

static Value buildPostExtractSlice(IRRewriter &rewriter, Location loc,
                                   Value staticResult,
                                   tensor::ExtractSliceOp templateSlice) {
  int64_t rank = cast<RankedTensorType>(staticResult.getType()).getRank();
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  return tensor::ExtractSliceOp::create(rewriter, loc, staticResult, offsets,
                                        templateSlice.getMixedSizes(), strides);
}

//===----------------------------------------------------------------------===//
// scf.if hoisting
//===----------------------------------------------------------------------===//

static bool matchIfResultForHoist(scf::IfOp ifOp, unsigned resultIdx,
                                  tensor::ExtractSliceOp &thenSlice,
                                  tensor::ExtractSliceOp &elseSlice) {
  auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
  auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());

  thenSlice = matchStaticSliceYield(thenYield.getOperand(resultIdx));
  if (!thenSlice) return false;
  elseSlice = matchStaticSliceYield(elseYield.getOperand(resultIdx));
  if (!elseSlice) return false;

  auto thenSrcTy = cast<RankedTensorType>(thenSlice.getSource().getType());
  auto elseSrcTy = cast<RankedTensorType>(elseSlice.getSource().getType());
  return thenSrcTy == elseSrcTy;
}

static bool hoistIfResults(scf::IfOp ifOp, IRRewriter &rewriter) {
  if (!ifOp.elseBlock()) return false;

  unsigned numResults = ifOp.getNumResults();
  if (numResults == 0) return false;

  SmallVector<std::pair<tensor::ExtractSliceOp, tensor::ExtractSliceOp>>
      slicePairs(numResults, {nullptr, nullptr});
  bool anyHoistable = false;

  for (unsigned i = 0; i < numResults; ++i) {
    auto resultTy = dyn_cast<RankedTensorType>(ifOp.getResult(i).getType());
    if (!resultTy || resultTy.hasStaticShape()) continue;
    tensor::ExtractSliceOp thenSlice, elseSlice;
    if (matchIfResultForHoist(ifOp, i, thenSlice, elseSlice)) {
      slicePairs[i] = {thenSlice, elseSlice};
      anyHoistable = true;
    }
  }
  if (!anyHoistable) return false;

  SmallVector<Type> newResultTypes;
  for (unsigned i = 0; i < numResults; ++i) {
    if (slicePairs[i].first)
      newResultTypes.push_back(slicePairs[i].first.getSource().getType());
    else newResultTypes.push_back(ifOp.getResult(i).getType());
  }

  rewriter.setInsertionPoint(ifOp);
  auto newIf = scf::IfOp::create(rewriter, ifOp.getLoc(), newResultTypes,
                                 ifOp.getCondition(), /*withElseRegion=*/true);

  // Clone then-block.
  {
    rewriter.setInsertionPointToStart(newIf.thenBlock());
    IRMapping mapping;
    auto oldYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    for (Operation &op : ifOp.thenBlock()->without_terminator())
      rewriter.clone(op, mapping);
    SmallVector<Value> yieldVals;
    for (unsigned i = 0; i < numResults; ++i) {
      if (slicePairs[i].first)
        yieldVals.push_back(
            mapping.lookupOrDefault(slicePairs[i].first.getSource()));
      else yieldVals.push_back(mapping.lookupOrDefault(oldYield.getOperand(i)));
    }
    scf::YieldOp::create(rewriter, ifOp.getLoc(), yieldVals);
  }

  // Clone else-block.
  {
    rewriter.setInsertionPointToStart(newIf.elseBlock());
    IRMapping mapping;
    auto oldYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    for (Operation &op : ifOp.elseBlock()->without_terminator())
      rewriter.clone(op, mapping);
    SmallVector<Value> yieldVals;
    for (unsigned i = 0; i < numResults; ++i) {
      if (slicePairs[i].second)
        yieldVals.push_back(
            mapping.lookupOrDefault(slicePairs[i].second.getSource()));
      else yieldVals.push_back(mapping.lookupOrDefault(oldYield.getOperand(i)));
    }
    scf::YieldOp::create(rewriter, ifOp.getLoc(), yieldVals);
  }

  rewriter.setInsertionPointAfter(newIf);
  SmallVector<Value> replacements;
  for (unsigned i = 0; i < numResults; ++i) {
    if (slicePairs[i].first) {
      Value slice = buildPostExtractSlice(
          rewriter, ifOp.getLoc(), newIf.getResult(i), slicePairs[i].first);
      replacements.push_back(slice);
    } else {
      replacements.push_back(newIf.getResult(i));
    }
  }

  rewriter.replaceOp(ifOp, replacements);
  return true;
}

//===----------------------------------------------------------------------===//
// scf.for hoisting
//===----------------------------------------------------------------------===//

struct ForHoistInfo {
  unsigned argIdx;
  RankedTensorType staticTy;
  Value fillVal;
  Value staticSrc;
  SmallVector<Value> dynDims;
  Operation *padOp;
  Operation *sliceOp;
  Operation *insertSliceOp;
};

static bool matchFillOnEmpty(Value val, Value &fillScalar,
                             SmallVectorImpl<Value> &dynDims) {
  auto matchOutput = [&](Value output) -> bool {
    auto emptyOp = output.getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp) return false;
    RankedTensorType ty = cast<RankedTensorType>(val.getType());
    if ((size_t)ty.getNumDynamicDims() != emptyOp.getDynamicSizes().size())
      return false;
    dynDims.assign(emptyOp.getDynamicSizes().begin(),
                   emptyOp.getDynamicSizes().end());
    return true;
  };

  if (auto fillOp = val.getDefiningOp<linalg::FillOp>()) {
    if (fillOp.getOutputs().size() != 1) return false;
    if (!matchOutput(fillOp.getOutputs()[0])) return false;
    fillScalar = fillOp.getInputs()[0];
    return true;
  }

  if (auto genericOp = val.getDefiningOp<linalg::GenericOp>()) {
    if (genericOp.getOutputs().size() != 1) return false;
    if (genericOp.getInputs().size() != 1) return false;
    if (!llvm::all_of(genericOp.getIteratorTypesArray(),
                      [](utils::IteratorType t) {
                        return t == utils::IteratorType::parallel;
                      }))
      return false;
    Block &body = genericOp.getRegion().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) return false;
    if (yieldOp.getOperand(0) != body.getArgument(0)) return false;
    Value inputVal = genericOp.getInputs()[0];
    AffineMap inMap = genericOp.getIndexingMapsArray()[0];
    if (inMap.getNumResults() != 0) return false;
    if (!matchOutput(genericOp.getOutputs()[0])) return false;
    fillScalar = inputVal;
    return true;
  }

  return false;
}

static bool matchSliceOfStatic(Value val, Value &staticSrc,
                               SmallVectorImpl<Value> &dynDims,
                               RankedTensorType expectedStaticTy) {
  auto sliceOp = val.getDefiningOp<tensor::ExtractSliceOp>();
  if (!sliceOp) return false;
  auto srcTy = cast<RankedTensorType>(sliceOp.getSource().getType());
  if (!srcTy.hasStaticShape()) return false;
  if (srcTy != expectedStaticTy) return false;
  if (!allZero(sliceOp.getMixedOffsets()) || !allOne(sliceOp.getMixedStrides()))
    return false;
  staticSrc = sliceOp.getSource();
  for (OpFoldResult sz : sliceOp.getMixedSizes()) {
    if (auto v = dyn_cast<Value>(sz)) dynDims.push_back(v);
  }
  return true;
}

static tensor::PadOp findUniquePadOfArg(BlockArgument blockArg) {
  tensor::PadOp result = nullptr;

  SmallVector<Value> candidates;
  candidates.push_back(blockArg);

  for (Operation *user : blockArg.getUsers()) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    if (!sliceOp) continue;
    if (sliceOp.getResult().getType() != blockArg.getType()) continue;
    if (!allZero(sliceOp.getMixedOffsets()) ||
        !allOne(sliceOp.getMixedStrides()))
      continue;
    candidates.push_back(sliceOp.getResult());
  }

  for (Value candidate : candidates) {
    for (Operation *user : candidate.getUsers()) {
      auto padOp = dyn_cast<tensor::PadOp>(user);
      if (!padOp) continue;
      if (!padOp.getResultType().hasStaticShape()) continue;
      if (!allZero(padOp.getMixedLowPad())) continue;
      if (result) return nullptr;
      result = padOp;
    }
  }
  return result;
}

static bool isHoistingSafe(scf::ForOp forOp, const ForHoistInfo &info) {
  RankedTensorType dynTy =
      cast<RankedTensorType>(forOp.getRegionIterArgs()[info.argIdx].getType());

  SmallVector<int64_t> paddedTensorDims;
  for (int64_t i = 0; i < (int64_t)dynTy.getRank(); ++i)
    if (ShapedType::isDynamic(dynTy.getDimSize(i)))
      paddedTensorDims.push_back(i);

  if (paddedTensorDims.empty()) return true;

  llvm::DenseSet<Value> visited;
  SmallVector<Value> worklist;
  Value padResult = info.padOp->getResult(0);
  worklist.push_back(padResult);
  visited.insert(padResult);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (user->getParentOp() != forOp.getOperation()) continue;
      if (user == info.sliceOp) continue;
      if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(user)) continue;
      if (isa<tensor::DimOp, memref::DimOp>(user)) continue;
      if (isa<scf::YieldOp>(user)) continue;

      auto indexingOp = dyn_cast<IndexingMapOpInterface>(user);
      auto tilingOp = dyn_cast<TilingInterface>(user);
      if (!indexingOp || !tilingOp) {
        LLVM_DEBUG(llvm::dbgs() << "[hoist-safety]   op " << user->getName()
                                << " lacks required interfaces -> reject\n");
        return false;
      }

      OpOperand *matchedOperand = nullptr;
      for (OpOperand &operand : user->getOpOperands()) {
        if (operand.get() == v) {
          matchedOperand = &operand;
          break;
        }
      }
      if (!matchedOperand) continue;

      AffineMap map = indexingOp.getMatchingIndexingMap(matchedOperand);
      SmallVector<utils::IteratorType> iterTypes;
      for (auto attr : tilingOp.getLoopIteratorTypes())
        iterTypes.push_back(attr);

      for (int64_t paddedDim : paddedTensorDims) {
        if (paddedDim >= (int64_t)map.getNumResults()) return false;
        AffineExpr expr = map.getResult(paddedDim);
        auto dimExpr = dyn_cast<AffineDimExpr>(expr);
        if (!dimExpr) return false;
        unsigned iterDim = dimExpr.getPosition();
        if (iterTypes[iterDim] != utils::IteratorType::parallel) {
          LLVM_DEBUG(llvm::dbgs() << "[hoist-safety]   op " << user->getName()
                                  << " reduction on padded dim -> reject\n");
          return false;
        }
      }

      for (Value result : user->getResults())
        if (visited.insert(result).second) worklist.push_back(result);
    }
  }
  return true;
}

static SmallVector<ForHoistInfo> collectForHoistInfo(scf::ForOp forOp) {
  SmallVector<ForHoistInfo> infos;
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

  for (auto [idx, initArg, blockArg] :
       llvm::enumerate(forOp.getInitArgs(), forOp.getRegionIterArgs())) {

    auto dynTy = dyn_cast<RankedTensorType>(blockArg.getType());
    if (!dynTy || dynTy.hasStaticShape()) continue;

    tensor::PadOp padOp = findUniquePadOfArg(blockArg);
    if (!padOp) continue;

    RankedTensorType staticTy = padOp.getResultType();

    Value fillScalar;
    Value staticSrc;
    SmallVector<Value> dynDims;
    bool initMatched = matchFillOnEmpty(initArg, fillScalar, dynDims);
    if (!initMatched)
      initMatched = matchSliceOfStatic(initArg, staticSrc, dynDims, staticTy);
    if (!initMatched) continue;

    Value yieldOperand = yieldOp.getOperand(idx);
    Operation *sliceOpRaw = nullptr;
    Operation *yieldInsertSlice = nullptr;
    if (auto extractOp = yieldOperand.getDefiningOp<tensor::ExtractSliceOp>()) {
      if (cast<RankedTensorType>(extractOp.getSource().getType()) == staticTy &&
          allZero(extractOp.getMixedOffsets()) &&
          allOne(extractOp.getMixedStrides())) {
        sliceOpRaw = extractOp.getOperation();
      }
    } else if (auto insertOp =
                   yieldOperand.getDefiningOp<tensor::InsertSliceOp>()) {
      if (insertOp.getDest() == forOp.getRegionIterArgs()[idx] &&
          allZero(insertOp.getMixedOffsets()) &&
          allOne(insertOp.getMixedStrides())) {
        if (auto srcExtract =
                insertOp.getSource().getDefiningOp<tensor::ExtractSliceOp>()) {
          if (cast<RankedTensorType>(srcExtract.getSource().getType()) ==
                  staticTy &&
              allZero(srcExtract.getMixedOffsets()) &&
              allOne(srcExtract.getMixedStrides())) {
            sliceOpRaw = srcExtract.getOperation();
            yieldInsertSlice = insertOp.getOperation();
          }
        }
      }
    }
    if (!sliceOpRaw) continue;

    bool compatible = true;
    for (auto [s, d] : llvm::zip(staticTy.getShape(), dynTy.getShape())) {
      if (!ShapedType::isDynamic(d) && s < d) {
        compatible = false;
        break;
      }
    }
    if (!compatible) continue;

    ForHoistInfo candidate{(unsigned)idx,      staticTy,
                           fillScalar,         staticSrc,
                           std::move(dynDims), padOp.getOperation(),
                           sliceOpRaw,         yieldInsertSlice};

    if (!isHoistingSafe(forOp, candidate)) continue;

    LLVM_DEBUG(llvm::dbgs() << "[hoist] ACCEPTED for iter_arg " << idx << "\n");
    infos.push_back(std::move(candidate));
  }
  return infos;
}

static void hoistForToStatic(scf::ForOp forOp, ArrayRef<ForHoistInfo> infos,
                             IRRewriter &rewriter) {
  Location loc = forOp.getLoc();
  rewriter.setInsertionPoint(forOp);

  llvm::SmallDenseSet<unsigned> transformedIdxs;
  for (auto &info : infos) transformedIdxs.insert(info.argIdx);

  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                 forOp.getInitArgs().end());
  for (auto &info : infos) {
    if (info.staticSrc) {
      newInitArgs[info.argIdx] = info.staticSrc;
    } else {
      Value staticEmpty =
          tensor::EmptyOp::create(rewriter, loc, info.staticTy.getShape(),
                                  info.staticTy.getElementType());
      Value staticFill =
          linalg::FillOp::create(rewriter, loc, info.fillVal, staticEmpty)
              ->getResult(0);
      newInitArgs[info.argIdx] = staticFill;
    }
  }

  auto emptyBuilder = [](OpBuilder &, Location, Value, ValueRange) {};
  auto newFor = scf::ForOp::create(rewriter, loc, forOp.getLowerBound(),
                                   forOp.getUpperBound(), forOp.getStep(),
                                   newInitArgs, emptyBuilder);

  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newFor.getInductionVar());
  for (auto [oldArg, newArg] :
       llvm::zip(forOp.getRegionIterArgs(), newFor.getRegionIterArgs()))
    mapping.map(oldArg, newArg);

  for (auto &info : infos) {
    BlockArgument newArg = newFor.getRegionIterArgs()[info.argIdx];
    mapping.map(info.padOp->getResult(0), newArg);
  }

  rewriter.setInsertionPointToEnd(newFor.getBody());
  for (Operation &op : forOp.getBody()->without_terminator()) {
    bool skip = llvm::any_of(infos, [&op](const ForHoistInfo &i) {
      return &op == i.padOp || &op == i.insertSliceOp;
    });
    if (!skip) rewriter.clone(op, mapping);
  }

  auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<Value> newYieldOperands;
  for (auto [idx, operand] : llvm::enumerate(oldYield.getOperands())) {
    if (transformedIdxs.contains(idx)) {
      for (auto &info : infos) {
        if (info.argIdx == idx) {
          Value sliceSrc = info.sliceOp->getOperand(0);
          newYieldOperands.push_back(mapping.lookup(sliceSrc));
          break;
        }
      }
    } else {
      newYieldOperands.push_back(mapping.lookupOrDefault(operand));
    }
  }
  scf::YieldOp::create(rewriter, loc, newYieldOperands);

  rewriter.setInsertionPointAfter(newFor);
  SmallVector<Value> replacements;
  for (auto [idx, oldResult] : llvm::enumerate(forOp.getResults())) {
    Value newResult = newFor.getResult(idx);
    if (transformedIdxs.contains(idx)) {
      for (auto &info : infos) {
        if (info.argIdx == idx) {
          RankedTensorType srcTy = info.staticTy;
          int64_t rank = srcTy.getRank();
          SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
          SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
          SmallVector<OpFoldResult> sizes;
          unsigned dynDimIdx = 0;
          for (int64_t r = 0; r < rank; ++r) {
            if (ShapedType::isDynamic(
                    cast<RankedTensorType>(oldResult.getType()).getShape()[r]))
              sizes.push_back(info.dynDims[dynDimIdx++]);
            else sizes.push_back(rewriter.getIndexAttr(srcTy.getShape()[r]));
          }
          Value slice = tensor::ExtractSliceOp::create(rewriter, loc, newResult,
                                                       offsets, sizes, strides);
          replacements.push_back(slice);
          break;
        }
      }
    } else {
      replacements.push_back(newResult);
    }
  }

  rewriter.replaceOp(forOp, replacements);
}

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

struct HoistStaticForInit final
    : gc::impl::HoistStaticForInitBase<HoistStaticForInit> {

  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(funcOp->getContext());

    SmallVector<scf::IfOp> ifOps;
    funcOp->walk([&](scf::IfOp ifOp) { ifOps.push_back(ifOp); });
    for (scf::IfOp ifOp : ifOps) hoistIfResults(ifOp, rewriter);

    SmallVector<scf::ForOp> loops;
    funcOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
    for (scf::ForOp forOp : loops) {
      SmallVector<ForHoistInfo> infos = collectForHoistInfo(forOp);
      if (infos.empty()) continue;
      hoistForToStatic(forOp, infos, rewriter);
    }
  }
};

} // namespace
