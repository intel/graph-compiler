//===-- PropagateLayout.cpp - Propagate packing on named ops*- C++ ------*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <numeric>

#include "gc/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"

#include "gc/Analysis/MatmulConfigAnalysis.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Dialect/Linalgx/Utils.h"
#include "gc/Transforms/Passes.h"
namespace mlir {
namespace gc {
#define GEN_PASS_DEF_PROPAGATELAYOUTONNAMEDOPS
#include "gc/Transforms/Passes.h.inc"

#define DEBUG_TYPE "named-op-layout-propagation"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::tensor;

// insert pack when innerPosDims is non-empty
// insert linalg.transpose otherwise
static Value insertLayoutPack(RewriterBase &rewriter, Location loc, Value input,
                              Value dest, ArrayRef<int64_t> innerDimsPos,
                              ArrayRef<OpFoldResult> innerTiles,
                              ArrayRef<int64_t> outerDimsPerm) {
  if (!innerDimsPos.empty())
    return rewriter.create<tensor::PackOp>(
        loc, input, dest, innerDimsPos, innerTiles,
        /*padding=*/std::nullopt, outerDimsPerm);
  if (!TensorLayout::isPlainOuterAxis(outerDimsPerm)) {
    return rewriter.create<linalg::TransposeOp>(loc, input, dest, outerDimsPerm)
        .getResults()[0];
  }
  return input;
}

// insert unpack when innerPosDims is non-empty
// insert linalg.transpose otherwise
static Value insertLayoutUnpack(RewriterBase &rewriter, Location loc,
                                Value input, ArrayRef<int64_t> innerDimsPos,
                                ArrayRef<OpFoldResult> innerTiles,
                                ArrayRef<int64_t> outerDimsPerm) {
  Value dest = tensor::UnPackOp::createDestinationTensor(
      rewriter, loc, input, innerTiles, innerDimsPos, outerDimsPerm);
  if (!innerDimsPos.empty()) {
    return rewriter.create<tensor::UnPackOp>(loc, input, dest, innerDimsPos,
                                             innerTiles, outerDimsPerm);
  }
  if (!TensorLayout::isPlainOuterAxis(outerDimsPerm)) {
    // inverse the permutationVector
    SmallVector<int64_t> permAxes(outerDimsPerm.size());
    for (auto [idx, axis] : llvm::enumerate(outerDimsPerm)) {
      permAxes[axis] = idx;
    }
    return rewriter.create<linalg::TransposeOp>(loc, input, dest, permAxes)
        .getResults()[0];
  }
  return input;
}

static SmallVector<int64_t> getPackedAxes(ArrayRef<int64_t> dimensions,
                                          TensorLayout targetLayout) {
  SmallVector<int64_t> result;
  // permuting on outer axis
  auto outerPerm = targetLayout.getOuterAxis();
  for (size_t i = 0; i < dimensions.size(); ++i) {
    auto pos = std::find(outerPerm.begin(), outerPerm.end(), dimensions[i]);
    assert(pos != outerPerm.end() && "dimension must be within output perm.");
    result.push_back(std::distance(outerPerm.begin(), pos));
  }
  // inserting inner axis
  auto innerPos = targetLayout.getInnerAxis();
  for (size_t i = 0; i < dimensions.size(); ++i) {
    if (std::find(innerPos.begin(), innerPos.end(), dimensions[i]) !=
        innerPos.end()) {
      result.push_back(i + targetLayout.getOuterAxis().size());
    }
  }
  return result;
}

static SmallVector<int64_t> getPackedPermAxes(ArrayRef<int64_t> plainPermAxes,
                                              TensorLayout inputLayout,
                                              TensorLayout outputLayout) {
  // dim(result, i) = dim(input, permutation[i])
  // input: permutation[i] --> output: i
  // input: permutation[i] --> packed input: std::find(permutation[i]) - begin()
  // output: i --> packed output: std::find(permutation[i]) - begin()
  int64_t packedRank =
      outputLayout.getInnerAxis().size() + outputLayout.getOuterAxis().size();
  SmallVector<int64_t> result(packedRank, 0);
  SmallVector<int64_t> inputCount(inputLayout.getOuterAxis().size(), 0);
  auto axisPlainToPacked = inputLayout.getPlainToPackedAxisMapping();
  for (int64_t i = 0; i < packedRank; ++i) {
    // packedOutput[i] --> originalOutputAxis --> originalInputAxis
    // TODO: add failed check here
    int64_t originalOutputAxis = *outputLayout.getPlainAxis(i);
    int64_t originalInputAxis = plainPermAxes[originalOutputAxis];
    SmallVector<int64_t> packedInputAxes = axisPlainToPacked[originalInputAxis];
    result[i] = packedInputAxes[inputCount[originalInputAxis]++];
  }
  return result;
}

static int64_t applyPermutationAndReindexReassoc(
    SmallVector<ReassociationIndices> &reassocIndices,
    ArrayRef<int64_t> permutation) {
  if (!permutation.empty())
    applyPermutationToVector<ReassociationIndices>(reassocIndices, permutation);
  int64_t nextPos = 0;
  for (ReassociationIndices &indices : reassocIndices) {
    for (auto &index : indices) {
      index = nextPos;
      nextPos += 1;
    }
  }
  return nextPos;
}

// extends linalg::pack(...) for named ops
LogicalResult packLinalgOp(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                           OperatorLayout opLayout) {
  if (linalgOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(linalgOp, "require tensor semantics");
  LLVM_DEBUG(llvm::dbgs() << "Try packing named op "
                          << linalgOp.getOperation()->getName() << ".\n");
  Location loc = linalgOp->getLoc();
  SmallVector<Value> packOps;
  SmallVector<Value> unPackOps;
  SmallVector<Value> inputsAndInits, results;
  SmallVector<OpOperand *> initOperands = llvm::to_vector(llvm::map_range(
      linalgOp.getDpsInitsMutable(), [](OpOperand &o) { return &o; }));
  SmallVector<OpOperand *> inputOperands = linalgOp.getDpsInputOperands();
  SmallVector<TensorLayout> inputLayouts = opLayout.getSupportedInputLayouts();
  SmallVector<TensorLayout> initLayouts = opLayout.getSupportedOutputLayouts();
  // check all inputs and inits are tensor, otherwise no need for layout
  // propagation
  bool allTensor =
      llvm::all_of(inputOperands,
                   [](OpOperand *opOperand) {
                     return mlir::isa<TensorType>(opOperand->get().getType());
                   }) &&
      llvm::all_of(initOperands, [](OpOperand *opOperand) {
        return mlir::isa<TensorType>(opOperand->get().getType());
      });
  if (!allTensor) {
    LLVM_DEBUG(llvm::dbgs() << "At least one input of named op: "
                            << linalgOp.getOperation()->getName()
                            << " is not tensor. Skip.\n");
    return failure("The op does not need packing.");
  }
  for (const auto &operandsList : {inputOperands, initOperands}) {
    for (OpOperand *opOperand : operandsList) {
      size_t pos = opOperand->getOperandNumber();
      Value operand = opOperand->get();
      TensorLayout targetLayout = pos >= inputLayouts.size()
                                      ? initLayouts[pos - inputLayouts.size()]
                                      : inputLayouts[pos];
      SmallVector<int64_t> outerPerm = targetLayout.getOuterAxis();
      SmallVector<int64_t> innerPos = targetLayout.getInnerAxis();
      SmallVector<OpFoldResult> innerPackSizes = targetLayout.getTileSizes();
      Value dest = tensor::PackOp::createDestinationTensor(
          rewriter, loc, operand, innerPackSizes, innerPos, outerPerm);
      ShapedType operandType = cast<ShapedType>(operand.getType());
      bool areConstantTiles =
          llvm::all_of(innerPackSizes, [](OpFoldResult tile) {
            return getConstantIntValue(tile).has_value();
          });
      if (areConstantTiles && operandType.hasStaticShape() &&
          !tensor::PackOp::requirePaddingValue(
              operandType.getShape(), innerPos,
              cast<ShapedType>(dest.getType()).getShape(), {},
              innerPackSizes)) {
        packOps.push_back(insertLayoutPack(
            rewriter, loc, operand, dest, innerPos, innerPackSizes, outerPerm));
      } else {
        return failure();
      }
      inputsAndInits.push_back(packOps.back());
    }
  }

  // Step 3. Build the packed op, use the type of `inits` as result types.
  ValueRange inputs =
      ValueRange{inputsAndInits}.take_front(linalgOp.getNumDpsInputs());
  ValueRange inits =
      ValueRange{inputsAndInits}.take_back(linalgOp.getNumDpsInits());
  // TODO: deal with generic
  linalg::LinalgOp packedLinalgOp;
  if (auto reduceOp = dyn_cast<linalg::ReduceOp>(&linalgOp)) {
    SmallVector<int64_t> packedAxes =
        getPackedAxes(reduceOp->getDimensions(), inputLayouts[0]);
    packedLinalgOp = rewriter.create<linalg::ReduceOp>(
        loc, inits.getTypes(), inputs, inits, packedAxes);
    packedLinalgOp->getRegion(0).takeBody(linalgOp->getRegion(0));
  } else if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(&linalgOp)) {
    SmallVector<int64_t> packedAxes =
        getPackedAxes(broadcastOp->getDimensions(), initLayouts[0]);
    packedLinalgOp = rewriter.create<linalg::BroadcastOp>(loc, inputs[0],
                                                          inits[0], packedAxes);
  } else if (auto transposeOp = dyn_cast<linalg::TransposeOp>(&linalgOp)) {
    SmallVector<int64_t> packedPermAxes = getPackedPermAxes(
        transposeOp->getPermutation(), inputLayouts[0], initLayouts[0]);
    packedLinalgOp = rewriter.create<linalg::TransposeOp>(
        loc, inputs[0], inits[0], packedPermAxes);
  } else if (isa<linalg::SoftmaxOp>(linalgOp) || isa<linalg::MapOp>(linalgOp) ||
             isa<linalg::YieldOp>(linalgOp) || isa<linalg::IndexOp>(linalgOp)) {
    return failure(
        "Packing logic not implemented for SoftMax/Map/Yield/Index.");
  } else {
    packedLinalgOp = mlir::clone(
        rewriter, linalgOp, SmallVector<Type>{inputsAndInits.back().getType()},
        inputsAndInits);
  }

  // Step 4. Unpack all the op results.
  for (OpResult result : packedLinalgOp->getResults()) {
    int64_t resultNum = result.getResultNumber();
    assert(resultNum < initLayouts.size() &&
           "Linalg op results num exceeds inits num.");
    // Build the symmetrical UnPackOp to the existing PackOp.
    unPackOps.push_back(
        insertLayoutUnpack(rewriter, packedLinalgOp->getLoc(), result,
                           initLayouts[resultNum].getInnerAxis(),
                           initLayouts[resultNum].getTileSizes(),
                           initLayouts[resultNum].getOuterAxis()));
    results.push_back(unPackOps.back());
  }

  // Step 5. Replace `linalgOp`.
  rewriter.replaceOp(linalgOp, results);
  return success();
}

// check whether the op is already packed or not
static bool checkPacked(Operation *op, const OperatorLayout &opLayout) {
  // check whether rank match
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    assert(linalgOp.getDpsInits().size() ==
               opLayout.getSupportedOutputLayouts().size() &&
           linalgOp.getDpsInputs().size() ==
               opLayout.getSupportedInputLayouts().size());
    for (auto [index, layout] :
         llvm::enumerate(opLayout.getSupportedInputLayouts())) {
      // if dimension mismatch, then the op itself is already packed
      if (layout.getOuterAxis().size() !=
          cast<RankedTensorType>(linalgOp.getDpsInputs()[index].getType())
              .getShape()
              .size())
        return true;
    }
    for (auto [index, layout] :
         llvm::enumerate(opLayout.getSupportedOutputLayouts())) {
      // if dimension mismatch, then the op itself is already packed
      if (layout.getOuterAxis().size() !=
          cast<RankedTensorType>(linalgOp.getDpsInits()[index].getType())
              .getShape()
              .size())
        return true;
    }
  } else {
    assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
  }
  return false;
}

using ControlPackNamedOpsFn =
    std::function<FailureOr<OperatorLayout>(Operation *)>;

class PropagateLayoutOnNamedOps
    : public impl::PropagateLayoutOnNamedOpsBase<PropagateLayoutOnNamedOps> {
public:
  using impl::PropagateLayoutOnNamedOpsBase<
      PropagateLayoutOnNamedOps>::PropagateLayoutOnNamedOpsBase;
  void runOnOperation() final;
};

LogicalResult namedOpLayoutPropagation(MLIRContext *ctx, mlir::Operation *graph,
                                       ControlPackNamedOpsFn controlFn) {
  IRRewriter rewriter(ctx);
  graph->walk([&](Operation *op) {
    if (mlir::gc::utils::isPackableNamedOp(op)) {
      LLVM_DEBUG(llvm::dbgs() << "Op " << op->getName() << " visited.\n");
      FailureOr<OperatorLayout> opLayout = controlFn(op);
      if (failed(opLayout)) {
        LLVM_DEBUG(llvm::dbgs() << "Op " << op->getName()
                                << " does not have layout information.\n");
        return WalkResult::skip();
      }
      if ((*opLayout).isPlain()) {
        LLVM_DEBUG(llvm::dbgs() << "Op " << op->getName()
                                << " has plain layout, skip packing.\n");
        return WalkResult::advance();
      }
      // pack op into ideal layout
      LLVM_DEBUG(llvm::dbgs()
                 << "Op " << op->getName() << "'s inferred layout:\n"
                 << *opLayout << "\n");
      // insert pack
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);
      if (checkPacked(op, *opLayout)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Op " << op->getName() << " already packed.\n");
        return WalkResult::advance();
      }
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        if (failed(packLinalgOp(rewriter, linalgOp, *opLayout))) {
          return WalkResult::skip();
        }
      } else if (auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
        Location loc = expandShapeOp->getLoc();
        auto inputLayout = opLayout->getSupportedInputLayouts()[0];
        auto outputLayout = opLayout->getSupportedOutputLayouts()[0];
        Value curSrc = expandShapeOp.getSrc();
        Value curDst = expandShapeOp.getResult();
        Value dest = tensor::PackOp::createDestinationTensor(
            rewriter, loc, curSrc, inputLayout.getTileSizes(),
            inputLayout.getInnerAxis(), inputLayout.getOuterAxis());
        Value packedSource = insertLayoutPack(
            rewriter, loc, curSrc, dest, inputLayout.getInnerAxis(),
            inputLayout.getTileSizes(), inputLayout.getOuterAxis());
        SmallVector<ReassociationIndices> newReassocIndices =
            expandShapeOp.getReassociationIndices();
        int64_t nextPos = applyPermutationAndReindexReassoc(
            newReassocIndices, inputLayout.getOuterAxis());
        // Then add direct mapping for the inner tile dims.
        for (size_t i = 0; i < inputLayout.getInnerAxis().size(); ++i) {
          newReassocIndices.push_back({nextPos});
          nextPos += 1;
        }
        RankedTensorType newExpandType = tensor::PackOp::inferPackedType(
            dyn_cast<RankedTensorType>(curDst.getType()),
            *getConstantIntValues(outputLayout.getTileSizes()),
            outputLayout.getInnerAxis(), outputLayout.getOuterAxis());
        Value packedExpandShape = rewriter.create<tensor::ExpandShapeOp>(
            loc, newExpandType, packedSource, newReassocIndices);
        Value newUnPackOp = insertLayoutUnpack(
            rewriter, loc, packedExpandShape, outputLayout.getInnerAxis(),
            outputLayout.getTileSizes(), outputLayout.getOuterAxis());
        rewriter.replaceOp(expandShapeOp, newUnPackOp);
      } else if (auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
        Location loc = collapseShapeOp->getLoc();
        auto inputLayout = opLayout->getSupportedInputLayouts()[0];
        auto outputLayout = opLayout->getSupportedOutputLayouts()[0];
        Value curSrc = collapseShapeOp.getSrc();
        Value curDst = collapseShapeOp.getResult();
        Value dest = tensor::PackOp::createDestinationTensor(
            rewriter, loc, curSrc, inputLayout.getTileSizes(),
            inputLayout.getInnerAxis(), inputLayout.getOuterAxis());
        Value packedSource = insertLayoutPack(
            rewriter, loc, curSrc, dest, inputLayout.getInnerAxis(),
            inputLayout.getTileSizes(), inputLayout.getOuterAxis());
        SmallVector<ReassociationIndices> newReassocIndices =
            collapseShapeOp.getReassociationIndices();
        int64_t nextPos = applyPermutationAndReindexReassoc(
            newReassocIndices, outputLayout.getOuterAxis());
        // Then add direct mapping for the inner tile dims.
        for (size_t i = 0; i < inputLayout.getInnerAxis().size(); ++i) {
          newReassocIndices.push_back({nextPos});
          nextPos += 1;
        }
        RankedTensorType newCollapseType = tensor::PackOp::inferPackedType(
            dyn_cast<RankedTensorType>(curDst.getType()),
            *getConstantIntValues(outputLayout.getTileSizes()),
            outputLayout.getInnerAxis(), outputLayout.getOuterAxis());
        Value packedCollapseShape = rewriter.create<tensor::CollapseShapeOp>(
            loc, newCollapseType, packedSource, newReassocIndices);
        Value newUnPackOp = insertLayoutUnpack(
            rewriter, loc, packedCollapseShape, outputLayout.getInnerAxis(),
            outputLayout.getTileSizes(), outputLayout.getOuterAxis());
        rewriter.replaceOp(collapseShapeOp, newUnPackOp);
      }
    }
    return WalkResult::advance();
  });
  return success();
}

static LogicalResult createAndReplaceWithGenericVNNIMatmul(
    RewriterBase &rewriter, MLIRContext *context, SmallVector<Value> inputs,
    SmallVector<Value> inits, int64_t batchDimSize, int64_t blockingFactor,
    Operation *matmulOp) {
  AffineMap mapInput, mapWeight, mapOutput;
  int64_t dims = batchDimSize + 7;
  SmallVector<AffineExpr> exprs(dims);
  // dims is in order B1, ..., Bn, M, N, K, m, n, k, vnni
  bindDimsList<AffineExpr>(context, exprs);
  SmallVector<AffineExpr> batchExprs(exprs.begin(),
                                     exprs.begin() + batchDimSize);
  AffineExpr M = exprs[batchDimSize], N = exprs[batchDimSize + 1],
             K = exprs[batchDimSize + 2], m = exprs[batchDimSize + 3],
             n = exprs[batchDimSize + 4], k = exprs[batchDimSize + 5],
             vnni = exprs[batchDimSize + 6];
  SmallVector<AffineExpr> resultA{M, K, m, k};
  SmallVector<AffineExpr> resultB{N, K, k.floorDiv(blockingFactor), n, vnni};
  SmallVector<AffineExpr> resultC{M, N, m, n};
  resultA.insert(resultA.begin(), batchExprs.begin(), batchExprs.end());
  resultB.insert(resultB.begin(), batchExprs.begin(), batchExprs.end());
  resultC.insert(resultC.begin(), batchExprs.begin(), batchExprs.end());
  mapInput = AffineMap::get(/*dims=*/dims, /*symbols=*/0, resultA, context);
  mapWeight = AffineMap::get(/*dims=*/dims, /*symbols=*/0, resultB, context);
  mapOutput = AffineMap::get(/*dims=*/dims, /*symbols=*/0, resultC, context);
  SmallVector<mlir::utils::IteratorType> batchIterators(
      batchDimSize, mlir::utils::IteratorType::parallel);
  SmallVector<mlir::utils::IteratorType> iterators{
      mlir::utils::IteratorType::parallel,
      mlir::utils::IteratorType::parallel,
      mlir::utils::IteratorType::reduction,
      mlir::utils::IteratorType::parallel,
      mlir::utils::IteratorType::parallel,
      mlir::utils::IteratorType::reduction,
      mlir::utils::IteratorType::reduction};
  iterators.insert(iterators.begin(), batchIterators.begin(),
                   batchIterators.end());
  auto replacementOp = rewriter.create<linalg::GenericOp>(
      matmulOp->getLoc(), inits[0].getType(), inputs, inits,
      ArrayRef<AffineMap>{mapInput, mapWeight, mapOutput}, iterators,
      /*doc=*/"", /*libraryCall=*/"");
  rewriter.inlineRegionBefore(matmulOp->getRegion(0), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());
  rewriter.replaceOp(matmulOp, replacementOp.getResult(0));
  return success();
}

template <typename OpTy>
static LogicalResult packVNNIMMT4D(RewriterBase &rewriter, OpTy mmt4dOp,
                                   bool useNamedOp = false) {
  auto elementType = getElementTypeOrSelf(mmt4dOp.getInputs()[0].getType());
  if (!elementType.isBF16() && !elementType.isInteger(8))
    return rewriter.notifyMatchFailure(mmt4dOp, "require bf16/int8 data type");
  Location loc = mmt4dOp.getLoc();
  // BNKnk --> BNKkn2k
  auto weightShape =
      cast<ShapedType>(mmt4dOp.getInputs()[1].getType()).getShape();
  int64_t weightRank = weightShape.size();
  // pack innermost k axis
  SmallVector<int64_t> innerPos{weightRank - 1};
  int64_t blockingFactor = elementType.isBF16() ? 2 : 4;
  SmallVector<OpFoldResult> tileSize{rewriter.getIndexAttr(blockingFactor)};
  // BNKnk --> BNKkn2k
  int64_t batchDimSize = weightRank - 4;
  SmallVector<int64_t> batchPerm(batchDimSize, 0);
  std::iota(batchPerm.begin(), batchPerm.end(), 0);
  SmallVector<int64_t> outerPerm{batchDimSize, batchDimSize + 1,
                                 batchDimSize + 3, batchDimSize + 2};
  outerPerm.insert(outerPerm.begin(), batchPerm.begin(), batchPerm.end());
  OpOperand *RHSOperand = mmt4dOp.getDpsInputOperand(1);
  Value dest = tensor::PackOp::createDestinationTensor(
      rewriter, loc, RHSOperand->get(), tileSize, innerPos, outerPerm);
  auto zeroAttr = rewriter.getZeroAttr(getElementTypeOrSelf(dest.getType()));
  Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
  Value VNNIPack = rewriter.create<tensor::PackOp>(
      loc, RHSOperand->get(), dest, innerPos, tileSize, zero, outerPerm);
  // check whether VNNIPack causes padding
  int64_t innermostKDim = weightShape[weightRank - 1];
  int64_t paddingSize = (innermostKDim % blockingFactor)
                            ? (blockingFactor - innermostKDim % blockingFactor)
                            : 0;
  assert(!paddingSize && "Padding shall not be introduced by VNNI pack.");
  SmallVector<Value> inputsValues{mmt4dOp.getInputs()[0], VNNIPack};
  FailureOr<linalg::GenericOp> op = linalgx::makeGenericPackedMatmulOp(
      rewriter, loc, linalgx::PackingType::VNNI_MM4D, inputsValues,
      mmt4dOp.getDpsInits());
  if (failed(op))
    return failure();
  rewriter.replaceOp(mmt4dOp, *op);
  return success();
}

// strictly check whether the packed matmul is BMKmk & BNKkn
static bool isMM4DMatmul(linalg::GenericOp matmulOp) {
  return linalgx::isGenericPackedMatmulOp(matmulOp.getOperation(),
                                          linalgx::PackingType::MM4D);
}

/*
If possible, pack to Mm2DVnniOp or Mm4DVnniOp.
If not possible, pack to GenericOp.
*/
static LogicalResult packVNNIGeneric(RewriterBase &rewriter,
                                     linalg::GenericOp matmulOp,
                                     bool useNamedOp = false) {
  if (matmulOp.getDpsInputs().size() != 2)
    return rewriter.notifyMatchFailure(matmulOp, "require 2 inputs");

  auto elementType = getElementTypeOrSelf(matmulOp.getInputs()[0].getType());
  if (!elementType.isBF16() && !elementType.isInteger(8))
    return rewriter.notifyMatchFailure(matmulOp, "require bf16/int8 data type");

  if (matmulOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(matmulOp, "require static shape");

  if (matmulOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(matmulOp, "require tensor semantics");

  // isContractionInterfaceImpl checks the following restrictions:
  // 1. has 2 inputs && 1 outputs
  // 2. has >=1 reduction loop
  // 3. all affine maps are projected permutations:
  //    a. no symbols or zeros in result
  //    b. result is a non-duplicated subset of input
  // 4. op body contains both mul&&add
  if (!mlir::linalg::isaContractionOpInterface(matmulOp))
    return rewriter.notifyMatchFailure(matmulOp, "require matmul semantics");

  // check whether generic op is packed as BMKmk & BNKkn
  if (!isMM4DMatmul(matmulOp))
    return rewriter.notifyMatchFailure(matmulOp,
                                       "require packed MM4D matmul semantics");

  OpOperand &weight = matmulOp->getOpOperand(1);
  // TODO(yifei): check ISA
  Location loc = matmulOp.getLoc();
  int64_t blockingFactor = elementType.isBF16() ? 2 : 4;
  SmallVector<OpFoldResult> tileSize{rewriter.getIndexAttr(blockingFactor)};
  // BNKkn, get weight's rank
  auto weightShape =
      cast<ShapedType>(matmulOp.getInputs()[1].getType()).getShape();
  int64_t weightRank = weightShape.size();
  auto innerPos = SmallVector<int64_t>{weightRank - 2};
  // pack weight
  Value dest = tensor::PackOp::createDestinationTensor(
      rewriter, loc, weight.get(), tileSize, innerPos, SmallVector<int64_t>{});
  auto zeroAttr = rewriter.getZeroAttr(getElementTypeOrSelf(dest.getType()));
  Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
  Value VNNIPack = rewriter.create<tensor::PackOp>(loc, weight.get(), dest,
                                                   innerPos, tileSize, zero);

  int64_t batchDimSize = weightRank - 4;
  SmallVector<Value> inputsValues{matmulOp.getInputs()[0], VNNIPack};
  // check whether VNNIPack causes padding, weightShape is BNKkn
  int64_t innermostKDim = weightShape[weightRank - 2];
  int64_t paddingSize = (innermostKDim % blockingFactor)
                            ? (blockingFactor - innermostKDim % blockingFactor)
                            : 0;
  assert(!paddingSize && "Padding shall not be introduced by VNNI pack.");
  FailureOr<linalg::GenericOp> op = linalgx::makeGenericPackedMatmulOp(
      rewriter, loc, linalgx::PackingType::VNNI_MM4D, inputsValues,
      matmulOp.getDpsInits());
  if (failed(op))
    return failure();
  rewriter.replaceOp(matmulOp, *op);
  return success();
}

template <typename OpTy> struct PackVNNI : public OpRewritePattern<OpTy> {
  PackVNNI(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(context, benefit) {}

  LogicalResult matchAndRewrite(OpTy linalgOp,
                                PatternRewriter &rewriter) const override {
    if (failed(packVNNIMMT4D(rewriter, linalgOp)))
      return failure();
    return success();
  }
};

template <>
struct PackVNNI<linalg::GenericOp>
    : public OpRewritePattern<linalg::GenericOp> {
  PackVNNI(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::GenericOp>(context, benefit) {}
  LogicalResult matchAndRewrite(linalg::GenericOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (failed(packVNNIGeneric(rewriter, matmulOp)))
      return failure();
    return success();
  }
};

static FailureOr<linalgx::PackingType>
shallRevertToType(linalg::GenericOp matmulOp) {
  if (linalgx::isGenericPackedMatmulOp(matmulOp.getOperation(),
                                       linalgx::PackingType::MM4D))
    return linalgx::PackingType::MM2D4D;
  else if (linalgx::isGenericPackedMatmulOp(matmulOp.getOperation(),
                                            linalgx::PackingType::VNNI_MM4D))
    return linalgx::PackingType::VNNI_MM2D;
  return failure();
}

static bool isPlainActivationMatmul(OperatorLayout matmulLayout) {
  auto inputLayout = matmulLayout.getSupportedInputLayouts()[0];
  auto outputLayout = matmulLayout.getSupportedInputLayouts()[0];
  return !inputLayout.isBlocking() && !outputLayout.isBlocking();
}

static LogicalResult
revertMatmulPacking(MLIRContext *ctx, mlir::Operation *graph,
                    const std::vector<OperatorLayout> &matmulLayouts) {
  IRRewriter rewriter(ctx);
  uint64_t layoutOffset = 0;
  graph->walk([&](Operation *op) {
    if (auto matmulOp = dyn_cast<linalg::GenericOp>(op)) {
      FailureOr<linalgx::PackingType> revertType = shallRevertToType(matmulOp);
      if (succeeded(revertType) &&
          isPlainActivationMatmul(matmulLayouts[layoutOffset])) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(op);
        // replace VNNI_MM4D with unpack + VNNI_MM2D + pack
        // get preceding pack and successive unpack
        auto packInputOp = matmulOp.getDpsInputOperand(0)
                               ->get()
                               .getDefiningOp<tensor::PackOp>();
        auto packInitOp = matmulOp.getDpsInitOperand(0)
                              ->get()
                              .getDefiningOp<tensor::PackOp>();
        if (!packInputOp || !packInitOp)
          return WalkResult::skip();
        if (!matmulOp.getResults()[0].hasOneUse())
          return WalkResult::skip();
        auto consumer = matmulOp.getResults()[0].getUses().begin();
        auto unPackOp = dyn_cast<tensor::UnPackOp>(consumer->getOwner());
        if (!unPackOp)
          return WalkResult::skip();
        Location loc = matmulOp.getLoc();
        // unpack input
        auto packInputInnerTiles = packInputOp.getMixedTiles();
        auto packInputInnerDimsPos = packInputOp.getInnerDimsPos();
        auto packInputOuterDimsPerm = packInputOp.getInnerDimsPos();
        Value unpackInputDest = tensor::UnPackOp::createDestinationTensor(
            rewriter, loc, packInputOp, packInputInnerTiles,
            packInputInnerDimsPos, packInputOuterDimsPerm);
        Value reUnpackInput = rewriter.create<tensor::UnPackOp>(
            loc, packInputOp, unpackInputDest, packInputInnerDimsPos,
            packInputInnerTiles, packInputOuterDimsPerm);
        // unpack init
        auto packInitInnerTiles = packInitOp.getMixedTiles();
        auto packInitInnerDimsPos = packInitOp.getInnerDimsPos();
        auto packInitOuterDimsPerm = packInitOp.getInnerDimsPos();
        Value unpackInitDest = tensor::UnPackOp::createDestinationTensor(
            rewriter, loc, packInitOp, packInitInnerTiles, packInitInnerDimsPos,
            packInitOuterDimsPerm);
        Value reUnpackInit = rewriter.create<tensor::UnPackOp>(
            loc, packInitOp, unpackInitDest, packInitInnerDimsPos,
            packInitInnerTiles, packInitOuterDimsPerm);
        // replace vnni_4D with vnni_2D
        auto VNNI2D = linalgx::makeGenericPackedMatmulOp(
            rewriter, loc, *revertType,
            ValueRange{reUnpackInput, matmulOp.getDpsInputOperand(1)->get()},
            ValueRange{reUnpackInit});
        if (failed(VNNI2D))
          return WalkResult::interrupt();
        // insert pack before unpack
        auto unPackInnerTiles = unPackOp.getMixedTiles();
        auto unPackInnerDimsPos = unPackOp.getInnerDimsPos();
        auto unPackOuterDimsPerm = unPackOp.getInnerDimsPos();
        Value packDest = tensor::PackOp::createDestinationTensor(
            rewriter, loc, (*VNNI2D)->getResult(0), unPackInnerTiles,
            unPackInnerDimsPos, unPackOuterDimsPerm);
        auto zeroAttr =
            rewriter.getZeroAttr(getElementTypeOrSelf(packDest.getType()));
        Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
        Value rePack = rewriter.create<tensor::PackOp>(
            loc, (*VNNI2D)->getResult(0), packDest, unPackInnerDimsPos,
            unPackInnerTiles, zero, unPackOuterDimsPerm);
        rewriter.replaceOp(op, rePack);
        layoutOffset++;
      }
    } else if (auto matmulOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (mlir::gc::utils::isSupportedContractionNamedOp(matmulOp)) {
        layoutOffset++;
      }
    }
    return WalkResult::advance();
  });
  return success();
}

/*
Match patterns like broadcast + pack, uplift pack
*/
struct UpliftPackOverBroadcast : public OpRewritePattern<tensor::PackOp> {
  UpliftPackOverBroadcast(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<tensor::PackOp>(context, benefit) {}
  LogicalResult matchAndRewrite(tensor::PackOp pack,
                                PatternRewriter &rewriter) const override {
    auto broadcastOp = pack.getSource().getDefiningOp<linalg::BroadcastOp>();
    if (!broadcastOp || !broadcastOp.getResult()[0].hasOneUse()) {
      return failure();
    }
    SmallVector<int64_t> innerTileSizes = pack.getStaticTiles();
    SmallVector<int64_t> innerDimsPos(pack.getInnerDimsPos());
    SmallVector<int64_t> outerDimsPerm(pack.getOuterDimsPerm());
    int64_t rank =
        cast<ShapedType>(pack.getSource().getType()).getShape().size();
    if (outerDimsPerm.empty()) {
      outerDimsPerm.resize(rank);
      std::iota(outerDimsPerm.begin(), outerDimsPerm.end(), 0);
    }
    ArrayRef<int64_t> broadcastAxis = broadcastOp.getDimensions();
    SmallVector<int64_t> newInnerDimsPos, newOuterDimsPerm, packedBroadcastAxis;
    SmallVector<OpFoldResult> newInnerTileSizes;
    llvm::SmallDenseMap<int64_t, int64_t> axisMapping;
    int64_t axisCounter = 0;
    for (int64_t axis = 0; axis < rank; ++axis) {
      if (std::find(broadcastAxis.begin(), broadcastAxis.end(), axis) ==
          broadcastAxis.end()) {
        // if the axis is not broadcasted, keep it
        axisMapping[axis] = axisCounter++;
      }
    }
    // update broadcast dims
    for (auto [index, axis] : llvm::enumerate(outerDimsPerm)) {
      if (std::find(broadcastAxis.begin(), broadcastAxis.end(), axis) !=
          broadcastAxis.end()) {
        packedBroadcastAxis.push_back(index);
      }
    }
    for (auto [index, axis] : llvm::enumerate(innerDimsPos)) {
      if (std::find(broadcastAxis.begin(), broadcastAxis.end(), axis) !=
          broadcastAxis.end()) {
        packedBroadcastAxis.push_back(index + rank);
      }
    }
    // update packing axis
    for (auto [index, axis] : llvm::enumerate(outerDimsPerm)) {
      if (std::find(broadcastAxis.begin(), broadcastAxis.end(), axis) ==
          broadcastAxis.end()) {
        newOuterDimsPerm.push_back(axisMapping[axis]);
      }
    }
    for (auto [index, axis] : llvm::enumerate(innerDimsPos)) {
      if (std::find(broadcastAxis.begin(), broadcastAxis.end(), axis) ==
          broadcastAxis.end()) {
        newInnerDimsPos.push_back(axisMapping[axis]);
        newInnerTileSizes.push_back(
            rewriter.getIndexAttr(innerTileSizes[index]));
      }
    }
    // replace ops
    auto loc = broadcastOp.getLoc();
    auto dest = tensor::PackOp::createDestinationTensor(
        rewriter, loc, broadcastOp.getDpsInputs()[0], newInnerTileSizes,
        newInnerDimsPos, newOuterDimsPerm);
    Value packedSource =
        insertLayoutPack(rewriter, loc, broadcastOp.getDpsInputs()[0], dest,
                         newInnerDimsPos, newInnerTileSizes, newOuterDimsPerm);
    auto newBroadcastOp = rewriter.create<linalg::BroadcastOp>(
        loc, packedSource, pack.getDest(), packedBroadcastAxis);
    rewriter.replaceOp(pack, newBroadcastOp.getResults());
    return success();
  }
};

void PropagateLayoutOnNamedOps::runOnOperation() {
  MLIRContext *ctx = &getContext();
  IRRewriter rewriter(ctx);
  mlir::Operation *graph = getOperation();
  // collect matmul layouts in topological order
  auto &layoutAnalysisResult = getAnalysis<GlobalAnalysis>();
  std::vector<OperatorLayout> matmulLayouts;
  graph->walk([&](Operation *op) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (mlir::gc::utils::isSupportedContractionNamedOp(linalgOp)) {
        matmulLayouts.push_back(*(layoutAnalysisResult.getOpLayout(op)));
      }
    }
    return WalkResult::advance();
  });
  // stage 1.1: pack matmul with `BlockPackMatmulPatterns` if any side of it
  // requires packing; do nothing if the matmul is computed on plain format
  // TODO(yifei): deal with transposed plain matmul...
  RewritePatternSet packMatmulPatterns(&getContext());
  mlir::linalg::ControlBlockPackMatmulFn packMatmulControlFn =
      [&](linalg::LinalgOp op) -> mlir::linalg::BlockPackMatmulOptions {
    mlir::linalg::BlockPackMatmulOptions options;
    auto matmulLayout = *(layoutAnalysisResult.getOpLayout(op));
    // currently supported combination: plain & blocking & plain OR blocking &
    // blocking & blocking
    TensorLayout inputLayout = matmulLayout.getSupportedInputLayouts()[0];
    TensorLayout weightLayout = matmulLayout.getSupportedInputLayouts()[1];
    TensorLayout outputLayout = matmulLayout.getSupportedOutputLayouts()[0];
    if (!inputLayout.isBlocking() && !weightLayout.isBlocking() &&
        !outputLayout.isBlocking())
      return options; // return default options to skip packing
    // specify B side as be NKkn
    options.rhsTransposeOuterBlocks = true;
    options.rhsTransposeInnerBlocks = false;
    // extract tile sizes
    auto matmulCfg = MatmulConfigAnalysis(op.getOperation()).getConfig();
    OpFoldResult MBlock = rewriter.getIndexAttr(matmulCfg.innerMostMBlock),
                 KBlock = rewriter.getIndexAttr(matmulCfg.innerMostKBlock),
                 NBlock = rewriter.getIndexAttr(matmulCfg.innerMostNBlock);
    if (!inputLayout.getTileSizes().empty())
      assert(inputLayout.getTileSizes()[0] == MBlock &&
             inputLayout.getTileSizes()[1] == KBlock &&
             "Layout tile size and matmul block size mismatch.");
    if (!weightLayout.getTileSizes().empty())
      assert(weightLayout.getTileSizes()[0] == KBlock &&
             weightLayout.getTileSizes()[1] == NBlock &&
             "Layout tile size and matmul block size mismatch.");
    if (!outputLayout.getTileSizes().empty())
      assert(outputLayout.getTileSizes()[0] == MBlock &&
             outputLayout.getTileSizes()[1] == NBlock &&
             "Layout tile size and matmul block size mismatch.");
    options.blockFactors.push_back(*getConstantIntValue(MBlock));
    options.blockFactors.push_back(*getConstantIntValue(NBlock));
    options.blockFactors.push_back(*getConstantIntValue(KBlock));
    return options;
  };
  linalg::populateBlockPackMatmulPatterns(packMatmulPatterns,
                                          packMatmulControlFn);
  if (failed(
          applyPatternsAndFoldGreedily(graph, std::move(packMatmulPatterns))))
    return signalPassFailure();

  // stage 1.2: pack VNNI
  RewritePatternSet packVNNIPatterns(&getContext());
  packVNNIPatterns.add<PackVNNI<linalg::GenericOp>, PackVNNI<linalg::Mmt4DOp>,
                       PackVNNI<linalg::BatchMmt4DOp>>(ctx);
  if (failed(applyPatternsAndFoldGreedily(graph, std::move(packVNNIPatterns))))
    return signalPassFailure();

  // stage 1.3: revert necessary blocking on matmul op
  // RevertMatmulPacking
  // double confirm the number of identifiable matmuls
  // collect matmul layouts in topological order
  uint64_t numMatmuls = 0;
  graph->walk([&](Operation *op) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (mlir::gc::utils::isSupportedContractionNamedOp(linalgOp) ||
          linalgx::isGenericPackedMatmulOp(linalgOp.getOperation(),
                                           linalgx::PackingType::MM4D) ||
          linalgx::isGenericPackedMatmulOp(linalgOp.getOperation(),
                                           linalgx::PackingType::VNNI_MM4D)) {
        numMatmuls += 1;
      }
    }
    return WalkResult::advance();
  });
  assert(matmulLayouts.size() == numMatmuls &&
         "One to one matmul mapping failed.");
  if (failed(revertMatmulPacking(ctx, graph, matmulLayouts)))
    return signalPassFailure();

  // stage 2: propagate layout on other named ops
  ControlPackNamedOpsFn layoutControlFn =
      [&](Operation *op) -> FailureOr<OperatorLayout> {
    return layoutAnalysisResult.getOpLayout(op);
  };
  if (failed(namedOpLayoutPropagation(ctx, graph, layoutControlFn)))
    return signalPassFailure();

  // stage 3: uplift pack through broadcast
  RewritePatternSet upliftPatterns(&getContext());
  upliftPatterns.add<UpliftPackOverBroadcast>(ctx);
  if (failed(applyPatternsAndFoldGreedily(graph, std::move(upliftPatterns))))
    return signalPassFailure();
}

} // namespace gc
} // namespace mlir
