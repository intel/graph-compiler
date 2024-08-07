//===- PropagateLayoutOnNamedOps.cpp - Propagate packing on linalg named ops*-
// C++-*-===//
//
// This file is only temporarily used to extend upstream or upcoming utility in
// TilingInterface, which finally aims for upstream.
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

#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Transforms/Passes.h"
namespace mlir {
namespace gc {
#define GEN_PASS_DEF_PROPAGATELAYOUTONNAMEDOPS
#include "gc/Transforms/Passes.h.inc"

#define DEBUG_TYPE "named-op-layout-propagation"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::tensor;

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
FailureOr<linalg::PackResult> packNamedOp(RewriterBase &rewriter,
                                          linalg::LinalgOp linalgOp,
                                          OperatorLayout opLayout) {
  if (linalgOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(linalgOp, "require tensor semantics");
  LLVM_DEBUG(llvm::dbgs() << "Try packing named op "
                          << linalgOp.getOperation()->getName() << ".\n");
  Location loc = linalgOp->getLoc();
  SmallVector<tensor::PackOp> packOps;
  SmallVector<tensor::UnPackOp> unPackOps;
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
        packOps.push_back(rewriter.create<tensor::PackOp>(
            loc, operand, dest, innerPos, innerPackSizes, std::nullopt,
            outerPerm));
      } else {
        // TODO: value of the padding attribute should be determined by
        // consumers.
        auto zeroAttr =
            rewriter.getZeroAttr(getElementTypeOrSelf(dest.getType()));
        Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
        packOps.push_back(rewriter.create<tensor::PackOp>(
            loc, operand, dest, innerPos, innerPackSizes, zero, outerPerm));
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
  } else if (isa<linalg::SoftmaxOp>(linalgOp) ||
             isa<linalg::GenericOp>(linalgOp) || isa<linalg::MapOp>(linalgOp) ||
             isa<linalg::YieldOp>(linalgOp) || isa<linalg::IndexOp>(linalgOp)) {
    return failure(
        "Packing logic not implemented for SoftMax/Generic/Map/Yield/Index.");
  } else {
    packedLinalgOp = mlir::clone(
        rewriter, linalgOp, SmallVector<Type>{inputsAndInits.back().getType()},
        inputsAndInits);
  }

  // Step 4. Unpack all the op results.
  for (OpResult result : packedLinalgOp->getResults()) {
    int64_t resultNum = result.getResultNumber();
    tensor::PackOp maybePackedInit =
        inits[resultNum].getDefiningOp<tensor::PackOp>();
    if (!maybePackedInit) {
      results.push_back(result);
      continue;
    }
    // Build the symmetrical UnPackOp to the existing PackOp.
    unPackOps.push_back(rewriter.create<tensor::UnPackOp>(
        packedLinalgOp->getLoc(), result, maybePackedInit.getSource(),
        maybePackedInit.getInnerDimsPos(), maybePackedInit.getMixedTiles(),
        maybePackedInit.getOuterDimsPerm()));
    results.push_back(unPackOps.back());
  }

  // Step 5. Replace `linalgOp`.
  rewriter.replaceOp(linalgOp, results);

  // Return packedLinalgOp.
  return linalg::PackResult{
      packOps, cast<linalg::LinalgOp>(packedLinalgOp.getOperation()),
      unPackOps};
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
        FailureOr<linalg::PackResult> packedOp =
            packNamedOp(rewriter, linalgOp, *opLayout);
        if (failed(packedOp)) {
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
        Value packedSource = rewriter.create<tensor::PackOp>(
            loc, curSrc, dest, inputLayout.getInnerAxis(),
            inputLayout.getTileSizes(), std::nullopt,
            inputLayout.getOuterAxis());
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
        auto unpackDst = tensor::UnPackOp::createDestinationTensor(
            rewriter, loc, packedExpandShape, outputLayout.getTileSizes(),
            outputLayout.getInnerAxis(), outputLayout.getOuterAxis());
        auto newUnPackOp = rewriter.create<tensor::UnPackOp>(
            loc, packedExpandShape, unpackDst, outputLayout.getInnerAxis(),
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
        Value packedSource = rewriter.create<tensor::PackOp>(
            loc, curSrc, dest, inputLayout.getInnerAxis(),
            inputLayout.getTileSizes(), std::nullopt,
            inputLayout.getOuterAxis());
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
        auto unpackDst = tensor::UnPackOp::createDestinationTensor(
            rewriter, loc, packedCollapseShape, outputLayout.getTileSizes(),
            outputLayout.getInnerAxis(), outputLayout.getOuterAxis());
        auto newUnPackOp = rewriter.create<tensor::UnPackOp>(
            loc, packedCollapseShape, unpackDst, outputLayout.getInnerAxis(),
            outputLayout.getTileSizes(), outputLayout.getOuterAxis());
        rewriter.replaceOp(collapseShapeOp, newUnPackOp);
      }
    }
    return WalkResult::advance();
  });
  return success();
}

static void createAndReplaceWithGenericVNNIMatmul(
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
}

template <typename OpTy>
static LogicalResult packVNNIMMT4D(RewriterBase &rewriter, OpTy mmt4dOp) {
  auto elementType = getElementTypeOrSelf(mmt4dOp.getInputs()[0].getType());
  if (!elementType.isBF16() && !elementType.isInteger(8))
    return rewriter.notifyMatchFailure(mmt4dOp, "require bf16/int8 data type");
  Location loc = mmt4dOp.getLoc();
  // BNKnk --> BNKkn2k
  int64_t weightRank =
      cast<ShapedType>(mmt4dOp.getInputs()[1].getType()).getShape().size();
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
  Value VNNIPack =
      rewriter.create<tensor::PackOp>(loc, RHSOperand->get(), dest, innerPos,
                                      tileSize, std::nullopt, outerPerm);
  SmallVector<Value> inputsValues{mmt4dOp.getInputs()[0], VNNIPack};
  if (!batchDimSize) {
    auto vnniOp = rewriter.create<mlir::linalgx::Mm4DVnniOp>(
        loc, mmt4dOp.getDpsInits().getTypes(), inputsValues,
        mmt4dOp.getDpsInits());
    rewriter.replaceOp(mmt4dOp, vnniOp);
  } else {
    mlir::gc::createAndReplaceWithGenericVNNIMatmul(
        rewriter, mmt4dOp.getContext(), inputsValues, mmt4dOp.getDpsInits(),
        batchDimSize, blockingFactor, mmt4dOp);
  }
  return success();
}

// strictly check whether the packed matmul is BMKmk & BNKkn
static bool isMM4DMatmul(linalg::GenericOp matmulOp) {
  SmallVector<AffineMap> indexingMaps = matmulOp.getIndexingMapsArray();
  auto iterators = matmulOp.getIteratorTypesArray();
  AffineMap inputMap = indexingMaps[0], weightMap = indexingMaps[1],
            outputMap = indexingMaps[2];
  int64_t inputRank = inputMap.getNumResults(),
          weightRank = weightMap.getNumResults(),
          outputRank = outputMap.getNumResults();
  // check rank
  if ((weightRank < 4) || (inputRank != weightRank) ||
      (weightRank != outputRank))
    return false;
  // check mapping --> find batch, M, N, K
  FailureOr<mlir::linalg::ContractionDimensions> res =
      mlir::linalg::inferContractionDims(matmulOp);
  assert(succeeded(res) && "unexpected failure in infer contraction dims");
  unsigned batchDimSize = res->batch.size();
  SmallVector<unsigned> expectedM{batchDimSize, batchDimSize + 3};
  SmallVector<unsigned> expectedN{batchDimSize + 1, batchDimSize + 4};
  SmallVector<unsigned> expectedK{batchDimSize + 2, batchDimSize + 5};
  if (expectedM == res->m && expectedN == res->n && expectedK == res->k)
    return true;
  return false;
}

/*
If possible, pack to Mm2DVnniOp or Mm4DVnniOp.
If not possible, pack to GenericOp.
*/
static LogicalResult packVNNIGeneric(RewriterBase &rewriter,
                                     linalg::GenericOp matmulOp) {
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
  // get weight's rank
  int64_t weightRank =
      cast<ShapedType>(weight.get().getType()).getShape().size();
  auto innerPos = SmallVector<int64_t>{weightRank - 2};
  // pack weight.
  Value dest = tensor::PackOp::createDestinationTensor(
      rewriter, loc, weight.get(), tileSize, innerPos, SmallVector<int64_t>{});
  Value VNNIPack = rewriter.create<tensor::PackOp>(
      loc, weight.get(), dest, innerPos, tileSize, std::nullopt);

  int64_t batchDimSize = weightRank - 4;
  SmallVector<Value> inputsValues{matmulOp.getInputs()[0], VNNIPack};
  if (!batchDimSize) {
    Value operandC = matmulOp.getDpsInits()[0];
    auto VNNIMatmulOp = rewriter.create<mlir::linalgx::Mm4DVnniOp>(
        loc, operandC.getType(), inputsValues, ValueRange{operandC});
    rewriter.replaceOp(matmulOp, VNNIMatmulOp);
  } else {
    mlir::gc::createAndReplaceWithGenericVNNIMatmul(
        rewriter, matmulOp.getContext(), inputsValues, matmulOp.getDpsInits(),
        batchDimSize, blockingFactor, matmulOp);
  }
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
    Value packedSource = rewriter.create<tensor::PackOp>(
        loc, broadcastOp.getDpsInputs()[0], dest, newInnerDimsPos,
        newInnerTileSizes,
        /*padding=*/std::nullopt, newOuterDimsPerm);
    auto newBroadcastOp = rewriter.create<linalg::BroadcastOp>(
        loc, packedSource, pack.getDest(), packedBroadcastAxis);
    rewriter.replaceOp(pack, newBroadcastOp.getResults());
    return success();
  }
};

void PropagateLayoutOnNamedOps::runOnOperation() {
  MLIRContext *ctx = &getContext();
  mlir::Operation *graph = getOperation();
  // stage1: pack matmul
  RewritePatternSet packMatmulPatterns(&getContext());
  mlir::linalg::ControlBlockPackMatmulFn packMatmulControlFn =
      [&](linalg::LinalgOp op) -> mlir::linalg::BlockPackMatmulOptions {
    mlir::linalg::BlockPackMatmulOptions options;
    auto &layoutAnalysisResult = getAnalysis<GlobalAnalysis>();
    auto matmulLayout = *(layoutAnalysisResult.getOpLayout(op));
    TensorLayout LHSLayout = matmulLayout.getSupportedInputLayouts()[0];
    TensorLayout RHSLayout = matmulLayout.getSupportedInputLayouts()[1];
    // hardcode to let B side to be NKkn
    options.rhsTransposeOuterBlocks = true;
    options.rhsTransposeInnerBlocks = false;
    assert(LHSLayout.getTileSizes()[1] == RHSLayout.getTileSizes()[0] &&
           "Inconsistent matmul tile size.");
    options.blockFactors.push_back(
        *getConstantIntValue(LHSLayout.getTileSizes()[0]));
    options.blockFactors.push_back(
        *getConstantIntValue(LHSLayout.getTileSizes()[1]));
    options.blockFactors.push_back(
        *getConstantIntValue(RHSLayout.getTileSizes()[1]));
    return options;
  };
  linalg::populateBlockPackMatmulPatterns(packMatmulPatterns,
                                          packMatmulControlFn);
  if (failed(
          applyPatternsAndFoldGreedily(graph, std::move(packMatmulPatterns))))
    return signalPassFailure();

  // stage2: pack VNNI
  RewritePatternSet packVNNIPatterns(&getContext());
  packVNNIPatterns.add<PackVNNI<linalg::GenericOp>, PackVNNI<linalg::Mmt4DOp>,
                       PackVNNI<linalg::BatchMmt4DOp>>(ctx);
  if (failed(applyPatternsAndFoldGreedily(graph, std::move(packVNNIPatterns))))
    return signalPassFailure();

  // stage3: propagate layout on other named ops
  ControlPackNamedOpsFn layoutControlFn =
      [&](Operation *op) -> FailureOr<OperatorLayout> {
    auto &layoutAnalysisResult = getAnalysis<GlobalAnalysis>();
    return layoutAnalysisResult.getOpLayout(op);
  };
  if (failed(namedOpLayoutPropagation(ctx, graph, layoutControlFn)))
    return signalPassFailure();

  // stage4: uplift pack through broadcast
  RewritePatternSet upliftPatterns(&getContext());
  upliftPatterns.add<UpliftPackOverBroadcast>(ctx);
  if (failed(applyPatternsAndFoldGreedily(graph, std::move(upliftPatterns))))
    return signalPassFailure();
}

} // namespace gc
} // namespace mlir
