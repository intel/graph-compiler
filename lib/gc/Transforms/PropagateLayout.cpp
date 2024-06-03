//===- PropagateLayoutOnNamedOps.cpp - Propagate packing on linalg named ops*-
// C++-*-===//
//
// This file is only temporarily used to extend upstream or upcoming utility in
// TilingInterface, which finally aims for upstream.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <numeric>

#include "gc/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
  SmallVector<int64_t> result(dimensions);
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
  size_t packedRank =
      outputLayout.getInnerAxis().size() + outputLayout.getOuterAxis().size();
  SmallVector<int64_t> result(packedRank, 0);
  SmallVector<int64_t> inputCount(inputLayout.getOuterAxis().size(), 0);
  auto inputP2B = inputLayout.getPlain2PackedMapping();
  for (size_t i = 0; i < packedRank; ++i) {
    // packedOutput[i] --> originalOutputAxis --> originalInputAxis
    size_t originalOutputAxis = *outputLayout.getOriginalAxis(i);
    size_t originalInputAxis = plainPermAxes[originalOutputAxis];
    SmallVector<int64_t> packedInputAxes = inputP2B[originalInputAxis];
    result[i] = packedInputAxes[inputCount[originalInputAxis]++];
  }
  return result;
}

// extends linalg::pack(...) for named ops
FailureOr<linalg::PackResult> packNamedOp(RewriterBase &rewriter,
                                          linalg::LinalgOp linalgOp,
                                          OperatorLayout opLayout) {
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
    packedLinalgOp = rewriter.create<linalg::BroadcastOp>(
        loc, inputs[0], inits[0], broadcastOp->getDimensions());
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
  auto walk = graph->walk([&](Operation *op) {
    FailureOr<OperatorLayout> opLayout = controlFn(op);
    if ((isa<linalg::LinalgOp>(op) && !mlir::linalg::isaContractionOpInterface(
                                          dyn_cast<linalg::LinalgOp>(op))) ||
        isa<tensor::ExpandShapeOp>(op) || isa<tensor::PadOp>(op)) {
      if (failed(opLayout)) {
        LLVM_DEBUG(llvm::dbgs() << "Op " << op->getName()
                                << "does not have layout information.\n");
        return WalkResult::skip();
      } else {
        // pack op into ideal layout
        LLVM_DEBUG(llvm::dbgs()
                   << "Op " << op->getName() << "'s inferred layout:\n"
                   << *opLayout << "\n");
        // insert pack
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(op);
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
          Value dest = tensor::PackOp::createDestinationTensor(
              rewriter, loc, expandShapeOp.getSrc(), inputLayout.getTileSizes(),
              inputLayout.getInnerAxis(), inputLayout.getOuterAxis());
          Value packedSource = rewriter.create<tensor::PackOp>(
              loc, expandShapeOp.getSrc(), dest, inputLayout.getInnerAxis(),
              inputLayout.getTileSizes(), std::nullopt,
              inputLayout.getOuterAxis());
          auto resultType = RankedTensorType::get(
              expandShapeOp.getStaticOutputShape(),
              expandShapeOp.getSrcType().getElementType());
          RankedTensorType resultPackType = tensor::PackOp::inferPackedType(
              resultType, vector::getAsIntegers(outputLayout.getTileSizes()),
              outputLayout.getInnerAxis(), outputLayout.getOuterAxis());
          auto reassocExpand = getReassociationIndicesForReshape(
              cast<ShapedType>(dest.getType()), resultPackType);
          auto packedExpandShape = rewriter.create<tensor::ExpandShapeOp>(
              loc, expandShapeOp.getSrcType().getElementType(), packedSource,
              *reassocExpand);
          Value result = rewriter.create<tensor::UnPackOp>(
              packedExpandShape->getLoc(), packedExpandShape, packedExpandShape,
              outputLayout.getInnerAxis(), outputLayout.getTileSizes(),
              outputLayout.getOuterAxis());
          rewriter.replaceOp(expandShapeOp, result);
        }
      }
    }
    return WalkResult::advance();
  });
  if (walk.wasSkipped())
    return failure();
  return success();
}

void PropagateLayoutOnNamedOps::runOnOperation() {
  MLIRContext *ctx = &getContext();
  mlir::Operation *graph = getOperation();
  ControlPackNamedOpsFn controlFn =
      [&](Operation *op) -> FailureOr<OperatorLayout> {
    auto &layoutAnalysisResult = getAnalysis<GlobalAnalysis>();
    return layoutAnalysisResult.getOpLayout(op);
  };
  if (failed(namedOpLayoutPropagation(ctx, graph, controlFn)))
    return signalPassFailure();
}

} // namespace gc
} // namespace mlir
