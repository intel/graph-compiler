//===- PropagateLayout.cpp - Propagate pack unpack on linalg named ops --*- C++
//-*-===//
//
// This file is only temporarily used to extend upstream or upcoming utility in
// TilingInterface, which finally aims for upstream.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <numeric>

#include "gc/Analysis/GlobalAnalysis.h"
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
#define GEN_PASS_DEF_PROPAGATELAYOUT
#include "gc/Transforms/Passes.h.inc"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::tensor;

static FailureOr<linalg::PackResult> packNamedOp(RewriterBase &rewriter,
                                                 linalg::LinalgOp linalgOp,
                                                 OperatorLayout opLayout) {
  std::cout << "----------------------------------" << std::endl;
  std::cout << " Visiting op in packNamedOp ";
  linalgOp->getName().print(llvm::errs());
  std::cout << std::endl;
  std::cout << "----------------------------------" << std::endl;
  Location loc = linalgOp->getLoc();
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();

  SmallVector<tensor::PackOp> packOps;
  SmallVector<tensor::UnPackOp> unPackOps;
  SmallVector<Value> inputsAndInits, results;
  SmallVector<OpOperand *> initOperands = llvm::to_vector(llvm::map_range(
      linalgOp.getDpsInitsMutable(), [](OpOperand &o) { return &o; }));
  SmallVector<OpOperand *> inputOperands = linalgOp.getDpsInputOperands();
  std::cout << "Num of input operands: " << inputOperands.size() << std::endl;
  std::cout << "Num of init operands: " << initOperands.size() << std::endl;
  SmallVector<TensorLayout> inputLayouts = opLayout.getSupportedInputLayouts();
  SmallVector<TensorLayout> initLayouts = opLayout.getSupportedOutputLayouts();
  std::cout << "Num of input layouts: " << inputLayouts.size() << std::endl;
  std::cout << "Num of init layouts: " << initLayouts.size() << std::endl;

  // check all inputs and inits are tensor, otherwise no need for layout
  // propagation
  bool allTensor =
      llvm::all_of(inputOperands,
                   [](OpOperand *opOperand) {
                     return opOperand->get().getType().isa<TensorType>();
                   }) &&
      llvm::all_of(initOperands, [](OpOperand *opOperand) {
        return opOperand->get().getType().isa<TensorType>();
      });
  std::cout << "The op's input is all tensor?" << allTensor << std::endl;
  if (!allTensor) {
    return failure("the op does not need packing.");
  }
  for (const auto &operandsList : {inputOperands, initOperands}) {
    for (OpOperand *opOperand : operandsList) {
      int64_t pos = opOperand->getOperandNumber();
      std::cout << "pos: " << pos << std::endl;
      Value operand = opOperand->get();
      TensorLayout targetLayout = pos >= inputLayouts.size()
                                      ? initLayouts[pos - inputLayouts.size()]
                                      : inputLayouts[pos];
      SmallVector<int64_t> outerPerm = targetLayout.getOuterAxis();
      SmallVector<int64_t> innerPos = targetLayout.getInnerAxis();
      SmallVector<OpFoldResult> innerPackSizes = targetLayout.getTileSizes();

      std::cout << "Suggested layout: " << targetLayout << std::endl;

      std::cout << "Operand shape: ";
      for (auto dim :
           llvm::cast<RankedTensorType>(operand.getType()).getShape()) {
        std::cout << dim << ", ";
      }
      std::cout << std::endl;

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
  // TODO(yifei): the axis info of reduce/broadcast/transpose may change
  auto packedLinalgOp = mlir::clone(
      rewriter, linalgOp, SmallVector<Type>{inputsAndInits.back().getType()},
      inputsAndInits);

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
        maybePackedInit.getInnerDimsPos(), maybePackedInit.getMixedTiles()));
    results.push_back(unPackOps.back());
  }

  // Step 5. Replace `linalgOp`.
  rewriter.replaceOp(linalgOp, results);

  // Return packedLinalgOp.
  return linalg::PackResult{
      packOps, cast<linalg::LinalgOp>(packedLinalgOp.getOperation()),
      unPackOps};
}

class PropagateLayout : public impl::PropagateLayoutBase<PropagateLayout> {
public:
  using impl::PropagateLayoutBase<PropagateLayout>::PropagateLayoutBase;
  void runOnOperation() final;
};

void PropagateLayout::runOnOperation() {
  MLIRContext *ctx = &getContext();
  mlir::Operation *graph = getOperation();
  IRRewriter rewriter(ctx);
  // walk the entire graph
  auto &layoutAnalysisResult = getAnalysis<GlobalAnalysis>();
  SmallVector<Operation *> packTODOList;
  graph->walk([&](Operation *op) {
    if (isa<linalg::LinalgOp>(op) && !mlir::linalg::isaContractionOpInterface(
                                         dyn_cast<linalg::LinalgOp>(op))) {
      packTODOList.push_back(op);
    }
  });
  for (auto op : packTODOList) {
    std::cout << std::endl;
    std::cout << "----------------------------------" << std::endl;
    std::cout << "Visiting op ";
    op->getName().print(llvm::errs());
    std::cout << std::endl;
    std::cout << "----------------------------------" << std::endl;
    FailureOr<OperatorLayout> opLayout = layoutAnalysisResult.getOpLayout(op);
    if (failed(opLayout)) {
      std::cout << "infer failed" << std::endl;
    } else {
      // pack op into ideal layout
      std::cout << "-------- supported layouts -------" << std::endl;
      std::cout << *opLayout << std::endl;
      // insert pack
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        FailureOr<linalg::PackResult> packedOp =
            packNamedOp(rewriter, linalgOp, *opLayout);
      }
      graph->dump();
    }
  }
  graph->dump();
}

} // namespace gc
} // namespace mlir
