//===- EliminateConstantWeightPack.cpp - Eliminate Const Weight  *-- C++-*-===//
//
// This file is only temporarily used to extend upstream or upcoming utility in
// TilingInterface, which finally aims for upstream.
//
//===----------------------------------------------------------------------===//

#include <numeric>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "gc/Dialect/Linalgx/Utils.h"
#include "gc/Transforms/Passes.h"
#include "gc/Transforms/Transforms.h"

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_ELIMINATECONSTANTWEIGHTPACK
#include "gc/Transforms/Passes.h.inc"

using namespace mlir;

class EliminateConstantWeightPack
    : public impl::EliminateConstantWeightPackBase<
          EliminateConstantWeightPack> {
public:
  using impl::EliminateConstantWeightPackBase<
      EliminateConstantWeightPack>::EliminateConstantWeightPackBase;
  void runOnOperation() final;
};

void EliminateConstantWeightPack::runOnOperation() {
  MLIRContext *ctx = &getContext();
  IRRewriter rewriter(ctx);
  mlir::Operation *graph = getOperation();
  ValueTypeRange<Block::BlockArgListType> finalArgTypes =
      graph->getBlock()->getArgumentTypes();
  bool updated = false;
  graph->walk([&](Operation *op) {
    if (auto packedMatmul = dyn_cast<linalg::GenericOp>(op)) {
      if (linalgx::isGenericPackedMatmulOp(packedMatmul.getOperation(),
                                           linalgx::PackingType::MM2D4D) ||
          linalgx::isGenericPackedMatmulOp(packedMatmul.getOperation(),
                                           linalgx::PackingType::MM4D) ||
          linalgx::isGenericPackedMatmulOp(packedMatmul.getOperation(),
                                           linalgx::PackingType::VNNI_MM2D) ||
          linalgx::isGenericPackedMatmulOp(packedMatmul.getOperation(),
                                           linalgx::PackingType::VNNI_MM4D)) {
        auto srcVal = packedMatmul.getDpsInputOperands()[1]->get();
        mlir::Operation *argPack = nullptr;
        while (auto pack = srcVal.getDefiningOp<tensor::PackOp>()) {
          srcVal = pack.getSource();
          argPack = pack;
        }
        if (!isa<BlockArgument>(srcVal) || !argPack)
          return WalkResult::skip();
        // querying the block
        auto parentBlock = packedMatmul.getOperation()->getBlock();
        auto blockArgs = parentBlock->getArguments();
        auto found = std::find(blockArgs.begin(), blockArgs.end(), srcVal);
        assert(found != blockArgs.end());
        size_t idx = std::distance(blockArgs.begin(), found);
        assert(idx < blockArgs.size() && "Within index.");

        auto ty = dyn_cast<TensorType>(srcVal.getType());
        auto newArgTy = dyn_cast<TensorType>(
            packedMatmul.getDpsInputOperands()[1]->get().getType());
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(argPack);
        Value argReplace = rewriter.create<tensor::EmptyOp>(
            argPack->getLoc(), ty.getShape(), ty.getElementType());
        rewriter.replaceAllUsesWith(srcVal, argReplace);
        parentBlock->eraseArgument(idx);
        parentBlock->addArgument(newArgTy, argPack->getLoc());
        Value newPackedArg = parentBlock->getArguments().back();
        rewriter.replaceAllUsesWith(
            packedMatmul.getDpsInputOperands()[1]->get(), newPackedArg);
        updated = true;
        finalArgTypes = parentBlock->getArgumentTypes();
      }
    }
    return WalkResult::advance();
  });
  // Get funcOp
  if (updated) {
    func::FuncOp func = getOperation();
    FunctionType computeFuncType = func.getFunctionType();
    func.setType(
        FunctionType::get(ctx, finalArgTypes, computeFuncType.getResults()));
  }
}

} // namespace gc
} // namespace mlir
