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
  graph->walk([&](linalg::LinalgOp linalgOp) {
    std::cout << std::endl;
    std::cout << "----------------------------------" << std::endl;
    std::cout << "Visiting op ";
    linalgOp.getOperation()->getName().print(llvm::errs());
    std::cout << std::endl;
    std::cout << "----------------------------------" << std::endl;
    FailureOr<OperatorLayout> opLayout =
        layoutAnalysisResult.getOpLayout(linalgOp);
    if (failed(opLayout)) {
      std::cout << "infer failed" << std::endl;
    } else {
      // pack op into ideal layout
      std::cout << "-------- supported layouts -------" << std::endl;
      std::cout << *opLayout << std::endl;
      // insert pack
    }
  });
  graph->dump();
}

} // namespace gc
} // namespace mlir
