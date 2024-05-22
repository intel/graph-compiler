//===- Tilig.hpp - Tiling ops using TilingInterface --*- C++ -*-===//
//
// This file is only temporarily used to extend upstream or upcoming utility in
// TilingInterface, which finally aims for upstream.
//
//===----------------------------------------------------------------------===//

#ifndef TEMPORARY_TILEUSINGINTERFACE_X_H
#define TEMPORARY_TILEUSINGINTERFACE_X_H

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include <optional>
#include <utility>
namespace mlir {
namespace linalgX {

FailureOr<linalg::ForallReductionTilingResult> tileReductionUsingForall(
    RewriterBase &b, PartialReductionOpInterface op,
    ArrayRef<OpFoldResult> threadNums, ArrayRef<OpFoldResult> tileSizes,
    ArrayRef<OpFoldResult> newParallelDims, std::optional<ArrayAttr> mapping);

FailureOr<linalg::ForallReductionTilingResult>
tileAllUsingForall(RewriterBase &b, PartialReductionOpInterface op,
                   ArrayRef<OpFoldResult> numThreads,
                   ArrayRef<OpFoldResult> tileSizes,
                   std::optional<ArrayAttr> mapping);

} // namespace linalgX
} // namespace mlir

#endif