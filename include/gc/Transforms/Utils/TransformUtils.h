//===- TransformUtils.h - Transform utils -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_TRANSFORMS_UTILS_TRANSFORMUTILS_H
#define GC_TRANSFORMS_UTILS_TRANSFORMUTILS_H

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {

class Operation;
class OpBuilder;
struct Range;
class RewriterBase;
class TilingInterface;

namespace linalg {
class LinalgOp;
struct ContractionDimensions;
} // namespace linalg

namespace linalgx {
namespace utils {

// Return true if `op` is a blocked convolution.
bool isBlockedConvolution(Operation *op);

// Return true if `op` is a blocked matmul.
bool isBlockedMatmul(Operation *op);

// Return true if the `op` is a contraction defined as:
// - 2 input operands (LHS and RHS), and 1 output operand OUT.
// - The body is matmul-like
// - We have at least 1 m dimension involved in an outer-product along LHS.
// - We have at lest 1 n dimension involved in an outer-product along RHS.
// - We have at least 1 k dimension as a permutation on LHS and RHS.
// - The output map is a permutation map, while not gurantee is given on the
// input maps.
FailureOr<linalg::ContractionDimensions>
isContraction(linalg::LinalgOp linalgOp);

// Validate a tile configuration for a linalgOp when we can statically do that.
// Specific dims can be passed using 'dims'. If dims is empty the validation
// will start from the outermost dimension, moving to innermost ones up to the
// number of tiles.
// Tiling application can restricted based on the workload dimension size.
// The tiling is applied only to when all dimensions fulfill the predicate:
// '(dimSize[i] / tiles[i]) >= minTileFactor'.
bool validateFullTilesOnDims(TilingInterface tileOp,
                             ArrayRef<OpFoldResult> tiles,
                             ArrayRef<size_t> dims = {},
                             int64_t minTileFactor = 2);

// Rewrite scf.for to scf.forall. Assumes the loop to be parallel and
// marked with `kLoopId`.
constexpr const static llvm::StringLiteral kLoopParallel = "parallel";
constexpr const static llvm::StringLiteral kLoopRoot = "root";
void populateScfForToForAllRewritePattern(RewritePatternSet &patterns);

} // namespace utils
} // namespace linalgx
} // namespace mlir

#endif
