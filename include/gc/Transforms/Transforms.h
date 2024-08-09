//===- Transforms.h - transformation utilities ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_TRANSFORMS_TRANSFORMS_H
#define GC_TRANSFORMS_TRANSFORMS_H

#include "gc/Analysis/GlobalAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir {
namespace gc {
LogicalResult packLinalgOp(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                          OperatorLayout opLayout);

LogicalResult namedOpLayoutPropagation(RewriterBase &rewriter,
                                       linalg::LinalgOp linalgOp,
                                       OperatorLayout opLayout);
} // namespace gc
} // namespace mlir

#endif // GC_TRANSFORMS_TRANSFORMS_H
