//===--- AttentionUtils.h - Common utilities for attention optimizations --===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_TRANSFORMS_GPU_ATTENTIONOPTIMIZATIONS_UTILS_H
#define GC_TRANSFORMS_GPU_ATTENTIONOPTIMIZATIONS_UTILS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"

namespace mlir::gc::attention {

/// Collect all xegpu.dpas operations in the loop body.
SmallVector<xegpu::DpasOp> collectDpasOps(scf::ForOp forOp);

/// Recursively collect all operations inside `region` that `value`
/// transitively depends on. Returns them in topological order
/// (dependencies first).
void collectDepsInRegion(Value value, Region *region,
                         SmallVectorImpl<Operation *> &deps,
                         DenseSet<Operation *> &visited);

/// Check if any operation in `deps` uses `value` as an operand.
bool usesValue(const SmallVectorImpl<Operation *> &deps, Value value);

/// Compute sg_data = shape / sg_layout element-wise.
/// Returns failure if any dimension is not evenly divisible.
FailureOr<SmallVector<int32_t>> computeSgData(ArrayRef<int64_t> shape,
                                              ArrayRef<int32_t> sgLayout);

/// Build an xegpu::LayoutAttr from the given components.
/// Empty arrays for instData/order are treated as "not set" (nullptr).
xegpu::LayoutAttr makeLayout(MLIRContext *ctx, ArrayRef<int32_t> sgLayout,
                             ArrayRef<int32_t> sgData,
                             ArrayRef<int32_t> instData = {},
                             ArrayRef<int32_t> order = {});

} // namespace mlir::gc::attention

#endif // GC_TRANSFORMS_GPU_ATTENTIONOPTIMIZATIONS_UTILS_H
