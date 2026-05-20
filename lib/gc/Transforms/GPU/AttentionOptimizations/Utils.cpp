//===--- Utils.cpp - Common utilities for attention optimizations ---------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/GPU/AttentionOptimizations/Utils.h"

namespace mlir::gc::attention {

SmallVector<xegpu::DpasOp> collectDpasOps(scf::ForOp forOp) {
  SmallVector<xegpu::DpasOp> dpasOps;
  forOp.getBody()->walk([&](xegpu::DpasOp dpas) { dpasOps.push_back(dpas); });
  return dpasOps;
}

void collectDepsInRegion(Value value, Region *region,
                         SmallVectorImpl<Operation *> &deps,
                         DenseSet<Operation *> &visited) {
  Operation *def = value.getDefiningOp();
  if (!def || !region->isAncestor(def->getParentRegion()))
    return;
  if (!visited.insert(def).second)
    return;
  for (Value operand : def->getOperands())
    collectDepsInRegion(operand, region, deps, visited);
  deps.push_back(def);
}

bool usesValue(const SmallVectorImpl<Operation *> &deps, Value value) {
  for (auto *op : deps)
    for (auto operand : op->getOperands())
      if (operand == value)
        return true;
  return false;
}

FailureOr<SmallVector<int32_t>> computeSgData(ArrayRef<int64_t> shape,
                                              ArrayRef<int32_t> sgLayout) {
  if (shape.size() != sgLayout.size())
    return failure();
  SmallVector<int32_t> sgData;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (sgLayout[i] == 0 || shape[i] % sgLayout[i] != 0)
      return failure();
    sgData.push_back(static_cast<int32_t>(shape[i] / sgLayout[i]));
  }
  return sgData;
}

xegpu::LayoutAttr makeLayout(MLIRContext *ctx, ArrayRef<int32_t> sgLayout,
                             ArrayRef<int32_t> sgData,
                             ArrayRef<int32_t> instData,
                             ArrayRef<int32_t> order) {
  DenseI32ArrayAttr instDataAttr =
      instData.empty() ? nullptr : DenseI32ArrayAttr::get(ctx, instData);
  DenseI32ArrayAttr orderAttr =
      order.empty() ? nullptr : DenseI32ArrayAttr::get(ctx, order);
  return xegpu::LayoutAttr::get(ctx, DenseI32ArrayAttr::get(ctx, sgLayout),
                                DenseI32ArrayAttr::get(ctx, sgData),
                                /*inst_data=*/instDataAttr,
                                /*lane_layout=*/nullptr,
                                /*lane_data=*/nullptr, /*order=*/orderAttr);
}

} // namespace mlir::gc::attention
