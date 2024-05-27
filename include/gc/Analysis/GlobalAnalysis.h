//===- GlobalAnalysis.h - Graph Compiler analysis pass ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_GLOBALANALYSIS_H
#define MLIR_ANALYSIS_GLOBALANALYSIS_H

#include <iostream>
#include <memory>
#include <numeric>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace gc {

using namespace mlir;

class TensorLayout {
public:
  TensorLayout(ArrayRef<int64_t> outerAxis, ArrayRef<int64_t> innerAxis,
               ArrayRef<OpFoldResult> tileSizes) {
    assert(innerAxis.size() == tileSizes.size());
    for (auto oa : outerAxis) {
      OuterAxis.push_back(oa);
    }
    for (auto ia : innerAxis) {
      InnerAxis.push_back(ia);
    }
    for (auto ts : tileSizes) {
      TileSizes.push_back(ts);
    }
  }

  bool isPlainLayout() const {
    for (int64_t i = 0; i < static_cast<int64_t>(OuterAxis.size()); ++i) {
      if (i != OuterAxis[i])
        return false;
    }
    return TileSizes.empty() && InnerAxis.empty();
  }

  static TensorLayout createPlainLayout(int64_t rank) {
    SmallVector<int64_t> outerAxis(rank, 0);
    std::iota(outerAxis.begin(), outerAxis.end(), 0);
    return TensorLayout(outerAxis, SmallVector<int64_t>{},
                        SmallVector<OpFoldResult>{});
  }

  size_t getTensorRank() const { return OuterAxis.size(); }

  SmallVector<int64_t> getOuterAxis() const { return OuterAxis; }

  SmallVector<int64_t> getInnerAxis() const { return InnerAxis; }

  SmallVector<OpFoldResult> getTileSizes() const { return TileSizes; }

  friend std::ostream &operator<<(std::ostream &ss, const TensorLayout &layout);

  bool operator==(const TensorLayout &layout);

private:
  SmallVector<int64_t> OuterAxis;
  SmallVector<int64_t> InnerAxis;
  SmallVector<OpFoldResult> TileSizes;
};

class OperatorLayout {
public:
  OperatorLayout() {}

  OperatorLayout(SmallVector<TensorLayout> inputLayouts,
                 SmallVector<TensorLayout> outputLayouts) {
    supportedInputLayouts = inputLayouts;
    supportedOutputLayouts = outputLayouts;
  }

  SmallVector<TensorLayout> getSupportedInputLayouts() const {
    return supportedInputLayouts;
  }

  SmallVector<TensorLayout> getSupportedOutputLayouts() const {
    return supportedOutputLayouts;
  }

  TensorLayout getOutputLayout(int64_t idx) const {
    assert(idx < static_cast<int64_t>(supportedOutputLayouts.size()));
    return supportedOutputLayouts[idx];
  }

  friend std::ostream &operator<<(std::ostream &ss,
                                  const OperatorLayout &opLayout);

private:
  SmallVector<TensorLayout> supportedInputLayouts;
  SmallVector<TensorLayout> supportedOutputLayouts;
};

class GlobalAnalysis {
public:
  explicit GlobalAnalysis(Operation *root);

  FailureOr<OperatorLayout> getOpLayout(Operation *op) {
    if (layout.find(op) != layout.end())
      return layout[op];
    else
      return op->emitError("Current op does not have layout information.");
  }

private:
  DenseMap<Operation *, OperatorLayout> layout;
};

} // namespace gc
} // namespace mlir

#endif
