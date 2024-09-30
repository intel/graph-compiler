//===- GlobalAnalysis.h - Graph Compiler analysis pass ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_GLOBALANALYSIS_H
#define MLIR_ANALYSIS_GLOBALANALYSIS_H

#include <numeric>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace gc {

using namespace mlir;

class TensorLayout {
public:
  TensorLayout(ArrayRef<int64_t> outerAxis, ArrayRef<int64_t> innerAxis,
               ArrayRef<OpFoldResult> tileSizes)
      : outerAxis(outerAxis), innerAxis(innerAxis), tileSizes(tileSizes) {
    assert(innerAxis.size() == tileSizes.size());
  }

  static bool isPlainOuterAxis(ArrayRef<int64_t> outerAxis) {
    for (int64_t i = 0; i < static_cast<int64_t>(outerAxis.size()); ++i) {
      if (i != outerAxis[i])
        return false;
    }
    return true;
  }

  bool isPlain() const {
    if (isPlainOuterAxis(outerAxis))
      return tileSizes.empty() && innerAxis.empty();
    return false;
  }

  bool isBlocking() const { return !tileSizes.empty() && !innerAxis.empty(); }

  static TensorLayout createPlainLayout(int64_t rank) {
    SmallVector<int64_t> outerAxis(rank, 0);
    std::iota(outerAxis.begin(), outerAxis.end(), 0);
    return TensorLayout(outerAxis, SmallVector<int64_t>{},
                        SmallVector<OpFoldResult>{});
  }

  DenseMap<int64_t, SmallVector<int64_t>> getPlainToPackedAxisMapping() {
    DenseMap<int64_t, SmallVector<int64_t>> axisMapping;
    int64_t outerAxisSize = outerAxis.size();
    for (int64_t i = 0; i < outerAxisSize; ++i) {
      axisMapping[outerAxis[i]].push_back(i);
    }
    for (int64_t i = 0; i < static_cast<int64_t>(innerAxis.size()); ++i) {
      axisMapping[innerAxis[i]].push_back(outerAxisSize + i);
    }
    return axisMapping;
  }

  FailureOr<int64_t> getPlainAxis(int64_t idx) {
    int64_t totalRank = outerAxis.size() + innerAxis.size();
    if (idx >= totalRank || idx < 0) {
      return failure();
    } else if (idx >= static_cast<int64_t>(outerAxis.size())) {
      return innerAxis[idx - outerAxis.size()];
    } else {
      return outerAxis[idx];
    }
  }

  size_t getRank() const { return outerAxis.size(); }

  SmallVector<int64_t> getOuterAxis() const { return outerAxis; }

  SmallVector<int64_t> getInnerAxis() const { return innerAxis; }

  SmallVector<OpFoldResult> getTileSizes() const { return tileSizes; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &ss,
                                       const TensorLayout &layout);

  bool operator==(const TensorLayout &layout) const;

private:
  SmallVector<int64_t> outerAxis;
  SmallVector<int64_t> innerAxis;
  SmallVector<OpFoldResult> tileSizes;
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

  bool isPlain() const {
    for (const auto &layout : llvm::concat<const TensorLayout>(
             supportedInputLayouts, supportedOutputLayouts)) {
      if (!layout.isPlain())
        return false;
    }
    return true;
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &ss,
                                       const OperatorLayout &opLayout);

private:
  SmallVector<TensorLayout> supportedInputLayouts;
  SmallVector<TensorLayout> supportedOutputLayouts;
};

class GlobalAnalysis {
public:
  explicit GlobalAnalysis(Operation *root);

  FailureOr<OperatorLayout> getOpLayout(Operation *op) {
    if (layoutCache.find(op) != layoutCache.end())
      return layoutCache[op];
    else
      return failure("Current op does not have layout information.");
  }

private:
  DenseMap<Operation *, OperatorLayout> layoutCache;
};

namespace utils {
bool isSupportedContractionNamedOp(const linalg::LinalgOp &linalgOp);

bool isPackableOp(Operation *op);

bool hasAllTensorSemantics(linalg::LinalgOp linalgOp);
} // namespace utils
} // namespace gc
} // namespace mlir

#endif
