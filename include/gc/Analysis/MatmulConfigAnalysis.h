//===-- MatmulConfigAnalysis.h - DESC ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_MATMULCONFIGANALYSIS_H
#define MLIR_ANALYSIS_MATMULCONFIGANALYSIS_H

#include "gc/Dialect/Linalgx/IR/LinalgxOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include <llvm/Support/Debug.h>
#include <memory>
#include <numeric>

namespace mlir {
namespace gc {

using namespace mlir;

struct SystemDesc {
  // get runtime OMP_NUM_THREADS
  uint32_t getNumThreads() {
    char *numThreads = getenv("OMP_NUM_THREADS");
    if (numThreads) {
      return std::stoi(numThreads);
    }
    return 1;
  }
  // get cache size by cacheLevel
  size_t getCacheSize(uint8_t cacheLevel) {
    if (cacheLevel == 1) {
      char *cacheSize = getenv("L1_CACHE_SIZE");
      if (cacheSize) {
        return std::stoi(cacheSize);
      }
    } else if (cacheLevel == 2) {
      char *cacheSize = getenv("L2_CACHE_SIZE");
      if (cacheSize) {
        return std::stoi(cacheSize);
      }
    } else if (cacheLevel == 3) {
      char *cacheSize = getenv("L3_CACHE_SIZE");
      if (cacheSize) {
        return std::stoi(cacheSize);
      }
    }
    return 0;
  }

  SmallVector<size_t> getContractionOperationMaxVectorLength() {
    return {512UL, 512UL};
  }
};

struct MatmulConfig {
  uint32_t MBlock, NBlock, KBlock;
  uint32_t MThreads, NThreads, KThreads;
  uint32_t innerMostMBlock, innerMostNBlock, innerMostKBlock;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &ss,
                                       const MatmulConfig &config);
};

enum DimType { Batch, M, N, K };

[[maybe_unused]] static SmallVector<unsigned>
extractDimTypeIdx(ArrayRef<DimType> tyList, DimType ty) {
  SmallVector<unsigned> idxList;
  for (auto [idx, type] : llvm::enumerate(tyList)) {
    if (type == ty) {
      idxList.push_back(idx);
    }
  }
  return idxList;
}

static FailureOr<SmallVector<SmallVector<DimType>>>
getOprandDimType(linalg::LinalgOp &linalgOp) {
  if (isa<linalg::MatmulOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K},
        SmallVector<DimType>{DimType::K, DimType::N},
        SmallVector<DimType>{DimType::M, DimType::N}};
  } else if (llvm::isa<linalgx::Mm2DVnniOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K},
        SmallVector<DimType>{DimType::N, DimType::K, DimType::K, DimType::N,
                             DimType::K},
        SmallVector<DimType>{DimType::M, DimType::N, DimType::M, DimType::N}};
  } else if (llvm::isa<linalgx::Mm4DVnniOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K, DimType::M, DimType::K},
        SmallVector<DimType>{DimType::N, DimType::K, DimType::K, DimType::N,
                             DimType::K},
        SmallVector<DimType>{DimType::M, DimType::N, DimType::M, DimType::N}};
  } else if (llvm::isa<linalg::BatchMatmulOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::Batch, DimType::M, DimType::K},
        SmallVector<DimType>{DimType::Batch, DimType::K, DimType::N},
        SmallVector<DimType>{DimType::Batch, DimType::M, DimType::N}};
  } else if (llvm::isa<linalg::GenericOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K, DimType::M, DimType::K},
        SmallVector<DimType>{DimType::N, DimType::K, DimType::K, DimType::N},
        SmallVector<DimType>{DimType::M, DimType::N, DimType::M, DimType::N}};
  }
  return failure();
}

struct MatmulConfigAnalysis {
public:
  explicit MatmulConfigAnalysis(Operation *root);
  MatmulConfig getConfig() { return config; }

private:
  MatmulConfig config;
};

} // namespace gc
} // namespace mlir

#endif