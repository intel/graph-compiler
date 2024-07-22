//===-- MatmulConfigAnalysis.h - the analysis for matmul config -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_MATMULCONFIGANALYSIS_H
#define MLIR_ANALYSIS_MATMULCONFIGANALYSIS_H

#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <cstring>

namespace mlir {
namespace gc {

using namespace mlir;

// A mock for the taget information
// TODO: replace it with upstream hardware description model
struct SystemDesc {

  static int getPositiveIntFromStr(char *str, int defaultValue = 1) {
    if (!str || strlen(str) == 0 || str[0] > '9' || str[0] < '0') {
      return defaultValue;
    }
    auto val = std::stoi(str);
    return val > 0 ? val : defaultValue;
  }

  // get runtime OMP_NUM_THREADS
  uint32_t getNumThreads() {
    char *numThreads = getenv("OMP_NUM_THREADS");
    return getPositiveIntFromStr(numThreads, 1);
  }
  // get cache size by cacheLevel
  size_t getCacheSize(uint8_t cacheLevel) {
    if (cacheLevel == 1) {
      char *cacheSize = getenv("L1_CACHE_SIZE");
      return getPositiveIntFromStr(cacheSize, 0);
    } else if (cacheLevel == 2) {
      char *cacheSize = getenv("L2_CACHE_SIZE");
      return getPositiveIntFromStr(cacheSize, 0);
    } else if (cacheLevel == 3) {
      char *cacheSize = getenv("L3_CACHE_SIZE");
      return getPositiveIntFromStr(cacheSize, 0);
    }
    return 0;
  }

  // get the maximum vector length in bits
  size_t getMaxVectorLength() {
    char *maxVectorLanes = getenv("MAX_VECTOR_LENGTH");
    return getPositiveIntFromStr(maxVectorLanes, 512);
  }
};

// The configuration for matmul tiling
// TODO: support batch matmul
struct MatmulConfig {
  // The number of threads distributed to M, N, K
  uint32_t MThreads, NThreads, KThreads;
  // The innermost block size for M, N, K which will be directly converted to
  // brgemm.
  uint32_t innerMostMBlock, innerMostNBlock, innerMostKBlock;
  // The outer block size for M, N, K which will be used to decide the loop tile
  // size in single thread
  uint32_t MBlock, NBlock, KBlock;
};

enum DimType { Batch, M, N, K };

// Extract the index of the given DimType in the DimType list
inline SmallVector<unsigned> extractDimTypeIdx(ArrayRef<DimType> tyList,
                                               DimType ty) {
  SmallVector<unsigned> idxList;
  for (auto [idx, type] : llvm::enumerate(tyList)) {
    if (type == ty) {
      idxList.push_back(idx);
    }
  }
  return idxList;
}

// Get the operand dim type for every operand for the given linalg op
inline FailureOr<SmallVector<SmallVector<DimType>>>
getOprandDimType(linalg::LinalgOp &linalgOp) {
  // TODO: replace the linalgx op with generic op
  if (llvm::isa<linalg::MatmulOp>(linalgOp)) {
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
  } else if (llvm::isa<linalg::MatmulTransposeAOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::K, DimType::M},
        SmallVector<DimType>{DimType::K, DimType::N},
        SmallVector<DimType>{DimType::M, DimType::N}};
  } else if (llvm::isa<linalg::MatmulTransposeBOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K},
        SmallVector<DimType>{DimType::N, DimType::K},
        SmallVector<DimType>{DimType::M, DimType::N}};
  } else if (llvm::isa<linalg::BatchMatmulTransposeAOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::Batch, DimType::K, DimType::M},
        SmallVector<DimType>{DimType::Batch, DimType::K, DimType::N},
        SmallVector<DimType>{DimType::Batch, DimType::M, DimType::N}};
  } else if (llvm::isa<linalg::BatchMatmulTransposeBOp>(linalgOp)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::Batch, DimType::M, DimType::K},
        SmallVector<DimType>{DimType::Batch, DimType::N, DimType::K},
        SmallVector<DimType>{DimType::Batch, DimType::M, DimType::N}};
  }
  return failure();
}

// The analysis to extract the matmul configuration from the given linalg op
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