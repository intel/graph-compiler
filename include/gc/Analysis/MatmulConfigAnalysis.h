//===-- MatmulConfigAnalysis.h - the analysis for matmul config -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_MATMULCONFIGANALYSIS_H
#define MLIR_ANALYSIS_MATMULCONFIGANALYSIS_H

#include "gc/Dialect/Linalgx/Utils.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace mlir {
namespace gc {

using namespace mlir;

// The configuration for matmul tiling
// TODO: support batch matmul
struct MatmulConfig {
  // The number of threads distributed to M, N, K
  uint32_t MThreads, NThreads, KThreads;
  // The outer block size for M, N, K which will be used to decide the loop tile
  // size in single thread
  uint32_t MBlock, NBlock, KBlock;
  // The innermost block size for M, N, K which will be directly converted to
  // brgemm.
  uint32_t innerMostMBlock, innerMostNBlock, innerMostKBlock;
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
  } else if (linalgx::isGenericPackedMatmulOp(
                 linalgOp.getOperation(), linalgx::PackingType::VNNI_MM2D)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K},
        SmallVector<DimType>{DimType::N, DimType::K, DimType::K, DimType::N,
                             DimType::K},
        SmallVector<DimType>{DimType::M, DimType::N, DimType::M, DimType::N}};
  } else if (linalgx::isGenericPackedMatmulOp(
                 linalgOp.getOperation(), linalgx::PackingType::VNNI_MM4D)) {
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
  } else if (linalgx::isGenericPackedMatmulOp(linalgOp.getOperation(),
                                              linalgx::PackingType::MM4D)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K, DimType::M, DimType::K},
        SmallVector<DimType>{DimType::N, DimType::K, DimType::K, DimType::N},
        SmallVector<DimType>{DimType::M, DimType::N, DimType::M, DimType::N}};
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