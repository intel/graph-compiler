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

inline void getDimTypeFromIterators(linalg::LinalgOp linalgOp,
                                    SmallVectorImpl<DimType> &dimTypes) {
  SmallVector<mlir::utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();

  for (const auto &&[idx, iterType] : llvm::enumerate(iteratorTypes)) {
    if (iterType == mlir::utils::IteratorType::parallel) {
      SmallVector<std::pair<Value, unsigned>> operandDimPairs;
      linalgOp.mapIterationSpaceDimToAllOperandDims(idx, operandDimPairs);
      if (operandDimPairs.size() == 3) {
        dimTypes.push_back(DimType::Batch);
      } else if (llvm::any_of(operandDimPairs,
                              [&](std::pair<Value, unsigned> pair) {
                                return pair.first ==
                                       dyn_cast<linalg::ContractionOpInterface>(
                                           linalgOp.getOperation())
                                           .lhs();
                              })) {
        dimTypes.push_back(DimType::M);
      } else {
        dimTypes.push_back(DimType::N);
      }
    } else if (iterType == mlir::utils::IteratorType::reduction) {
      dimTypes.push_back(DimType::K);
    }
  }
}

inline SmallVector<DimType>
matchOperandToDimTypes(linalg::LinalgOp linalgOp, OpOperand *operand,
                       ArrayRef<DimType> allDimTypes) {
  ArrayRef<AffineExpr> map =
      linalgOp.getMatchingIndexingMap(operand).getResults();
  SmallVector<DimType> res;
  for (const AffineExpr &dim : map) {
    AffineDimExpr dimExpr = dyn_cast<AffineDimExpr>(dim);
    res.push_back(allDimTypes[dimExpr.getPosition()]);
  }
  return res;
}

inline SmallVector<SmallVector<DimType>>
getContractionOpOperandDimType(linalg::LinalgOp linalgOp) {
  SmallVector<DimType> dimTypes;
  getDimTypeFromIterators(linalgOp, dimTypes);
  SmallVector<DimType> ADimTypes = matchOperandToDimTypes(
      linalgOp, linalgOp.getDpsInputOperand(0), dimTypes);
  SmallVector<DimType> BDimTypes = matchOperandToDimTypes(
      linalgOp, linalgOp.getDpsInputOperand(1), dimTypes);
  SmallVector<DimType> CDimTypes =
      matchOperandToDimTypes(linalgOp, linalgOp.getDpsInitOperand(0), dimTypes);

  return SmallVector<SmallVector<DimType>>{ADimTypes, BDimTypes, CDimTypes};
}

// Get the operand dim type for every operand for the given linalg op
inline FailureOr<SmallVector<SmallVector<DimType>>>
getOprandDimType(linalg::LinalgOp &linalgOp) {
  // TODO: replace the linalgx op with generic op
  if (llvm::isa<linalg::ContractionOpInterface>(linalgOp.getOperation())) {
    return getContractionOpOperandDimType(linalgOp);
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
  } else if (linalgx::isGenericPackedMatmulOp(linalgOp.getOperation(),
                                              linalgx::PackingType::MM4D)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K, DimType::M, DimType::K},
        SmallVector<DimType>{DimType::N, DimType::K, DimType::K, DimType::N},
        SmallVector<DimType>{DimType::M, DimType::N, DimType::M, DimType::N}};
  } else if (linalgx::isGenericPackedMatmulOp(linalgOp.getOperation(),
                                              linalgx::PackingType::MM2D4D)) {
    return SmallVector<SmallVector<DimType>>{
        SmallVector<DimType>{DimType::M, DimType::K},
        SmallVector<DimType>{DimType::N, DimType::K, DimType::K, DimType::N},
        SmallVector<DimType>{DimType::M, DimType::N}};
  }
  return failure();
}

// The analysis to extract the matmul configuration from the given linalg op
struct MatmulConfigAnalysis {
public:
  // Extract the matmul configuration from the given linalg op
  MatmulConfigAnalysis(Operation *root) : root(root){};

  // Get the matmul configuration
  MatmulConfig getConfig();

  void setAllowIndivisibleInnerBlock(bool allow) {
    allowIndivisibleInnerBlock = allow;
  }

private:
  MatmulConfig config = MatmulConfig{1, 1, 1, 1, 1, 1, 1, 1, 1};
  Operation *root;
  bool hasConfig = false;
  bool allowIndivisibleInnerBlock = true;
};

} // namespace gc
} // namespace mlir

#endif