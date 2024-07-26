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
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace mlir {
namespace gc {

using namespace mlir;

struct SystemDesc {
  // get runtime OMP_NUM_THREADS
  uint32_t getNumThreads() {
    std::optional<Attribute> numThreads = layout.getDevicePropertyValue(
        Builder(ctx).getStringAttr("CPU" /* device ID*/),
        Builder(ctx).getStringAttr("num_threads"));
    if (numThreads && isa<IntegerAttr>(*numThreads)) {
      return dyn_cast<IntegerAttr>(*numThreads).getInt();
    }
    return 1;
  }
  // get cache size by cacheLevel
  size_t getCacheSize(uint8_t cacheLevel) {
    if (cacheLevel == 1) {
      std::optional<Attribute> cacheSize = layout.getDevicePropertyValue(
          Builder(ctx).getStringAttr("CPU" /* device ID*/),
          Builder(ctx).getStringAttr("L1_cache_size_in_bytes"));
      if (cacheSize && isa<IntegerAttr>(*cacheSize)) {
        return dyn_cast<IntegerAttr>(*cacheSize).getInt();
      }
    } else if (cacheLevel == 2) {
      std::optional<Attribute> cacheSize = layout.getDevicePropertyValue(
          Builder(ctx).getStringAttr("CPU" /* device ID*/),
          Builder(ctx).getStringAttr("L2_cache_size_in_bytes"));
      if (cacheSize && isa<IntegerAttr>(*cacheSize)) {
        return dyn_cast<IntegerAttr>(*cacheSize).getInt();
      }
    } else if (cacheLevel == 3) {
      std::optional<Attribute> cacheSize = layout.getDevicePropertyValue(
          Builder(ctx).getStringAttr("CPU" /* device ID*/),
          Builder(ctx).getStringAttr("L3_cache_size_in_bytes"));
      if (cacheSize && isa<IntegerAttr>(*cacheSize)) {
        return dyn_cast<IntegerAttr>(*cacheSize).getInt();
      }
    }
    return 0;
  }

  // get the maximum vector length in bits
  size_t getMaxVectorLength() {
    std::optional<Attribute> maxVectorLength = layout.getDevicePropertyValue(
        Builder(ctx).getStringAttr("CPU" /* device ID*/),
        Builder(ctx).getStringAttr("max_vector_width"));
    if (maxVectorLength && isa<IntegerAttr>(*maxVectorLength)) {
      return dyn_cast<IntegerAttr>(*maxVectorLength).getInt();
    }
    return 512;
  }

  SystemDesc(ModuleOp m) : layout(m), ctx(m->getContext()) {}

private:
  DataLayout layout;
  MLIRContext *ctx;
};

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