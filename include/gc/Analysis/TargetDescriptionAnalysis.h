//===-- TargetDescriptionAnalysis.h - target description class --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_TARGETDESCRIPTIONANALYSIS_H
#define MLIR_ANALYSIS_TARGETDESCRIPTIONANALYSIS_H

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace gc {

using namespace mlir;

enum DeviceType { CPU = 0 };

class TargetDescriptionAnalysisBase {
public:
  TargetDescriptionAnalysisBase(Operation *op, DeviceType device)
      : ctx(op->getContext()), device(device),
        layout(isa<ModuleOp>(op) ? dyn_cast<ModuleOp>(op)
                                 : op->getParentOfType<ModuleOp>()),
        loc(op->getLoc()) {}

  // get the device ID
  DeviceType getDevice() { return device; }

  // get the MLIR context
  MLIRContext *getContext() { return ctx; }

  // get the data layout
  DataLayout getLayout() { return layout; }

  // get the property value by key
  std::optional<Attribute> getPropertyValue(StringRef key);

  // get the location
  Location getLocation() { return loc; }

  // check if the property exists
  bool hasProperty(StringRef key) { return getPropertyValue(key).has_value(); }

  // emit warning if the property is not found
  template <typename T>
  void emitNotFoundWarning(Location loc, StringRef key, T value);

  // the map from device type to device string
  static llvm::DenseMap<DeviceType, std::string> DeviceKeyMap;

  // set the emit warning flag
  void setEmitWarning(bool emit) { emitWarning = emit; }

private:
  MLIRContext *ctx;
  DeviceType device;
  DataLayout layout;
  Location loc;
  bool emitWarning = false;
};

class CPUTargetDescriptionAnalysis : public TargetDescriptionAnalysisBase {
public:
  static constexpr StringLiteral kL1CacheSize = "L1_cache_size_in_bytes";
  static constexpr StringLiteral kL2CacheSize = "L2_cache_size_in_bytes";
  static constexpr StringLiteral kL3CacheSize = "L3_cache_size_in_bytes";
  static constexpr StringLiteral kMaxVectorWidth = "max_vector_width";
  static constexpr StringLiteral kNumThreads = "num_threads";
  static constexpr StringLiteral kNumNode = "num_node";

  // get runtime OMP_NUM_THREADS
  unsigned getNumThreads();

  // get NUMA Node Num
  unsigned getNumNodes();

  // set the expected threads
  void limitOnSingleNode(uint32_t numa_node);

  // get cache size by cacheLevel
  unsigned getCacheSize(uint8_t cacheLevel);

  // get the maximum vector length in bits
  unsigned getMaxVectorWidth();

  CPUTargetDescriptionAnalysis(Operation *op)
      : TargetDescriptionAnalysisBase(op, DeviceType::CPU) {}

private:
  bool threads_limited = false;
  unsigned defaultNumThreads = 1;
};

} // namespace gc
} // namespace mlir

#endif