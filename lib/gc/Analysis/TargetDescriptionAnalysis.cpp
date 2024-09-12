//===-- TargetDescriptionAnalysis.cpp - target description impl -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Analysis/TargetDescriptionAnalysis.h"
#include "mlir/Support/LLVM.h"
#include <limits>
#include <llvm/Support/Debug.h>
#include <regex>

namespace mlir {
namespace gc {

#define DEBUG_TYPE "target-description-analysis"

llvm::DenseMap<DeviceType, std::string>
    TargetDescriptionAnalysisBase::DeviceKeyMap = {
        {CPU, "CPU"},
};

template <typename T>
void TargetDescriptionAnalysisBase::emitNotFoundWarning(Location loc,
                                                        StringRef key,
                                                        T value) {
  if (emitWarning)
    mlir::emitWarning(loc) << key << " not found, using default value "
                           << value;
}

static bool isIntegerNumber(const std::string &token) {
  return std::regex_match(token, std::regex(("(\\+|-)?[[:digit:]]+")));
}

static int64_t getIntFromAttribute(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    if (intAttr.getType().isSignedInteger())
      return intAttr.getSInt();
    else if (intAttr.getType().isUnsignedInteger())
      return intAttr.getUInt();
    else
      return intAttr.getInt();
  } else if (auto strAttr = dyn_cast<StringAttr>(attr)) {
    std::string str = strAttr.getValue().str();
    if (isIntegerNumber(str))
      return std::stoll(str);
  }
  llvm_unreachable("Not an integer attribute or integer like string attribute");
}

std::optional<Attribute>
TargetDescriptionAnalysisBase::getPropertyValue(StringRef key) {
  return layout.getDevicePropertyValue(
      Builder(getContext())
          .getStringAttr(DeviceKeyMap[getDevice()] /* device ID*/),
      Builder(getContext()).getStringAttr(key));
}

unsigned CPUTargetDescriptionAnalysis::getNumThreads() const {
  char *numThreads = getenv("OMP_NUM_THREADS");
  if (numThreads) {
    return std::stoi(numThreads);
  }
  return 1;
}

unsigned CPUTargetDescriptionAnalysis::getCacheSize(uint8_t cacheLevel) const {
  assert(cacheLevel > 0 && cacheLevel < 4 && "Invalid cache level");
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
}

unsigned CPUTargetDescriptionAnalysis::getMaxVectorWidth() const {
  static const unsigned defaultMaxVectorWidth = 512;
  return defaultMaxVectorWidth;
}

} // namespace gc
} // namespace mlir