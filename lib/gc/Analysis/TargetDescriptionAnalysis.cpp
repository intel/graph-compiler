//===-- TargetDescriptionAnalysis.cpp - target description impl -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Analysis/TargetDescriptionAnalysis.h"
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

// default values for properties
llvm::DenseMap<StringRef, int64_t>
    CPUTargetDescriptionAnalysis::CPUTargetDeafultValueMap = {
        {CPUTargetDescriptionAnalysis::kNumThreads, 1},
        {CPUTargetDescriptionAnalysis::kL1CacheSize, 32 * 1024},
        {CPUTargetDescriptionAnalysis::kL2CacheSize, 1024 * 1024},
        {CPUTargetDescriptionAnalysis::kL3CacheSize, 32 * 1024 * 1024},
        {CPUTargetDescriptionAnalysis::kMaxVectorWidth, 512},
};

template <typename T>
void TargetDescriptionAnalysisBase::emitNotFoundWarning(Location loc,
                                                        StringRef key,
                                                        T value) {
  mlir::emitWarning(loc) << key << " not found, using default value " << value;
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

size_t CPUTargetDescriptionAnalysis::getNumThreads() {
  std::optional<Attribute> numThreads = getPropertyValue(kNumThreads);

  if (numThreads)
    return getIntFromAttribute(*numThreads);
  emitNotFoundWarning(getLocation(), kNumThreads,
                      CPUTargetDeafultValueMap[kNumThreads]);
  return CPUTargetDeafultValueMap[kNumThreads];
}

size_t CPUTargetDescriptionAnalysis::getCacheSize(uint8_t cacheLevel) {
  assert(cacheLevel > 0 && cacheLevel < 4 && "Invalid cache level");
  StringLiteral key = "";
  if (cacheLevel == 1)
    key = kL1CacheSize;
  else if (cacheLevel == 2)
    key = kL2CacheSize;
  else if (cacheLevel == 3)
    key = kL3CacheSize;

  std::optional<Attribute> cacheSize = getPropertyValue(key);
  if (cacheSize)
    return getIntFromAttribute(*cacheSize);

  emitNotFoundWarning(getLocation(), key, CPUTargetDeafultValueMap[key]);
  return CPUTargetDeafultValueMap[key];
}

size_t CPUTargetDescriptionAnalysis::getMaxVectorWidth() {
  std::optional<Attribute> maxVectorWidth = getPropertyValue(kMaxVectorWidth);
  if (maxVectorWidth)
    return getIntFromAttribute(*maxVectorWidth);
  emitNotFoundWarning(getLocation(), kMaxVectorWidth,
                      CPUTargetDeafultValueMap[kMaxVectorWidth]);
  return CPUTargetDeafultValueMap[kMaxVectorWidth];
}

} // namespace gc
} // namespace mlir