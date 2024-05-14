//===- LegalizeUtils.h - Utils for LegalizeDtypeToF32 Pass ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LEGALIZE_UTILS_H
#define LEGALIZE_UTILS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ConversionTarget;
class RewritePatternSet;
class TypeConverter;

namespace gc {

struct CPUFlags {
  bool fAVX512FP16 = true;
};
template <typename T>
void populateLegalizeDTypeToF32TypeConverter(TypeConverter &typeConverter);
void populateBfloat16ToF32ConversionTarget(ConversionTarget &target,
                                           TypeConverter &typeConverter);
void populateFloat16ToF32ConversionTarget(ConversionTarget &target,
                                          TypeConverter &typeConverter);
void populateLegalizeDTypeToF32Patterns(RewritePatternSet &patterns,
                                        TypeConverter &typeConverter);
} // namespace gc
} // namespace mlir

#endif
