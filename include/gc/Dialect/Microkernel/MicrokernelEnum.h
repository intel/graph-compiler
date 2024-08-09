//===- MicrokernelEnum.h - microkernel dialect enums ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_DIALECTS_MICROKERNELENUM_H
#define GC_DIALECTS_MICROKERNELENUM_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "gc/Dialect/Microkernel/MicrokernelEnum.h.inc"

#endif // GC_DIALECTS_MICROKERNELENUM_H
