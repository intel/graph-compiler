//===- MicrokernelOps.h - microkernel dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_DIALECTS_MICROKERNELOPS_H
#define GC_DIALECTS_MICROKERNELOPS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "gc/Dialect/Microkernel/MicrokernelDialect.h"
#include "gc/Dialect/Microkernel/MicrokernelEnum.h"

#define GET_OP_CLASSES
#include "gc/Dialect/Microkernel/MicrokernelOps.h.inc"

#endif // GC_DIALECTS_MICROKERNELOPS_H
