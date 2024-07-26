
//===-- AllInterfaces.h - linalgx dialect interfaces ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_LINALGX_TRANSFORMS_ALLINTERFACES_H
#define DIALECT_LINALGX_TRANSFORMS_ALLINTERFACES_H

namespace mlir {
class DialectRegistry;

namespace linalgx {
void registerAllDialectInterfaceImplementations(DialectRegistry &registry);
} // namespace linalgx

} // namespace mlir

#endif // DIALECT_LINALGX_TRANSFORMS_ALLINTERFACES_H
