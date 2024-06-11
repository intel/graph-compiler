//===- BufferizableOpInterfaceImpl.h - linalgx Bufferize --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_LINALGX_BUFFERIZABLEOPINTERFACEIMPL_H
#define DIALECT_LINALGX_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace linalgx {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace linalgx
} // namespace mlir

#endif // DIALECT_LINALGX_BUFFERIZABLEOPINTERFACEIMPL_H
